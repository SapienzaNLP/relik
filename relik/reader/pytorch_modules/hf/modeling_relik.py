from typing import Any, Dict, Optional

import torch
from transformers import AutoModel, PreTrainedModel
from transformers.activations import ClippedGELUActivation, GELUActivation
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PoolerEndLogits

from .configuration_relik import RelikReaderConfig


class RelikReaderSample:
    def __init__(self, **kwargs):
        super().__setattr__("_d", {})
        self._d = kwargs

    def __getattribute__(self, item):
        return super(RelikReaderSample, self).__getattribute__(item)

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            # this is likely some python library-specific variable (such as __deepcopy__ for copy)
            # better follow standard behavior here
            raise AttributeError(item)
        elif item in self._d:
            return self._d[item]
        else:
            return None

    def __setattr__(self, key, value):
        if key in self._d:
            self._d[key] = value
        else:
            super().__setattr__(key, value)
            self._d[key] = value


activation2functions = {
    "relu": torch.nn.ReLU(),
    "gelu": GELUActivation(),
    "gelu_10": ClippedGELUActivation(-10, 10),
}


class PoolerEndLogitsBi(PoolerEndLogits):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.dense_1 = torch.nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        if p_mask is not None:
            p_mask = p_mask.unsqueeze(-1)
        logits = super().forward(
            hidden_states,
            start_states,
            start_positions,
            p_mask,
        )
        return logits


class RelikReaderSpanModel(PreTrainedModel):
    config_class = RelikReaderConfig

    def __init__(self, config: RelikReaderConfig, *args, **kwargs):
        super().__init__(config)
        # Transformer model declaration
        self.config = config
        self.transformer_model = (
            AutoModel.from_pretrained(self.config.transformer_model)
            if self.config.num_layers is None
            else AutoModel.from_pretrained(
                self.config.transformer_model, num_hidden_layers=self.config.num_layers
            )
        )
        self.transformer_model.resize_token_embeddings(
            self.transformer_model.config.vocab_size
            + self.config.additional_special_symbols
        )

        self.activation = self.config.activation
        self.linears_hidden_size = self.config.linears_hidden_size
        self.use_last_k_layers = self.config.use_last_k_layers

        # named entity detection layers
        self.ned_start_classifier = self._get_projection_layer(
            self.activation, last_hidden=2, layer_norm=False
        )
        if self.config.binary_end_logits:
            self.ned_end_classifier = PoolerEndLogitsBi(self.transformer_model.config)
        else:
            self.ned_end_classifier = PoolerEndLogits(self.transformer_model.config)

        # END entity disambiguation layer
        self.ed_start_projector = self._get_projection_layer(self.activation)
        self.ed_end_projector = self._get_projection_layer(self.activation)

        self.training = self.config.training

        # criterion
        self.criterion = torch.nn.CrossEntropyLoss()

    def _get_projection_layer(
        self,
        activation: str,
        last_hidden: Optional[int] = None,
        input_hidden=None,
        layer_norm: bool = True,
    ) -> torch.nn.Sequential:
        head_components = [
            torch.nn.Dropout(0.1),
            torch.nn.Linear(
                (
                    self.transformer_model.config.hidden_size * self.use_last_k_layers
                    if input_hidden is None
                    else input_hidden
                ),
                self.linears_hidden_size,
            ),
            activation2functions[activation],
            torch.nn.Dropout(0.1),
            torch.nn.Linear(
                self.linears_hidden_size,
                self.linears_hidden_size if last_hidden is None else last_hidden,
            ),
        ]

        if layer_norm:
            head_components.append(
                torch.nn.LayerNorm(
                    self.linears_hidden_size if last_hidden is None else last_hidden,
                    self.transformer_model.config.layer_norm_eps,
                )
            )

        return torch.nn.Sequential(*head_components)

    def _mask_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        if next(self.parameters()).dtype == torch.float16:
            logits = logits * (1 - mask) - 65500 * mask
        else:
            logits = logits * (1 - mask) - 1e30 * mask
        return logits

    def _get_model_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
    ):
        model_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": self.use_last_k_layers > 1,
        }

        if token_type_ids is not None:
            model_input["token_type_ids"] = token_type_ids

        model_output = self.transformer_model(**model_input)

        if self.use_last_k_layers > 1:
            model_features = torch.cat(
                model_output[1][-self.use_last_k_layers :], dim=-1
            )
        else:
            model_features = model_output[0]

        return model_features

    def compute_ned_end_logits(
        self,
        start_predictions,
        start_labels,
        model_features,
        prediction_mask,
        batch_size,
    ) -> Optional[torch.Tensor]:
        # todo: maybe when constraining on the spans,
        #  we should not use a prediction_mask for the end tokens.
        #  at least we should not during training imo
        start_positions = start_labels if self.training else start_predictions
        start_positions_indices = (
            torch.arange(start_positions.size(1), device=start_positions.device)
            .unsqueeze(0)
            .expand(batch_size, -1)[start_positions > 0]
        ).to(start_positions.device)

        if len(start_positions_indices) > 0:
            expanded_features = model_features.repeat_interleave(
                torch.sum(start_positions > 0, dim=-1), dim=0
            )
            expanded_prediction_mask = prediction_mask.repeat_interleave(
                torch.sum(start_positions > 0, dim=-1), dim=0
            )
            end_logits = self.ned_end_classifier(
                hidden_states=expanded_features,
                start_positions=start_positions_indices,
                p_mask=expanded_prediction_mask,
            )

            return end_logits

        return None

    def compute_classification_logits(
        self,
        model_features_start,
        model_features_end,
        special_symbols_features,
    ) -> torch.Tensor:
        model_start_features = self.ed_start_projector(model_features_start)
        model_end_features = self.ed_end_projector(model_features_end)
        model_start_features_symbols = self.ed_start_projector(special_symbols_features)
        model_end_features_symbols = self.ed_end_projector(special_symbols_features)

        model_ed_features = torch.cat(
            [model_start_features, model_end_features], dim=-1
        )
        special_symbols_representation = torch.cat(
            [model_start_features_symbols, model_end_features_symbols], dim=-1
        )

        logits = torch.bmm(
            model_ed_features,
            torch.permute(special_symbols_representation, (0, 2, 1)),
        )

        logits = self._mask_logits(logits, (model_features_start == -100).all(2).long())
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        prediction_mask: Optional[torch.Tensor] = None,
        special_symbols_mask: Optional[torch.Tensor] = None,
        start_labels: Optional[torch.Tensor] = None,
        end_labels: Optional[torch.Tensor] = None,
        use_predefined_spans: bool = False,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape

        model_features = self._get_model_features(
            input_ids, attention_mask, token_type_ids
        )

        ned_start_labels = None

        # named entity detection if required
        if use_predefined_spans:  # no need to compute spans
            ned_start_logits, ned_start_probabilities, ned_start_predictions = (
                None,
                None,
                (
                    torch.clone(start_labels)
                    if start_labels is not None
                    else torch.zeros_like(input_ids)
                ),
            )
            ned_end_logits, ned_end_probabilities, ned_end_predictions = (
                None,
                None,
                (
                    torch.clone(end_labels)
                    if end_labels is not None
                    else torch.zeros_like(input_ids)
                ),
            )
            ned_start_predictions[ned_start_predictions > 0] = 1
            ned_end_predictions[end_labels > 0] = 1
            ned_end_predictions = ned_end_predictions[~(end_labels == -100).all(2)]

        else:  # compute spans
            # start boundary prediction
            ned_start_logits = self.ned_start_classifier(model_features)
            ned_start_logits = self._mask_logits(ned_start_logits, prediction_mask)
            ned_start_probabilities = torch.softmax(ned_start_logits, dim=-1)
            ned_start_predictions = ned_start_probabilities.argmax(dim=-1)

            # end boundary prediction
            ned_start_labels = (
                torch.zeros_like(start_labels) if start_labels is not None else None
            )

            if ned_start_labels is not None:
                ned_start_labels[start_labels == -100] = -100
                ned_start_labels[start_labels > 0] = 1

            ned_end_logits = self.compute_ned_end_logits(
                ned_start_predictions,
                ned_start_labels,
                model_features,
                prediction_mask,
                batch_size,
            )

            if ned_end_logits is not None:
                ned_end_probabilities = torch.softmax(ned_end_logits, dim=-1)
                if not self.config.binary_end_logits:
                    ned_end_predictions = torch.argmax(
                        ned_end_probabilities, dim=-1, keepdim=True
                    )
                    ned_end_predictions = torch.zeros_like(
                        ned_end_probabilities
                    ).scatter_(1, ned_end_predictions, 1)
                else:
                    ned_end_predictions = torch.argmax(ned_end_probabilities, dim=-1)
            else:
                ned_end_logits, ned_end_probabilities = None, None
                ned_end_predictions = ned_start_predictions.new_zeros(
                    batch_size, seq_len
                )

            if not self.training:
                # if len(ned_end_predictions.shape) < 2:
                #     print(ned_end_predictions)
                end_preds_count = ned_end_predictions.sum(1)
                # If there are no end predictions for a start prediction, remove the start prediction
                if (end_preds_count == 0).any() and (ned_start_predictions > 0).any():
                    ned_start_predictions[ned_start_predictions == 1] = (
                        end_preds_count != 0
                    ).long()
                    ned_end_predictions = ned_end_predictions[end_preds_count != 0]

        if end_labels is not None:
            end_labels = end_labels[~(end_labels == -100).all(2)]

        start_position, end_position = (
            (start_labels, end_labels)
            if self.training
            else (ned_start_predictions, ned_end_predictions)
        )
        start_counts = (start_position > 0).sum(1)
        if (start_counts > 0).any():
            ned_end_predictions = ned_end_predictions.split(start_counts.tolist())
        # Entity disambiguation
        if (end_position > 0).sum() > 0:
            ends_count = (end_position > 0).sum(1)
            model_entity_start = torch.repeat_interleave(
                model_features[start_position > 0], ends_count, dim=0
            )
            model_entity_end = torch.repeat_interleave(
                model_features, start_counts, dim=0
            )[end_position > 0]
            ents_count = torch.nn.utils.rnn.pad_sequence(
                torch.split(ends_count, start_counts.tolist()),
                batch_first=True,
                padding_value=0,
            ).sum(1)

            model_entity_start = torch.nn.utils.rnn.pad_sequence(
                torch.split(model_entity_start, ents_count.tolist()),
                batch_first=True,
                padding_value=-100,
            )

            model_entity_end = torch.nn.utils.rnn.pad_sequence(
                torch.split(model_entity_end, ents_count.tolist()),
                batch_first=True,
                padding_value=-100,
            )

            ed_logits = self.compute_classification_logits(
                model_entity_start,
                model_entity_end,
                model_features[special_symbols_mask].view(
                    batch_size, -1, model_features.shape[-1]
                ),
            )
            ed_probabilities = torch.softmax(ed_logits, dim=-1)
            ed_predictions = torch.argmax(ed_probabilities, dim=-1)
        else:
            ed_logits, ed_probabilities, ed_predictions = (
                None,
                ned_start_predictions.new_zeros(batch_size, seq_len),
                ned_start_predictions.new_zeros(batch_size),
            )
        # output build
        output_dict = dict(
            batch_size=batch_size,
            ned_start_logits=ned_start_logits,
            ned_start_probabilities=ned_start_probabilities,
            ned_start_predictions=ned_start_predictions,
            ned_end_logits=ned_end_logits,
            ned_end_probabilities=ned_end_probabilities,
            ned_end_predictions=ned_end_predictions,
            ed_logits=ed_logits,
            ed_probabilities=ed_probabilities,
            ed_predictions=ed_predictions,
        )

        # compute loss if labels
        if start_labels is not None and end_labels is not None and self.training:
            # named entity detection loss

            # start
            if ned_start_logits is not None:
                ned_start_loss = self.criterion(
                    ned_start_logits.view(-1, ned_start_logits.shape[-1]),
                    ned_start_labels.view(-1),
                )
            else:
                ned_start_loss = 0

            # end
            # use ents_count to assign the labels to the correct positions i.e. using end_labels -> [[0,0,4,0], [0,0,0,2]] -> [4,2] (this is just an element, for batch we need to mask it with ents_count), ie -> [[4,2,-100,-100], [3,1,2,-100], [1,3,2,5]]

            if ned_end_logits is not None:
                ed_labels = end_labels.clone()
                ed_labels = torch.nn.utils.rnn.pad_sequence(
                    torch.split(ed_labels[ed_labels > 0], ents_count.tolist()),
                    batch_first=True,
                    padding_value=-100,
                )
                end_labels[end_labels > 0] = 1
                if not self.config.binary_end_logits:
                    # transform label to position in the sequence
                    end_labels = end_labels.argmax(dim=-1)
                    ned_end_loss = self.criterion(
                        ned_end_logits.view(-1, ned_end_logits.shape[-1]),
                        end_labels.view(-1),
                    )
                else:
                    ned_end_loss = self.criterion(
                        ned_end_logits.reshape(-1, ned_end_logits.shape[-1]),
                        end_labels.reshape(-1).long(),
                    )

                # entity disambiguation loss
                ed_loss = self.criterion(
                    ed_logits.view(-1, ed_logits.shape[-1]),
                    ed_labels.view(-1).long(),
                )

            else:
                ned_end_loss = 0
                ed_loss = 0

            output_dict["ned_start_loss"] = ned_start_loss
            output_dict["ned_end_loss"] = ned_end_loss
            output_dict["ed_loss"] = ed_loss

            output_dict["loss"] = ned_start_loss + ned_end_loss + ed_loss

        return output_dict


class RelikReaderREModel(PreTrainedModel):
    config_class = RelikReaderConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        # Transformer model declaration
        # self.transformer_model_name = transformer_model
        self.config = config
        self.transformer_model = (
            AutoModel.from_pretrained(config.transformer_model)
            if config.num_layers is None
            else AutoModel.from_pretrained(
                config.transformer_model, num_hidden_layers=config.num_layers
            )
        )
        self.transformer_model.resize_token_embeddings(
            self.transformer_model.config.vocab_size
            + config.additional_special_symbols
            + config.additional_special_symbols_types,
        )

        # named entity detection layers
        self.ned_start_classifier = self._get_projection_layer(
            config.activation, last_hidden=2, layer_norm=False
        )

        self.ned_end_classifier = PoolerEndLogitsBi(self.transformer_model.config)

        self.relation_disambiguation_loss = (
            config.relation_disambiguation_loss
            if hasattr(config, "relation_disambiguation_loss")
            else False
        )

        if self.config.entity_type_loss and self.config.add_entity_embedding:
            input_hidden_ents = 3
        else:
            input_hidden_ents = 2

        self.re_projector = self._get_projection_layer(
            config.activation,
            input_hidden=input_hidden_ents * self.transformer_model.config.hidden_size,
            hidden=input_hidden_ents * self.config.linears_hidden_size,
            last_hidden=2 * self.config.linears_hidden_size,
        )

        self.re_relation_projector = self._get_projection_layer(
            config.activation,
            input_hidden=self.transformer_model.config.hidden_size,
        )

        if self.config.entity_type_loss or self.relation_disambiguation_loss:
            self.re_entities_projector = self._get_projection_layer(
                config.activation,
                input_hidden=2 * self.transformer_model.config.hidden_size,
            )
            self.re_definition_projector = self._get_projection_layer(
                config.activation,
            )

        self.re_classifier = self._get_projection_layer(
            config.activation,
            input_hidden=config.linears_hidden_size,
            last_hidden=2,
            layer_norm=False,
        )

        self.training = config.training

        # criterion
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_type = torch.nn.BCEWithLogitsLoss()

    def _get_projection_layer(
        self,
        activation: str,
        last_hidden: Optional[int] = None,
        hidden: Optional[int] = None,
        input_hidden=None,
        layer_norm: bool = True,
    ) -> torch.nn.Sequential:
        head_components = [
            torch.nn.Dropout(0.1),
            torch.nn.Linear(
                (
                    self.transformer_model.config.hidden_size
                    * self.config.use_last_k_layers
                    if input_hidden is None
                    else input_hidden
                ),
                self.config.linears_hidden_size if hidden is None else hidden,
            ),
            activation2functions[activation],
            torch.nn.Dropout(0.1),
            torch.nn.Linear(
                self.config.linears_hidden_size if hidden is None else hidden,
                self.config.linears_hidden_size if last_hidden is None else last_hidden,
            ),
        ]

        if layer_norm:
            head_components.append(
                torch.nn.LayerNorm(
                    (
                        self.config.linears_hidden_size
                        if last_hidden is None
                        else last_hidden
                    ),
                    self.transformer_model.config.layer_norm_eps,
                )
            )

        return torch.nn.Sequential(*head_components)

    def _mask_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        if next(self.parameters()).dtype == torch.float16:
            logits = logits * (1 - mask) - 65500 * mask
        else:
            logits = logits * (1 - mask) - 1e30 * mask
        return logits

    def _get_model_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
    ):
        model_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": self.config.use_last_k_layers > 1,
        }

        if token_type_ids is not None:
            model_input["token_type_ids"] = token_type_ids

        model_output = self.transformer_model(**model_input)

        if self.config.use_last_k_layers > 1:
            model_features = torch.cat(
                model_output[1][-self.config.use_last_k_layers :], dim=-1
            )
        else:
            model_features = model_output[0]

        return model_features

    def compute_ned_end_logits(
        self,
        start_predictions,
        start_labels,
        model_features,
        prediction_mask,
        batch_size,
        mask_preceding: bool = False,
    ) -> Optional[torch.Tensor]:
        # todo: maybe when constraining on the spans,
        #  we should not use a prediction_mask for the end tokens.
        #  at least we should not during training imo
        start_positions = start_labels if self.training else start_predictions
        start_positions_indices = (
            torch.arange(start_positions.size(1), device=start_positions.device)
            .unsqueeze(0)
            .expand(batch_size, -1)[start_positions > 0]
        ).to(start_positions.device)

        if len(start_positions_indices) > 0:
            expanded_features = model_features.repeat_interleave(
                torch.sum(start_positions > 0, dim=-1), dim=0
            )
            expanded_prediction_mask = prediction_mask.repeat_interleave(
                torch.sum(start_positions > 0, dim=-1), dim=0
            )
            if mask_preceding:
                expanded_prediction_mask[
                    torch.arange(
                        expanded_prediction_mask.shape[1],
                        device=expanded_prediction_mask.device,
                    )
                    < start_positions_indices.unsqueeze(1)
                ] = 1
            end_logits = self.ned_end_classifier(
                hidden_states=expanded_features,
                start_positions=start_positions_indices,
                p_mask=expanded_prediction_mask,
            )

            return end_logits

        return None

    def compute_relation_logits(
        self,
        model_entity_features,
        special_symbols_features,
    ) -> torch.Tensor:
        model_subject_object_features = self.re_projector(model_entity_features)
        model_subject_features = model_subject_object_features[
            :, :, : model_subject_object_features.shape[-1] // 2
        ]
        model_object_features = model_subject_object_features[
            :, :, model_subject_object_features.shape[-1] // 2 :
        ]
        special_symbols_start_representation = self.re_relation_projector(
            special_symbols_features
        )
        re_logits = torch.einsum(
            "bse,bde,bfe->bsdfe",
            model_subject_features,
            model_object_features,
            special_symbols_start_representation,
        )
        re_logits = self.re_classifier(re_logits)

        return re_logits

    def compute_entity_logits(
        self,
        model_entity_features,
        special_symbols_features,
    ) -> torch.Tensor:
        model_ed_features = self.re_entities_projector(model_entity_features)
        special_symbols_ed_representation = self.re_definition_projector(
            special_symbols_features
        )

        logits = torch.bmm(
            model_ed_features,
            torch.permute(special_symbols_ed_representation, (0, 2, 1)),
        )
        logits = self._mask_logits(
            logits, (model_entity_features == -100).all(2).long()
        )
        return logits

    def compute_loss(self, logits, labels, mask=None):
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1).long()
        if mask is not None:
            return self.criterion(logits[mask], labels[mask])
        return self.criterion(logits, labels)

    def compute_ned_type_loss(
        self,
        disambiguation_labels,
        re_ned_entities_logits,
        ned_type_logits,
        re_entities_logits,
        entity_types,
        mask,
    ):
        if self.config.entity_type_loss and self.relation_disambiguation_loss:
            return self.criterion_type(
                re_ned_entities_logits[disambiguation_labels != -100],
                disambiguation_labels[disambiguation_labels != -100],
            )
        if self.config.entity_type_loss:
            return self.criterion_type(
                ned_type_logits[mask],
                disambiguation_labels[:, :, :entity_types][mask],
            )

        if self.relation_disambiguation_loss:
            return self.criterion_type(
                re_entities_logits[disambiguation_labels != -100],
                disambiguation_labels[disambiguation_labels != -100],
            )
        return 0

    def compute_relation_loss(self, relation_labels, re_logits):
        return self.compute_loss(
            re_logits, relation_labels, relation_labels.view(-1) != -100
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        prediction_mask: Optional[torch.Tensor] = None,
        special_symbols_mask: Optional[torch.Tensor] = None,
        special_symbols_mask_entities: Optional[torch.Tensor] = None,
        start_labels: Optional[torch.Tensor] = None,
        end_labels: Optional[torch.Tensor] = None,
        disambiguation_labels: Optional[torch.Tensor] = None,
        relation_labels: Optional[torch.Tensor] = None,
        relation_threshold: float = None,
        is_validation: bool = False,
        is_prediction: bool = False,
        use_predefined_spans: bool = False,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        relation_threshold = (
            self.config.threshold if relation_threshold is None else relation_threshold
        )

        batch_size = input_ids.shape[0]

        model_features = self._get_model_features(
            input_ids, attention_mask, token_type_ids
        )

        # named entity detection
        if use_predefined_spans:
            ned_start_logits, ned_start_probabilities, ned_start_predictions = (
                None,
                None,
                torch.zeros_like(start_labels),
            )
            ned_end_logits, ned_end_probabilities, ned_end_predictions = (
                None,
                None,
                torch.zeros_like(end_labels),
            )

            ned_start_predictions[start_labels > 0] = 1
            ned_end_predictions[end_labels > 0] = 1
            ned_end_predictions = ned_end_predictions[~(end_labels == -100).all(2)]
            ned_start_labels = start_labels
            ned_start_labels[start_labels > 0] = 1
        else:
            # start boundary prediction
            ned_start_logits = self.ned_start_classifier(model_features)
            if is_validation or is_prediction:
                ned_start_logits = self._mask_logits(
                    ned_start_logits, prediction_mask
                )  # why?
            ned_start_probabilities = torch.softmax(ned_start_logits, dim=-1)
            ned_start_predictions = ned_start_probabilities.argmax(dim=-1)

            # end boundary prediction
            ned_start_labels = (
                torch.zeros_like(start_labels) if start_labels is not None else None
            )

            # start_labels contain entity id at their position, we just need 1 for start of entity
            if ned_start_labels is not None:
                ned_start_labels[start_labels == -100] = -100
                ned_start_labels[start_labels > 0] = 1

            # compute end logits only if there are any start predictions.
            # For each start prediction, n end predictions are made
            ned_end_logits = self.compute_ned_end_logits(
                ned_start_predictions,
                ned_start_labels,
                model_features,
                prediction_mask,
                batch_size,
                True,
            )

            if ned_end_logits is not None:
                # For each start prediction, n end predictions are made based on
                # binary classification ie. argmax at each position.
                ned_end_probabilities = torch.softmax(ned_end_logits, dim=-1)
                ned_end_predictions = ned_end_probabilities.argmax(dim=-1)
            else:
                ned_end_logits, ned_end_probabilities = None, None
                ned_end_predictions = torch.zeros_like(ned_start_predictions)

            if is_prediction or is_validation:
                end_preds_count = ned_end_predictions.sum(1)
                # If there are no end predictions for a start prediction, remove the start prediction
                if (end_preds_count == 0).any() and (ned_start_predictions > 0).any():
                    ned_start_predictions[ned_start_predictions == 1] = (
                        end_preds_count != 0
                    ).long()
                    ned_end_predictions = ned_end_predictions[end_preds_count != 0]

        if end_labels is not None:
            end_labels = end_labels[~(end_labels == -100).all(2)]

        start_position, end_position = (
            (start_labels, end_labels)
            if (not is_prediction and not is_validation)
            else (ned_start_predictions, ned_end_predictions)
        )

        start_counts = (start_position > 0).sum(1)
        if (start_counts > 0).any():
            ned_end_predictions = ned_end_predictions.split(start_counts.tolist())
        else:
            ned_end_predictions = [torch.empty(0, input_ids.shape[1], dtype=torch.int64) for _ in range(batch_size)]
        # limit to 30 predictions per document using start_counts, by setting all po after sum is 30 to 0
        # if is_validation or is_prediction:
        #     ned_start_predictions[ned_start_predictions == 1] = start_counts
        # We can only predict relations if we have start and end predictions
        if (end_position > 0).sum() > 0:
            ends_count = (end_position > 0).sum(1)
            model_subject_features = torch.cat(
                [
                    torch.repeat_interleave(
                        model_features[start_position > 0], ends_count, dim=0
                    ),  # start position features
                    torch.repeat_interleave(model_features, start_counts, dim=0)[
                        end_position > 0
                    ],  # end position features
                ],
                dim=-1,
            )
            ents_count = torch.nn.utils.rnn.pad_sequence(
                torch.split(ends_count, start_counts.tolist()),
                batch_first=True,
                padding_value=0,
            ).sum(1)
            model_subject_features = torch.nn.utils.rnn.pad_sequence(
                torch.split(model_subject_features, ents_count.tolist()),
                batch_first=True,
                padding_value=-100,
            )

            # if is_validation or is_prediction:
            #     model_subject_features = model_subject_features[:, :30, :]

            # entity disambiguation. Here relation_disambiguation_loss would only be useful to
            # reduce the number of candidate relations for the next step, but currently unused.
            if self.config.entity_type_loss or self.relation_disambiguation_loss:
                (re_ned_entities_logits) = self.compute_entity_logits(
                    model_subject_features,
                    model_features[
                        special_symbols_mask | special_symbols_mask_entities
                    ].view(batch_size, -1, model_features.shape[-1]),
                )
                entity_types = torch.sum(special_symbols_mask_entities, dim=1)[0].item()
                ned_type_logits = re_ned_entities_logits[:, :, :entity_types]
                re_entities_logits = re_ned_entities_logits[:, :, entity_types:]

                if self.config.entity_type_loss:
                    ned_type_probabilities = torch.sigmoid(ned_type_logits)
                    ned_type_predictions = ned_type_probabilities.argmax(dim=-1)

                    if self.config.add_entity_embedding:
                        special_symbols_representation = model_features[
                            special_symbols_mask_entities
                        ].view(batch_size, entity_types, -1)

                        entities_representation = torch.einsum(
                            "bsp,bpe->bse",
                            ned_type_probabilities,
                            special_symbols_representation,
                        )
                        model_subject_features = torch.cat(
                            [model_subject_features, entities_representation], dim=-1
                        )
                re_entities_probabilities = torch.sigmoid(re_entities_logits)
                re_entities_predictions = re_entities_probabilities.round()
            else:
                (
                    ned_type_logits,
                    ned_type_probabilities,
                    re_entities_logits,
                    re_entities_probabilities,
                ) = (None, None, None, None)
                ned_type_predictions, re_entities_predictions = (
                    torch.zeros([batch_size, 1], dtype=torch.long).to(input_ids.device),
                    torch.zeros([batch_size, 1], dtype=torch.long).to(input_ids.device),
                )

            # Compute relation logits
            re_logits = self.compute_relation_logits(
                model_subject_features,
                model_features[special_symbols_mask].view(
                    batch_size, -1, model_features.shape[-1]
                ),
            )

            re_probabilities = torch.softmax(re_logits, dim=-1)
            # we set a thresshold instead of argmax in cause it needs to be tweaked
            re_predictions = re_probabilities[:, :, :, :, 1] > relation_threshold
            re_probabilities = re_probabilities[:, :, :, :, 1]
        else:
            (
                ned_type_logits,
                ned_type_probabilities,
                re_entities_logits,
                re_entities_probabilities,
            ) = (None, None, None, None)
            ned_type_predictions, re_entities_predictions = (
                torch.zeros([batch_size, 1], dtype=torch.long).to(input_ids.device),
                torch.zeros([batch_size, 1], dtype=torch.long).to(input_ids.device),
            )
            re_logits, re_probabilities, re_predictions = (
                torch.zeros(
                    [batch_size, 1, 1, special_symbols_mask.sum(1)[0]], dtype=torch.long
                ).to(input_ids.device),
                torch.zeros(
                    [batch_size, 1, 1, special_symbols_mask.sum(1)[0]], dtype=torch.long
                ).to(input_ids.device),
                torch.zeros(
                    [batch_size, 1, 1, special_symbols_mask.sum(1)[0]], dtype=torch.long
                ).to(input_ids.device),
            )

        # output build
        output_dict = dict(
            batch_size=batch_size,
            ned_start_logits=ned_start_logits,
            ned_start_probabilities=ned_start_probabilities,
            ned_start_predictions=ned_start_predictions,
            ned_end_logits=ned_end_logits,
            ned_end_probabilities=ned_end_probabilities,
            ned_end_predictions=ned_end_predictions,
            ned_type_logits=ned_type_logits,
            ned_type_probabilities=ned_type_probabilities,
            ned_type_predictions=ned_type_predictions,
            re_entities_logits=re_entities_logits,
            re_entities_probabilities=re_entities_probabilities,
            re_entities_predictions=re_entities_predictions,
            re_logits=re_logits,
            re_probabilities=re_probabilities,
            re_predictions=re_predictions,
        )

        if (
            start_labels is not None
            and end_labels is not None
            and relation_labels is not None
            and is_prediction is False
        ):
            ned_start_loss = self.compute_loss(ned_start_logits, ned_start_labels)
            end_labels[end_labels > 0] = 1
            ned_end_loss = self.compute_loss(ned_end_logits, end_labels)
            if self.config.entity_type_loss or self.relation_disambiguation_loss:
                ned_type_loss = self.compute_ned_type_loss(
                    disambiguation_labels,
                    re_ned_entities_logits,
                    ned_type_logits,
                    re_entities_logits,
                    entity_types,
                    (model_subject_features != -100).all(2),
                )
            relation_loss = self.compute_relation_loss(relation_labels, re_logits)
            # compute loss. We can skip the relation loss if we are in the first epochs (optional)
            if self.config.entity_type_loss or self.relation_disambiguation_loss:
                output_dict["loss"] = (
                    ned_start_loss + ned_end_loss + relation_loss + ned_type_loss
                ) / 4
                output_dict["ned_type_loss"] = ned_type_loss
            else:
                output_dict["loss"] = ((1 / 20) * (ned_start_loss + ned_end_loss)) + (
                    (9 / 10) * relation_loss
                )
            output_dict["ned_start_loss"] = ned_start_loss
            output_dict["ned_end_loss"] = ned_end_loss
            output_dict["re_loss"] = relation_loss

        return output_dict
