import collections
from typing import Any, Dict, Iterator, List, Optional

import torch
from transformers import AutoModel
from transformers.activations import ClippedGELUActivation, GELUActivation
from transformers.modeling_utils import PoolerEndLogits

from relik.reader.data.relik_reader_sample import RelikReaderSample

activation2functions = {
    "relu": torch.nn.ReLU(),
    "gelu": GELUActivation(),
    "gelu_10": ClippedGELUActivation(-10, 10),
}


class RelikReaderCoreModel(torch.nn.Module):
    def __init__(
        self,
        transformer_model: str,
        additional_special_symbols: int,
        num_layers: Optional[int] = None,
        activation: str = "gelu",
        linears_hidden_size: Optional[int] = 512,
        use_last_k_layers: int = 1,
        training: bool = False,
    ) -> None:
        super().__init__()

        # Transformer model declaration
        self.transformer_model_name = transformer_model
        self.transformer_model = (
            AutoModel.from_pretrained(transformer_model)
            if num_layers is None
            else AutoModel.from_pretrained(
                transformer_model, num_hidden_layers=num_layers
            )
        )
        self.transformer_model.resize_token_embeddings(
            self.transformer_model.config.vocab_size + additional_special_symbols
        )

        self.activation = activation
        self.linears_hidden_size = linears_hidden_size
        self.use_last_k_layers = use_last_k_layers

        # named entity detection layers
        self.ned_start_classifier = self._get_projection_layer(
            self.activation, last_hidden=2, layer_norm=False
        )
        self.ned_end_classifier = PoolerEndLogits(self.transformer_model.config)

        # END entity disambiguation layer
        self.ed_start_projector = self._get_projection_layer(self.activation)
        self.ed_end_projector = self._get_projection_layer(self.activation)

        self.training = training

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
                self.transformer_model.config.hidden_size * self.use_last_k_layers
                if input_hidden is None
                else input_hidden,
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
            expanded_features = torch.cat(
                [
                    model_features[i].unsqueeze(0).expand(x, -1, -1)
                    for i, x in enumerate(torch.sum(start_positions > 0, dim=-1))
                    if x > 0
                ],
                dim=0,
            ).to(start_positions_indices.device)

            expanded_prediction_mask = torch.cat(
                [
                    prediction_mask[i].unsqueeze(0).expand(x, -1)
                    for i, x in enumerate(torch.sum(start_positions > 0, dim=-1))
                    if x > 0
                ],
                dim=0,
            ).to(expanded_features.device)

            end_logits = self.ned_end_classifier(
                hidden_states=expanded_features,
                start_positions=start_positions_indices,
                p_mask=expanded_prediction_mask,
            )

            return end_logits

        return None

    def compute_classification_logits(
        self,
        model_features,
        special_symbols_mask,
        prediction_mask,
        batch_size,
        start_positions=None,
        end_positions=None,
    ) -> torch.Tensor:
        if start_positions is None or end_positions is None:
            start_positions = torch.zeros_like(prediction_mask)
            end_positions = torch.zeros_like(prediction_mask)

        model_start_features = self.ed_start_projector(model_features)
        model_end_features = self.ed_end_projector(model_features)
        model_end_features[start_positions > 0] = model_end_features[end_positions > 0]

        model_ed_features = torch.cat(
            [model_start_features, model_end_features], dim=-1
        )

        # computing ed features
        classes_representations = torch.sum(special_symbols_mask, dim=1)[0].item()
        special_symbols_representation = model_ed_features[special_symbols_mask].view(
            batch_size, classes_representations, -1
        )

        logits = torch.bmm(
            model_ed_features,
            torch.permute(special_symbols_representation, (0, 2, 1)),
        )

        logits = self._mask_logits(logits, prediction_mask)

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

        # named entity detection if required
        if use_predefined_spans:  # no need to compute spans
            ned_start_logits, ned_start_probabilities, ned_start_predictions = (
                None,
                None,
                torch.clone(start_labels)
                if start_labels is not None
                else torch.zeros_like(input_ids),
            )
            ned_end_logits, ned_end_probabilities, ned_end_predictions = (
                None,
                None,
                torch.clone(end_labels)
                if end_labels is not None
                else torch.zeros_like(input_ids),
            )

            ned_start_predictions[ned_start_predictions > 0] = 1
            ned_end_predictions[ned_end_predictions > 0] = 1

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
                ned_end_predictions = torch.argmax(ned_end_probabilities, dim=-1)
            else:
                ned_end_logits, ned_end_probabilities = None, None
                ned_end_predictions = ned_start_predictions.new_zeros(batch_size)

            # flattening end predictions
            #   (flattening can happen only if the
            #   end boundaries were not predicted using the gold labels)
            if not self.training:
                flattened_end_predictions = torch.clone(ned_start_predictions)
                flattened_end_predictions[flattened_end_predictions > 0] = 0

                batch_start_predictions = list()
                for elem_idx in range(batch_size):
                    batch_start_predictions.append(
                        torch.where(ned_start_predictions[elem_idx] > 0)[0].tolist()
                    )

                # check that the total number of start predictions
                # is equal to the end predictions
                total_start_predictions = sum(map(len, batch_start_predictions))
                total_end_predictions = len(ned_end_predictions)
                assert (
                    total_start_predictions == 0
                    or total_start_predictions == total_end_predictions
                ), (
                    f"Total number of start predictions = {total_start_predictions}. "
                    f"Total number of end predictions = {total_end_predictions}"
                )

                curr_end_pred_num = 0
                for elem_idx, bsp in enumerate(batch_start_predictions):
                    for sp in bsp:
                        ep = ned_end_predictions[curr_end_pred_num].item()
                        if ep < sp:
                            ep = sp

                        # if we already set this span throw it (no overlap)
                        if flattened_end_predictions[elem_idx, ep] == 1:
                            ned_start_predictions[elem_idx, sp] = 0
                        else:
                            flattened_end_predictions[elem_idx, ep] = 1

                        curr_end_pred_num += 1

                ned_end_predictions = flattened_end_predictions

        start_position, end_position = (
            (start_labels, end_labels)
            if self.training
            else (ned_start_predictions, ned_end_predictions)
        )

        # Entity disambiguation
        ed_logits = self.compute_classification_logits(
            model_features,
            special_symbols_mask,
            prediction_mask,
            batch_size,
            start_position,
            end_position,
        )
        ed_probabilities = torch.softmax(ed_logits, dim=-1)
        ed_predictions = torch.argmax(ed_probabilities, dim=-1)

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
            if ned_end_logits is not None:
                ned_end_labels = torch.zeros_like(end_labels)
                ned_end_labels[end_labels == -100] = -100
                ned_end_labels[end_labels > 0] = 1

                ned_end_loss = self.criterion(
                    ned_end_logits,
                    (
                        torch.arange(
                            ned_end_labels.size(1), device=ned_end_labels.device
                        )
                        .unsqueeze(0)
                        .expand(batch_size, -1)[ned_end_labels > 0]
                    ).to(ned_end_labels.device),
                )

            else:
                ned_end_loss = 0

            # entity disambiguation loss
            start_labels[ned_start_labels != 1] = -100
            ed_labels = torch.clone(start_labels)
            ed_labels[end_labels > 0] = end_labels[end_labels > 0]
            ed_loss = self.criterion(
                ed_logits.view(-1, ed_logits.shape[-1]),
                ed_labels.view(-1),
            )

            output_dict["ned_start_loss"] = ned_start_loss
            output_dict["ned_end_loss"] = ned_end_loss
            output_dict["ed_loss"] = ed_loss

            output_dict["loss"] = ned_start_loss + ned_end_loss + ed_loss

        return output_dict

    def batch_predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        prediction_mask: Optional[torch.Tensor] = None,
        special_symbols_mask: Optional[torch.Tensor] = None,
        sample: Optional[List[RelikReaderSample]] = None,
        top_k: int = 5,  # the amount of top-k most probable entities to predict
        *args,
        **kwargs,
    ) -> Iterator[RelikReaderSample]:
        forward_output = self.forward(
            input_ids,
            attention_mask,
            token_type_ids,
            prediction_mask,
            special_symbols_mask,
        )

        ned_start_predictions = forward_output["ned_start_predictions"].cpu().numpy()
        ned_end_predictions = forward_output["ned_end_predictions"].cpu().numpy()
        ed_predictions = forward_output["ed_predictions"].cpu().numpy()
        ed_probabilities = forward_output["ed_probabilities"].cpu().numpy()

        batch_predictable_candidates = kwargs["predictable_candidates"]
        patch_offset = kwargs["patch_offset"]
        for ts, ne_sp, ne_ep, edp, edpr, pred_cands, po in zip(
            sample,
            ned_start_predictions,
            ned_end_predictions,
            ed_predictions,
            ed_probabilities,
            batch_predictable_candidates,
            patch_offset,
        ):
            ne_start_indices = [ti for ti, c in enumerate(ne_sp[1:]) if c > 0]
            ne_end_indices = [ti for ti, c in enumerate(ne_ep[1:]) if c > 0]

            final_class2predicted_spans = collections.defaultdict(list)
            spans2predicted_probabilities = dict()
            for start_token_index, end_token_index in zip(
                ne_start_indices, ne_end_indices
            ):
                # predicted candidate
                token_class = edp[start_token_index + 1] - 1
                predicted_candidate_title = pred_cands[token_class]
                final_class2predicted_spans[predicted_candidate_title].append(
                    [start_token_index, end_token_index]
                )

                # candidates probabilities
                classes_probabilities = edpr[start_token_index + 1]
                classes_probabilities_best_indices = classes_probabilities.argsort()[
                    ::-1
                ]
                titles_2_probs = []
                top_k = (
                    min(
                        top_k,
                        len(classes_probabilities_best_indices),
                    )
                    if top_k != -1
                    else len(classes_probabilities_best_indices)
                )
                for i in range(top_k):
                    titles_2_probs.append(
                        (
                            pred_cands[classes_probabilities_best_indices[i] - 1],
                            classes_probabilities[
                                classes_probabilities_best_indices[i]
                            ].item(),
                        )
                    )
                spans2predicted_probabilities[
                    (start_token_index, end_token_index)
                ] = titles_2_probs

            if "patches" not in ts._d:
                ts._d["patches"] = dict()

            ts._d["patches"][po] = dict()
            sample_patch = ts._d["patches"][po]

            sample_patch["predicted_window_labels"] = final_class2predicted_spans
            sample_patch["span_title_probabilities"] = spans2predicted_probabilities

            # additional info
            sample_patch["predictable_candidates"] = pred_cands

            yield ts
