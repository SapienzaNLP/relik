import contextlib
import logging
from typing import Any, Dict, Iterator, List

import numpy as np
import torch
import transformers as tr
from lightning_fabric.utilities import move_data_to_device
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from relik.common.log import get_logger
from relik.common.torch_utils import get_autocast_context
from relik.common.utils import get_callable_from_string
from relik.inference.data.objects import AnnotationType
from relik.reader.data.relik_reader_sample import RelikReaderSample
from relik.reader.pytorch_modules.base import RelikReaderBase
from relik.retriever.pytorch_modules import PRECISION_MAP

logger = get_logger(__name__, level=logging.INFO)


class RelikReaderForTripletExtraction(RelikReaderBase):
    """
    A class for the RelikReader model for triplet extraction.

    Args:
        transformer_model (:obj:`str` or :obj:`transformers.PreTrainedModel` or :obj:`None`, `optional`):
            The transformer model to use. If `None`, the default model is used.
        additional_special_symbols (:obj:`int`, `optional`, defaults to 0):
            The number of additional special symbols to add to the tokenizer.
        num_layers (:obj:`int`, `optional`):
            The number of layers to use. If `None`, all layers are used.
        activation (:obj:`str`, `optional`, defaults to "gelu"):
            The activation function to use.
        linears_hidden_size (:obj:`int`, `optional`, defaults to 512):
            The hidden size of the linears.
        use_last_k_layers (:obj:`int`, `optional`, defaults to 1):
            The number of last layers to use.
        training (:obj:`bool`, `optional`, defaults to False):
            Whether the model is in training mode.
        device (:obj:`str` or :obj:`torch.device` or :obj:`None`, `optional`):
            The device to use. If `None`, the default device is used.
        tokenizer (:obj:`str` or :obj:`transformers.PreTrainedTokenizer` or :obj:`None`, `optional`):
            The tokenizer to use. If `None`, the default tokenizer is used.
        dataset (:obj:`IterableDataset` or :obj:`str` or :obj:`None`, `optional`):
            The dataset to use. If `None`, the default dataset is used.
        dataset_kwargs (:obj:`Dict[str, Any]` or :obj:`None`, `optional`):
            The keyword arguments to pass to the dataset class.
        default_reader_class (:obj:`str` or :obj:`transformers.PreTrainedModel` or :obj:`None`, `optional`):
            The default reader class to use. If `None`, the default reader class is used.
        **kwargs:
            Keyword arguments.
    """

    default_reader_class: str = (
        "relik.reader.pytorch_modules.hf.modeling_relik.RelikReaderREModel"
    )
    default_data_class: str = "relik.reader.data.relik_reader_re_data.RelikREDataset"

    def __init__(
        self,
        transformer_model: str | tr.PreTrainedModel | None = None,
        additional_special_symbols: int = 0,
        additional_special_symbols_types: int = 0,
        entity_type_loss: bool | None = None,
        add_entity_embedding: bool | None = None,
        num_layers: int | None = None,
        activation: str = "gelu",
        linears_hidden_size: int | None = 512,
        use_last_k_layers: int = 1,
        training: bool = False,
        device: str | torch.device | None = None,
        tokenizer: str | tr.PreTrainedTokenizer | None = None,
        dataset: IterableDataset | str | None = None,
        dataset_kwargs: Dict[str, Any] | None = None,
        default_reader_class: tr.PreTrainedModel | str | None = None,
        **kwargs,
    ):
        super().__init__(
            transformer_model=transformer_model,
            additional_special_symbols=additional_special_symbols,
            additional_special_symbols_types=additional_special_symbols_types,
            entity_type_loss=entity_type_loss,
            add_entity_embedding=add_entity_embedding,
            num_layers=num_layers,
            activation=activation,
            linears_hidden_size=linears_hidden_size,
            use_last_k_layers=use_last_k_layers,
            training=training,
            device=device,
            tokenizer=tokenizer,
            dataset=dataset,
            default_reader_class=default_reader_class,
            **kwargs,
        )
        # and instantiate the dataset class
        self.dataset = dataset
        if self.dataset is None and training is False:
            self.default_data_class = get_callable_from_string(self.default_data_class)
            default_data_kwargs = dict(
                dataset_path=None,
                materialize_samples=False,
                transformer_model=self.tokenizer,
                special_symbols=self.default_data_class.get_special_symbols_re(
                    self.relik_reader_model.config.additional_special_symbols,
                    use_nme=kwargs.get("use_nme_re", False),
                ),
                special_symbols_types=self.default_data_class.get_special_symbols(
                    self.relik_reader_model.config.additional_special_symbols_types - 1
                )
                if self.relik_reader_model.config.additional_special_symbols_types > 0
                else [],
                for_inference=True,
                use_nme=kwargs.get("use_nme", False),
            )
            # merge the default data kwargs with the ones passed to the model
            default_data_kwargs.update(dataset_kwargs or {})
            self.dataset = self.default_data_class(**default_data_kwargs)

    @torch.no_grad()
    @torch.inference_mode()
    def _read(
        self,
        samples: List[RelikReaderSample] | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        prediction_mask: torch.Tensor | None = None,
        special_symbols_mask: torch.Tensor | None = None,
        max_length: int = 2048,
        max_batch_size: int = 128,
        token_batch_size: int = 2048,
        precision: str = 32,
        annotation_type: AnnotationType = AnnotationType.CHAR,
        progress_bar: bool = False,
        *args: object,
        **kwargs: object,
    ) -> List[RelikReaderSample] | List[List[RelikReaderSample]]:
        """
        A wrapper around the forward method that returns the predicted labels for each sample.

        Args:
            samples (:obj:`List[RelikReaderSample]`, `optional`):
                The samples to read. If provided, `text` and `candidates` are ignored.
            input_ids (:obj:`torch.Tensor`, `optional`):
                The input ids of the text. If `samples` is provided, this is ignored.
            attention_mask (:obj:`torch.Tensor`, `optional`):
                The attention mask of the text. If `samples` is provided, this is ignored.
            token_type_ids (:obj:`torch.Tensor`, `optional`):
                The token type ids of the text. If `samples` is provided, this is ignored.
            prediction_mask (:obj:`torch.Tensor`, `optional`):
                The prediction mask of the text. If `samples` is provided, this is ignored.
            special_symbols_mask (:obj:`torch.Tensor`, `optional`):
                The special symbols mask of the text. If `samples` is provided, this is ignored.
            max_length (:obj:`int`, `optional`, defaults to 1000):
                The maximum length of the text.
            max_batch_size (:obj:`int`, `optional`, defaults to 128):
                The maximum batch size.
            token_batch_size (:obj:`int`, `optional`):
                The token batch size.
            progress_bar (:obj:`bool`, `optional`, defaults to False):
                Whether to show a progress bar.
            precision (:obj:`str`, `optional`, defaults to 32):
                The precision to use for the model.
            annotation_type (`AnnotationType`, `optional`, defaults to `AnnotationType.CHAR`):
                The type of annotation to return. If `char`, the spans will be in terms of
                character offsets. If `word`, the spans will be in terms of word offsets.
            *args:
                Positional arguments.
            **kwargs:
                Keyword arguments.

        Returns:
            :obj:`List[RelikReaderSample]` or :obj:`List[List[RelikReaderSample]]`:
                The predicted labels for each sample.
        """

        precision = precision or self.precision
        if samples is not None:

            def _read_iterator():
                def samples_it():
                    for i, sample in enumerate(samples):
                        assert sample._mixin_prediction_position is None
                        sample._mixin_prediction_position = i
                        if sample.spans is not None and len(sample.spans) > 0:
                            entities = []
                            offset_span = sample.char2token_start[str(sample.offset)]
                            for span_start, span_end in sample.spans:
                                if str(span_start) not in sample.char2token_start:
                                    # span_start is in the middle of a word
                                    # retrieve the first token of the word
                                    while str(span_start) not in sample.char2token_start:
                                        span_start -= 1
                                        # skip
                                        if span_start < 0:
                                            break
                                if str(span_end) not in sample.char2token_end:
                                    # span_end is in the middle of a word
                                    # retrieve the last token of the word
                                    while str(span_end) not in sample.char2token_end:
                                        span_end += 1
                                        # skip
                                        if span_end >= int(list(sample.char2token_end.keys())[-1]):
                                            break

                                if span_start < 0 or span_end > int(list(sample.char2token_end.keys())[-1]):
                                    continue
                                entities.append([sample.char2token_start[str(span_start)]-offset_span, sample.char2token_end[str(span_end)]+1-offset_span, ""])
                            sample.entities = entities
                        yield sample

                next_prediction_position = 0
                position2predicted_sample = {}

                # instantiate dataset
                if self.dataset is None:
                    raise ValueError(
                        "You need to pass a dataset to the model in order to predict"
                    )
                self.dataset.samples = samples_it()
                self.dataset.model_max_length = max_length
                self.dataset.tokens_per_batch = token_batch_size
                self.dataset.max_batch_size = max_batch_size

                # instantiate dataloader
                iterator = DataLoader(
                    self.dataset, batch_size=None, num_workers=0, shuffle=False
                )
                if progress_bar:
                    iterator = tqdm(iterator, desc="Predicting with RelikReader")

                with get_autocast_context(self.device, precision):
                    for batch in iterator:
                        batch = move_data_to_device(batch, self.device)
                        batch.update(kwargs)
                        batch_out = self._batch_predict(**batch)

                        for sample in batch_out:
                            if (
                                sample._mixin_prediction_position
                                >= next_prediction_position
                            ):
                                position2predicted_sample[
                                    sample._mixin_prediction_position
                                ] = sample

                        # yield
                        while next_prediction_position in position2predicted_sample:
                            yield position2predicted_sample[next_prediction_position]
                            del position2predicted_sample[next_prediction_position]
                            next_prediction_position += 1

            outputs = list(_read_iterator())
            for sample in outputs:
                self.dataset.merge_patches_predictions(sample)
                if annotation_type == AnnotationType.CHAR:
                    self.dataset.convert_to_char_annotations(sample)
                elif annotation_type == AnnotationType.WORD:
                    self.dataset.convert_to_word_annotations(sample)
                else:
                    raise ValueError(
                        f"Annotation type {annotation_type} not recognized. "
                        f"Please choose one of {list(AnnotationType)}."
                    )

        else:
            outputs = list(
                self._batch_predict(
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    prediction_mask,
                    special_symbols_mask,
                    *args,
                    **kwargs,
                )
            )
        return outputs

    def _batch_predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        prediction_mask: torch.Tensor | None = None,
        special_symbols_mask: torch.Tensor | None = None,
        special_symbols_mask_entities: torch.Tensor | None = None,
        sample: List[RelikReaderSample] | None = None,
        *args,
        **kwargs,
    ) -> Iterator[RelikReaderSample]:
        """
        A wrapper around the forward method that returns the predicted labels for each sample.
        It also adds the predicted labels to the samples.

        Args:
            input_ids (:obj:`torch.Tensor`):
                The input ids of the text.
            attention_mask (:obj:`torch.Tensor`):
                The attention mask of the text.
            token_type_ids (:obj:`torch.Tensor`, `optional`):
                The token type ids of the text.
            prediction_mask (:obj:`torch.Tensor`, `optional`):
                The prediction mask of the text.
            special_symbols_mask (:obj:`torch.Tensor`, `optional`):
                The special symbols mask of the text.
            sample (:obj:`List[RelikReaderSample]`, `optional`):
                The samples to read. If provided, `text` and `candidates` are ignored.
            top_k (:obj:`int`, `optional`, defaults to 5):
                The amount of top-k most probable entities to predict.
            *args:
                Positional arguments.
            **kwargs:
                Keyword arguments.

        Returns:
            The predicted labels for each sample.
        """
        forward_output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            prediction_mask=prediction_mask,
            special_symbols_mask=special_symbols_mask,
            special_symbols_mask_entities=special_symbols_mask_entities,
            is_prediction=True,
            *args,
            **kwargs,
        )

        ned_start_predictions = forward_output["ned_start_predictions"].cpu().numpy()
        ned_end_predictions = forward_output["ned_end_predictions"]  # .cpu().numpy()
        ed_predictions = forward_output["re_entities_predictions"].cpu().numpy()
        ned_type_predictions = forward_output["ned_type_predictions"].cpu().numpy()
        re_predictions = forward_output["re_predictions"].cpu().numpy()
        re_probabilities = forward_output["re_probabilities"].detach().cpu().numpy()

        for ts, ne_st, ne_end, re_pred, re_prob, edp, ne_et in zip(
            sample,
            ned_start_predictions,
            ned_end_predictions,
            re_predictions,
            re_probabilities,
            ed_predictions,
            ned_type_predictions,
        ):
            ne_end = ne_end.cpu().numpy()
            entities = []
            if self.relik_reader_model.config.entity_type_loss:
                starts = np.argwhere(ne_st)
                i = 0
                for start, end in zip(starts, ne_end):
                    ends = np.argwhere(end)
                    for e in ends:
                        entities.append([start[0], e[0], ne_et[i]])
                        i += 1
                    #     if i == len(ne_et):
                    #         break
                    # if i == len(ne_et):
                    #     break
            else:
                starts = np.argwhere(ne_st)
                for start, end in zip(starts, ne_end):
                    ends = np.argwhere(end)
                    for e in ends:
                        entities.append([start[0], e[0]])

            edp = edp[: len(entities)]
            re_pred = re_pred[: len(entities), : len(entities)]
            re_prob = re_prob[: len(entities), : len(entities)]
            possible_re = np.argwhere(re_pred)
            predicted_triplets = []
            predicted_triplets_prob = []
            for i, j, r in possible_re:
                if self.relik_reader_model.relation_disambiguation_loss:
                    if not (
                        i != j
                        and edp[i, r] == 1
                        and edp[j, r] == 1
                        and edp[i, 0] == 0
                        and edp[j, 0] == 0
                    ):
                        continue
                predicted_triplets.append([i, j, r])
                predicted_triplets_prob.append(re_prob[i, j, r])

            ts._d["predicted_relations"] = predicted_triplets
            ts._d["predicted_entities"] = entities
            ts._d["predicted_relations_probabilities"] = predicted_triplets_prob

            # try-out for a new format
            ts._d["predicted_triples"] = predicted_triplets

            yield ts
