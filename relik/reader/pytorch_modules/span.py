import collections
import logging
from typing import Any, Dict, Iterator, List

import torch
import transformers as tr
from lightning_fabric.utilities import move_data_to_device
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from relik.common.torch_utils import get_autocast_context
from relik.common.log import get_logger
from relik.common.utils import get_callable_from_string
from relik.inference.data.objects import AnnotationType
from relik.reader.data.relik_reader_sample import RelikReaderSample
from relik.reader.pytorch_modules.base import RelikReaderBase

logger = get_logger(__name__, level=logging.INFO)


class RelikReaderForSpanExtraction(RelikReaderBase):
    """
    A class for the RelikReader model for span extraction.

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
        "relik.reader.pytorch_modules.hf.modeling_relik.RelikReaderSpanModel"
    )
    default_data_class: str = "relik.reader.data.relik_reader_data.RelikDataset"

    def __init__(
        self,
        transformer_model: str | tr.PreTrainedModel | None = None,
        additional_special_symbols: int = 0,
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
        if self.dataset is None:
            self.default_data_class = get_callable_from_string(self.default_data_class)
            default_data_kwargs = dict(
                dataset_path=None,
                materialize_samples=False,
                transformer_model=self.tokenizer,
                special_symbols=self.default_data_class.get_special_symbols(
                    self.relik_reader_model.config.additional_special_symbols
                ),
                for_inference=True,
                use_nme=kwargs.get("use_nme", True),
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
        max_length: int = 1000,
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
                            sample.window_labels = [[s[0], s[1], ""] for s in sample.spans]
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
        sample: List[RelikReaderSample] | None = None,
        top_k: int = 5,  # the amount of top-k most probable entities to predict
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
            *args,
            **kwargs,
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

            # try-out for a new format
            sample_patch["predicted_spans"] = final_class2predicted_spans
            sample_patch[
                "predicted_spans_probabilities"
            ] = spans2predicted_probabilities

            yield ts
