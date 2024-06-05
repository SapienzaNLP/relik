import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import transformers as tr
from torch.utils.data import IterableDataset
from transformers import AutoConfig

from relik.common.log import get_logger

# from relik.common.torch_utils import load_ort_optimized_hf_model
from relik.common.utils import get_callable_from_string
from relik.inference.data.objects import AnnotationType
from relik.reader.pytorch_modules.hf.modeling_relik import (
    RelikReaderConfig,
    RelikReaderSample,
)
from relik.retriever.pytorch_modules import PRECISION_MAP

logger = get_logger(__name__, level=logging.INFO)


class RelikReaderBase(torch.nn.Module):
    default_reader_class: str | None = None
    default_data_class: str | None = None

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
        precision: int = 32,
        tokenizer: str | tr.PreTrainedTokenizer | None = None,
        dataset: IterableDataset | str | None = None,
        default_reader_class: tr.PreTrainedModel | str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.default_reader_class = default_reader_class or self.default_reader_class

        if self.default_reader_class is None:
            raise ValueError("You must specify a default reader class.")

        # get the callable for the default reader class
        self.default_reader_class: tr.PreTrainedModel = get_callable_from_string(
            self.default_reader_class
        )

        if isinstance(transformer_model, str):
            self.name_or_path = transformer_model
            config = AutoConfig.from_pretrained(
                transformer_model, trust_remote_code=True
            )
            if "relik-reader" in config.model_type:
                transformer_model = self.default_reader_class.from_pretrained(
                    transformer_model, config=config, ignore_mismatched_sizes=True, trust_remote_code=True, **kwargs, 
                )
            else:
                reader_config = RelikReaderConfig(
                    transformer_model=transformer_model,
                    additional_special_symbols=additional_special_symbols,
                    num_layers=num_layers,
                    activation=activation,
                    linears_hidden_size=linears_hidden_size,
                    use_last_k_layers=use_last_k_layers,
                    training=training,
                    **kwargs,
                )
                transformer_model = self.default_reader_class(reader_config)
                self.name_or_path = self.relik_reader_model.config.transformer_model

        self.relik_reader_model = transformer_model

        self.relik_reader_model_config = self.relik_reader_model.config
        # self.name_or_path = self.relik_reader_model_config.name_or_path
        # self.name_or_path = self.relik_reader_model.config.transformer_model

        # get the tokenizer
        self._tokenizer = tokenizer

        # and instantiate the dataset class
        self.dataset: IterableDataset | None = dataset

        # move the model to the device
        self.to(device or torch.device("cpu"))

        # set the precision
        self.precision = precision
        self.to(PRECISION_MAP[precision])

    def forward(self, **kwargs) -> Dict[str, Any]:
        return self.relik_reader_model(**kwargs)

    def _read(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @torch.no_grad()
    @torch.inference_mode()
    def read(
        self,
        text: List[str] | List[List[str]] | None = None,
        samples: List[RelikReaderSample] | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        prediction_mask: torch.Tensor | None = None,
        special_symbols_mask: torch.Tensor | None = None,
        candidates: List[List[str]] | None = None,
        max_length: int = 1000,
        max_batch_size: int = 128,
        token_batch_size: int = 2048,
        precision: int | str | None = None,
        annotation_type: str | AnnotationType = AnnotationType.CHAR,
        progress_bar: bool = False,
        *args,
        **kwargs,
    ) -> List[RelikReaderSample] | List[List[RelikReaderSample]]:
        """
        Reads the given text.

        Args:
            text (:obj:`List[str]` or :obj:`List[List[str]]`, `optional`):
                The text to read in tokens. If a list of list of tokens is provided, each
                inner list is considered a sentence.
            samples (:obj:`List[RelikReaderSample]`, `optional`):
                The samples to read. If provided, `text` and `candidates` are ignored.
            input_ids (:obj:`torch.Tensor`, `optional`):
                The input ids of the text.
            attention_mask (:obj:`torch.Tensor`, `optional`):
                The attention mask of the text.
            token_type_ids (:obj:`torch.Tensor`, `optional`):
                The token type ids of the text.
            prediction_mask (:obj:`torch.Tensor`, `optional`):
                The prediction mask of the text.
            special_symbols_mask (:obj:`torch.Tensor`, `optional`):
                The special symbols mask of the text.
            candidates (:obj:`List[List[str]]`, `optional`):
                The candidates of the text.
            max_length (:obj:`int`, `optional`, defaults to 1024):
                The maximum length of the text.
            max_batch_size (:obj:`int`, `optional`, defaults to 128):
                The maximum batch size.
            token_batch_size (:obj:`int`, `optional`):
                The maximum number of tokens per batch.
            precision (:obj:`int` or :obj:`str`, `optional`):
                The precision to use. If not provided, the default is 32 bit.
            annotation_type (`str` or `AnnotationType`, `optional`, defaults to `char`):
                The type of annotation to return. If `char`, the spans will be in terms of
                character offsets. If `word`, the spans will be in terms of word offsets.
            progress_bar (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to show a progress bar.

        Returns:
            The predicted labels for each sample.
        """
        if isinstance(annotation_type, str):
            try:
                annotation_type = AnnotationType(annotation_type)
            except ValueError:
                raise ValueError(
                    f"Annotation type `{annotation_type}` not recognized. "
                    f"Please choose one of {list(AnnotationType)}."
                )

        if text is None and input_ids is None and samples is None:
            raise ValueError(
                "Either `text` or `input_ids` or `samples` must be provided."
            )
        if (input_ids is None and samples is None) and (
            text is None or candidates is None
        ):
            raise ValueError(
                "`text` and `candidates` must be provided to return the predictions when "
                "`input_ids` and `samples` is not provided."
            )
        if text is not None and samples is None:
            if len(text) != len(candidates):
                raise ValueError("`text` and `candidates` must have the same length.")
            if isinstance(text[0], str):  # change to list of text
                text = [text]
                candidates = [candidates]

            samples = [
                RelikReaderSample(tokens=t, candidates=c)
                for t, c in zip(text, candidates)
            ]

        return self._read(
            samples,
            input_ids,
            attention_mask,
            token_type_ids,
            prediction_mask,
            special_symbols_mask,
            max_length,
            max_batch_size,
            token_batch_size,
            precision or self.precision,
            annotation_type,
            progress_bar,
            *args,
            **kwargs,
        )

    @property
    def device(self) -> torch.device:
        """
        The device of the model.
        """
        return next(self.parameters()).device

    @property
    def tokenizer(self) -> tr.PreTrainedTokenizer:
        """
        The tokenizer.
        """
        if self._tokenizer:
            return self._tokenizer

        self._tokenizer = tr.AutoTokenizer.from_pretrained(
            self.relik_reader_model.config.name_or_path if self.relik_reader_model.config.name_or_path else self.relik_reader_model.config.transformer_model
        )
        return self._tokenizer

    def save_pretrained(
        self,
        output_dir: str | os.PathLike,
        model_name: str | None = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> None:
        """
        Saves the model to the given path.

        Args:
            output_dir (`str` or :obj:`os.PathLike`):
                The path to save the model to.
            model_name (`str`, `optional`):
                The name of the model. If not provided, the model will be saved as
                `default_reader_class.__name__`.
            push_to_hub (`bool`, `optional`, defaults to `False`):
                Whether to push the model to the HuggingFace Hub.
            **kwargs:
                Additional keyword arguments to pass to the `save_pretrained` method
        """
        # create the output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = model_name or output_dir.name

        logger.info(f"Saving reader to {output_dir / model_name}")

        # save the model
        self.relik_reader_model.register_for_auto_class()
        self.relik_reader_model.save_pretrained(
            str(output_dir / model_name), push_to_hub=push_to_hub, **kwargs
        )

        if self.tokenizer:
            logger.info("Saving also the tokenizer")
            self.tokenizer.save_pretrained(
                str(output_dir / model_name), push_to_hub=push_to_hub, **kwargs
            )
