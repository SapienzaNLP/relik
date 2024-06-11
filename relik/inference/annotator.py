from collections import defaultdict
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pprintpp import pformat

from relik.common.log import get_logger, print_relik_text_art
from relik.common.upload import get_logged_in_username, upload
from relik.common.utils import CONFIG_NAME, from_cache
from relik.inference.data.objects import (
    AnnotationType,
    Candidates,
    RelikOutput,
    Span,
    TaskType,
    Triples,
)
from relik.inference.data.splitters.base_sentence_splitter import BaseSentenceSplitter
from relik.inference.data.splitters.blank_sentence_splitter import BlankSentenceSplitter
from relik.inference.data.splitters.spacy_sentence_splitter import SpacySentenceSplitter
from relik.inference.data.splitters.window_based_splitter import WindowSentenceSplitter
from relik.inference.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from relik.inference.data.window.manager import WindowManager
from relik.reader.data.relik_reader_sample import RelikReaderSample
from relik.reader.pytorch_modules.base import RelikReaderBase
from relik.reader.pytorch_modules.span import RelikReaderForSpanExtraction
from relik.reader.pytorch_modules.triplet import RelikReaderForTripletExtraction
from relik.retriever.indexers.base import BaseDocumentIndex
from relik.retriever.indexers.document import Document
from relik.retriever.pytorch_modules import PRECISION_MAP
from relik.retriever.pytorch_modules.model import GoldenRetriever

# set tokenizers parallelism to False

os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")

LOG_QUERY = os.getenv("RELIK_LOG_QUERY_ON_FILE", "false").lower() == "true"

logger = get_logger(__name__)

file_logger = None
if LOG_QUERY:
    RELIK_LOG_PATH = Path(__file__).parent.parent.parent / "relik.log"
    # create file handler which logs even debug messages
    fh = logging.FileHandler(RELIK_LOG_PATH)
    fh.setLevel(logging.INFO)
    file_logger = get_logger("relik", level=logging.INFO)
    file_logger.addHandler(fh)


class Relik:
    """
    Relik main class. It is a wrapper around a retriever and a reader.

    Args:
        retriever (:obj:`GoldenRetriever`):
            The retriever to use.
        index (:obj:`BaseDocumentIndex`, `optional`):
            The document index to use. If `None`, the retriever's document index will be used.
        reader (:obj:`RelikReaderBase`):
            The reader to use.
        document_index (:obj:`BaseDocumentIndex`, `optional`):
            The document index to use. If `None`, the retriever's document index will be used.
        device (`str`, `optional`, defaults to `cpu`):
            The device to use for both the retriever and the reader.
        retriever_device (`str`, `optional`, defaults to `None`):
            The device to use for the retriever. If `None`, the `device` argument will be used.
        document_index_device (`str`, `optional`, defaults to `None`):
            The device to use for the document index. If `None`, the `device` argument will be used.
        reader_device (`str`, `optional`, defaults to `None`):
            The device to use for the reader. If `None`, the `device` argument will be used.
        precision (`int`, `str` or `torch.dtype`, `optional`, defaults to `32`):
            The precision to use for both the retriever and the reader.
        retriever_precision (`int`, `str` or `torch.dtype`, `optional`, defaults to `None`):
            The precision to use for the retriever. If `None`, the `precision` argument will be used.
        document_index_precision (`int`, `str` or `torch.dtype`, `optional`, defaults to `None`):
            The precision to use for the document index. If `None`, the `precision` argument will be used.
        reader_precision (`int`, `str` or `torch.dtype`, `optional`, defaults to `None`):
            The precision to use for the reader. If `None`, the `precision` argument will be used.
        metadata_fields (`list[str]`, `optional`, defaults to `None`):
            The fields to add to the candidates for the reader.
        top_k (`int`, `optional`, defaults to `None`):
            The number of candidates to retrieve for each window.
        window_size (`int`, `optional`, defaults to `None`):
            The size of the window. If `None`, the whole text will be annotated.
        window_stride (`int`, `optional`, defaults to `None`):
            The stride of the window. If `None`, there will be no overlap between windows.
        **kwargs:
            Additional keyword arguments to pass to the retriever and the reader.
    """

    def __init__(
        self,
        retriever: GoldenRetriever | DictConfig | Dict | None = None,
        index: BaseDocumentIndex | DictConfig | Dict | None = None,
        reader: RelikReaderBase | DictConfig | None = None,
        device: str | None = None,
        retriever_device: str | None = None,
        index_device: str | None = None,
        reader_device: str | None = None,
        precision: int | str | torch.dtype | None = None,
        retriever_precision: int | str | torch.dtype | None = None,
        index_precision: int | str | torch.dtype | None = None,
        reader_precision: int | str | torch.dtype | None = None,
        task: TaskType | str = TaskType.SPAN,
        metadata_fields: list[str] | None = None,
        top_k: int | None = None,
        window_size: int | str | None = None,
        window_stride: int | None = None,
        retriever_kwargs: Dict[str, Any] | None = None,
        index_kwargs: Dict[str, Any] | None = None,
        reader_kwargs: Dict[str, Any] | None = None,
        **kwargs,
    ) -> None:

        if retriever_kwargs is None:
            retriever_kwargs = {}
        if index_kwargs is None:
            index_kwargs = {}
        if reader_kwargs is None:
            reader_kwargs = {}

        # selected parameters that are popped from kwargs
        index_kwargs["use_faiss"] = kwargs.pop("use_faiss", False)
        reader_kwargs["use_nme"] = kwargs.pop("use_nme", False)

        # parse task into a TaskType
        if isinstance(task, str):
            try:
                task = TaskType(task.lower())
            except ValueError:
                raise ValueError(
                    f"Task `{task}` not recognized. "
                    f"Please choose one of {list(TaskType)}."
                )
        self.task = task

        # organize devices
        if device is not None:
            if retriever_device is None:
                retriever_device = device
            if index_device is None:
                index_device = device
            if reader_device is None:
                reader_device = device

        # organize precision
        if precision is not None:
            if retriever_precision is None:
                retriever_precision = precision
            if index_precision is None:
                index_precision = precision
            if reader_precision is None:
                reader_precision = precision

        # retriever
        self._retriever: Dict[TaskType, GoldenRetriever] = {
            TaskType.SPAN: None,
            TaskType.TRIPLET: None,
        }

        if retriever:
            # check retriever type, it can be a GoldenRetriever, a DictConfig or a Dict
            if not isinstance(retriever, (GoldenRetriever, DictConfig, Dict)):
                raise ValueError(
                    f"`retriever` must be a `GoldenRetriever`, a `DictConfig` or "
                    f"a `Dict`, got `{type(retriever)}`."
                )

            # we need to check weather the DictConfig is a DictConfig for an instance of GoldenRetriever
            # or a primitive Dict
            if isinstance(retriever, DictConfig):
                # then it is probably a primitive Dict
                if "_target_" not in retriever:
                    retriever = OmegaConf.to_container(retriever, resolve=True)
                    # convert the key to TaskType
                    try:
                        retriever = {
                            TaskType(k.lower()): v for k, v in retriever.items()
                        }
                    except ValueError as e:
                        raise ValueError(
                            f"Please choose a valid task type (one of {list(TaskType)}) for each retriever."
                        ) from e

            if isinstance(retriever, Dict):
                # convert the key to TaskType
                retriever = {TaskType(k): v for k, v in retriever.items()}
            else:
                retriever = {task: retriever}

            # instantiate each retriever
            if self.task in [TaskType.SPAN, TaskType.BOTH]:
                self._retriever[TaskType.SPAN] = self._instantiate_retriever(
                    retriever[TaskType.SPAN],
                    retriever_device,
                    retriever_precision,
                    retriever_kwargs,
                )
            if self.task in [TaskType.TRIPLET, TaskType.BOTH]:
                self._retriever[TaskType.TRIPLET] = self._instantiate_retriever(
                    retriever[TaskType.TRIPLET],
                    retriever_device,
                    retriever_precision,
                    retriever_kwargs,
                )

            # clean up None retrievers from the dictionary
            self._retriever = {
                task_type: r
                for task_type, r in self._retriever.items()
                if r is not None
            }
            # torch compile
            # self._retriever = {task_type: torch.compile(r, backend="onnxrt") for task_type, r in self._retriever.items()}

        # index
        self._index: Dict[TaskType, BaseDocumentIndex] = {
            TaskType.SPAN: None,
            TaskType.TRIPLET: None,
        }

        if index:
            # check retriever type, it can be a BaseDocumentIndex, a DictConfig or a Dict
            if not isinstance(index, (BaseDocumentIndex, DictConfig, Dict)):
                raise ValueError(
                    f"`index` must be a `BaseDocumentIndex`, a `DictConfig` or "
                    f"a `Dict`, got `{type(index)}`."
                )
            # we need to check weather the DictConfig is a DictConfig for an instance of BaseDocumentIndex
            # or a primitive Dict
            if isinstance(index, DictConfig):
                # then it is probably a primitive Dict
                if "_target_" not in index:
                    index = OmegaConf.to_container(index, resolve=True)
                    # convert the key to TaskType
                    try:
                        index = {TaskType(k.lower()): v for k, v in index.items()}
                    except ValueError as e:
                        raise ValueError(
                            f"Please choose a valid task type (one of {list(TaskType)}) for each index."
                        ) from e

            if isinstance(index, Dict):
                # convert the key to TaskType
                index = {TaskType(k): v for k, v in index.items()}
            else:
                index = {task: index}

            # instantiate each retriever
            if self.task in [TaskType.SPAN, TaskType.BOTH]:
                self._index[TaskType.SPAN] = self._instantiate_index(
                    index[TaskType.SPAN],
                    index_device,
                    index_precision,
                    index_kwargs,
                )
            if self.task in [TaskType.TRIPLET, TaskType.BOTH]:
                self._index[TaskType.TRIPLET] = self._instantiate_index(
                    index[TaskType.TRIPLET],
                    index_device,
                    index_precision,
                    index_kwargs,
                )

            # clean up None retrievers from the dictionary
            self._index = {
                task_type: i for task_type, i in self._index.items() if i is not None
            }

            # link each index to the retriever
            for task_type, retriever in self._retriever.items():
                if task_type in self._index:
                    retriever.document_index = self._index[task_type]

        # reader
        self.reader: RelikReaderBase | None = None
        if reader:
            reader = (
                hydra.utils.instantiate(
                    reader,
                    device=reader_device,
                    precision=reader_precision,
                    **reader_kwargs,
                )
                if isinstance(reader, DictConfig)
                else reader
            )
            reader.training = False
            reader.eval()
            if reader_device is not None:
                logger.info(f"Moving reader to `{reader_device}`.")
                reader.to(reader_device)
            if (
                reader_precision is not None
                and reader.precision != PRECISION_MAP[reader_precision]
            ):
                logger.info(
                    f"Setting precision of reader to `{PRECISION_MAP[reader_precision]}`."
                )
                reader.to(PRECISION_MAP[reader_precision])
            self.reader = reader
            # self.reader = torch.compile(self.reader, backend="tvm")

        # windowization stuff
        self.tokenizer = SpacyTokenizer(language="en")  # TODO: parametrize?
        self.sentence_splitter: BaseSentenceSplitter | None = None
        self.window_manager: WindowManager | None = None

        if metadata_fields is None:
            metadata_fields = []
        self.metadata_fields = metadata_fields

        # inference params
        self.top_k = top_k
        self.window_size = window_size
        self.window_stride = window_stride

    @staticmethod
    def _instantiate_index(
        index: BaseDocumentIndex | DictConfig | Dict,
        index_device: str | None | torch.device | int = None,
        index_precision: int | str | torch.dtype | None = None,
        index_kwargs: Dict[str, Any] | None = None,
    ) -> BaseDocumentIndex:
        """
        Instantiate a document index.

        Args:
            index (`BaseDocumentIndex`, `DictConfig` or `Dict`):
                The document index to instantiate.
            index_device (`str`, `optional`):
                The device to use for the document index.
            index_precision (`int`, `str` or `torch.dtype`, `optional`):
                The precision to use for the document index.
            index_kwargs (`Dict[str, Any]`, `optional`):
                Additional keyword arguments to pass to the document index.

        Returns:
            `BaseDocumentIndex`:
                The instantiated document index.
        """
        if not isinstance(index, BaseDocumentIndex):
            index = OmegaConf.create(index)
            use_faiss = index_kwargs.get("use_faiss", False)
            if use_faiss:
                index = OmegaConf.merge(
                    index,
                    {
                        "_target_": "relik.retriever.indexers.faissindex.FaissDocumentIndex.from_pretrained",
                    },
                )
            if index_device is not None:
                index_kwargs["device"] = index_device
            if index_precision is not None:
                index_kwargs["precision"] = index_precision

            # merge the kwargs
            index = OmegaConf.merge(index, OmegaConf.create(index_kwargs))
            index: BaseDocumentIndex = hydra.utils.instantiate(index)
        else:
            index = index
            if index_device is not None:
                logger.info(f"Moving index to `{index_device}`.")
                index.to(index_device)
            if index_device is not None:
                logger.info(
                    f"Setting precision of index to `{PRECISION_MAP[index_device]}`."
                )
                index.to(PRECISION_MAP[index_device])
        return index

    @staticmethod
    def _instantiate_retriever(
        retriever: GoldenRetriever | DictConfig | Dict,
        retriever_device: str | None,
        retriever_precision: int | str | torch.dtype | None,
        retriever_kwargs: Dict[str, Any] | None,
    ) -> GoldenRetriever:
        """
        Instantiate a retriever.

        Args:
            retriever (`GoldenRetriever`, `DictConfig` or `Dict`):
                The retriever to instantiate.
            retriever_device (`str`, `optional`):
                The device to use for the retriever.
            retriever_precision (`int`, `str` or `torch.dtype`, `optional`):
                The precision to use for the retriever.
            retriever_kwargs (`Dict[str, Any]`, `optional`):
                Additional keyword arguments to pass to the retriever.

        Returns:
            `GoldenRetriever`:
                The instantiated retriever.
        """
        if not isinstance(retriever, GoldenRetriever):
            # convert to DictConfig
            retriever = hydra.utils.instantiate(
                OmegaConf.create(retriever),
                device=retriever_device,
                precision=retriever_precision,
                # index_device=document_index_device,
                # index_precision=document_index_precision,
                **retriever_kwargs,
            )
        else:
            if retriever_device is not None:
                logger.info(f"Moving retriever to `{retriever_device}`.")
                retriever.to(retriever_device)
            if retriever_precision is not None:
                logger.info(
                    f"Setting precision of retriever to `{PRECISION_MAP[retriever_precision]}`."
                )
                retriever.to(PRECISION_MAP[retriever_precision])
        retriever.training = False
        retriever.eval()
        # if document_index is not None:
        #     if retriever.document_index is not None:
        #         logger.info(
        #             "The Retriever already has a document index, replacing it with the provided one."
        #             "If you want to keep using the old one, please do not provide a document index."
        #         )
        #         retriever.document_index = document_index
        # # we override the device and the precision of the document index if provided
        # if document_index_device is not None:
        #     logger.info(f"Moving document index to `{document_index_device}`.")
        #     retriever.document_index.to(document_index_device)
        # if document_index_precision is not None:
        #     logger.info(
        #         f"Setting precision of document index to `{PRECISION_MAP[document_index_precision]}`."
        #     )
        #     retriever.document_index.to(PRECISION_MAP[document_index_precision])
        # retriever.document_index = document_index
        # now we can move the retriever to the right device and set the precision
        return retriever

    @property
    def retriever(self) -> GoldenRetriever | Dict[TaskType, GoldenRetriever]:
        """
        Get the retriever.

        Returns:
            `GoldenRetriever` or `Dict[TaskType, GoldenRetriever]`:
                The retriever or a dictionary of retrievers.
        """
        if len(self._retriever) == 1:
            return list(self._retriever.values())[0]
        return self._retriever

    @property
    def index(self) -> BaseDocumentIndex | Dict[TaskType, BaseDocumentIndex]:
        """
        Get the document index.

        Returns:
            `BaseDocumentIndex` or `Dict[TaskType, BaseDocumentIndex]`:
                The document index or a dictionary of document indexes.
        """
        if len(self._index) == 1:
            return list(self._index.values())[0]
        return self._index

    def __call__(
        self,
        text: str | List[str] | None = None,
        windows: List[RelikReaderSample] | None = None,
        candidates: (
            List[str] | List[Document] | Dict[TaskType, List[Document]] | None
        ) = None,
        mentions: List[List[int]] | List[List[List[int]]] | None = None,
        top_k: int | None = None,
        window_size: int | str | None = None,
        window_stride: int | str | None = None,
        is_split_into_words: bool = False,
        retriever_batch_size: int | None = 32,
        reader_batch_size: int | None = 32,
        return_windows: bool = False,
        use_doc_topic: bool = False,
        annotation_type: str | AnnotationType = AnnotationType.CHAR,
        progress_bar: bool = False,
        **kwargs,
    ) -> Union[RelikOutput, list[RelikOutput]]:
        """
        Annotate a text with entities.

        Args:
            text (`str` or `list`):
                The text to annotate. If a list is provided, each element of the list
                 will be annotated separately.
            candidates (`list[str]`, `list[Document]`, `optional`, defaults to `None`):
                The candidates to use for the reader. If `None`, the candidates will be
                retrieved from the retriever.
            mentions (`list[list[int]]` or `list[list[list[int]]]`, `optional`, defaults to `None`):
                The mentions to use for the reader. If `None`, the mentions will be
                predicted by the reader.
            top_k (`int`, `optional`, defaults to `None`):
                The number of candidates to retrieve for each window.
            window_size (`int`, `optional`, defaults to `None`):
                The size of the window. If `None`, the whole text will be annotated.
            window_stride (`int`, `optional`, defaults to `None`):
                The stride of the window. If `None`, there will be no overlap between windows.
            retriever_batch_size (`int`, `optional`, defaults to `None`):
                The batch size to use for the retriever. The whole input is the batch for the retriever.
            reader_batch_size (`int`, `optional`, defaults to `None`):
                The batch size to use for the reader. The whole input is the batch for the reader.
            return_windows (`bool`, `optional`, defaults to `False`):
                Whether to return the windows in the output.
            annotation_type (`str` or `AnnotationType`, `optional`, defaults to `char`):
                The type of annotation to return. If `char`, the spans will be in terms of
                character offsets. If `word`, the spans will be in terms of word offsets.
            **kwargs:
                Additional keyword arguments to pass to the retriever and the reader.

        Returns:
            `RelikOutput` or `list[RelikOutput]`:
                The annotated text. If a list was provided as input, a list of
                `RelikOutput` objects will be returned.
        """

        if text is None and windows is None:
            raise ValueError(
                "Either `text` or `windows` must be provided. Both are `None`."
            )

        if isinstance(annotation_type, str):
            try:
                annotation_type = AnnotationType(annotation_type)
            except ValueError:
                raise ValueError(
                    f"Annotation type {annotation_type} not recognized. "
                    f"Please choose one of {list(AnnotationType)}."
                )

        if top_k is None:
            top_k = self.top_k or 100
        if window_size is None:
            window_size = self.window_size
        if window_stride is None:
            window_stride = self.window_stride

        if text:
            # normalize text to a list
            if isinstance(text, str):
                text = [text]
                # normalize mentions to a list
                if mentions is not None:
                    mentions = [mentions]

            if self.window_manager is None:
                # no actual windowization, use the input as is
                if window_size == "none":
                    self.sentence_splitter = BlankSentenceSplitter()
                # sentence-based windowization, uses a sentence splitter to create windows
                elif window_size == "sentence":
                    self.sentence_splitter = SpacySentenceSplitter()
                # word-based windowization, uses a window size and stride to create windows
                else:
                    self.sentence_splitter = WindowSentenceSplitter(
                        window_size=window_size, window_stride=window_stride
                    )
                self.window_manager = WindowManager(
                    self.tokenizer, self.sentence_splitter
                )

            # sanity check for window size and stride
            if (
                window_size not in ["sentence", "none"]
                and window_stride is not None
                and window_size < window_stride
            ):
                raise ValueError(
                    f"Window size ({window_size}) must be greater than window stride ({window_stride})"
                )

        # if there are no windows, create them
        if windows is None:
            # TODO: make it more consistent (no tuples or single elements in output)
            # windows were provided, use them
            if mentions is not None:
                windows, blank_windows = self.window_manager.create_windows(
                    text,
                    window_size,
                    window_stride,
                    is_split_into_words=is_split_into_words,
                    mentions=mentions,
                )
            else:
                blank_windows = []
                windows = self.window_manager.create_windows(
                    text,
                    window_size,
                    window_stride,
                    is_split_into_words=is_split_into_words,
                )
        else:
            # otherwise, use the provided windows, `text` is ignored
            blank_windows = []
            text = {w.doc_id: w.text for w in windows}

        if candidates is not None and any(
            r is not None for r in self._retriever.values()
        ):
            logger.info(
                "Both candidates and a retriever were provided. "
                "Retriever will be ignored."
            )

        windows_candidates = {TaskType.SPAN: None, TaskType.TRIPLET: None}
        # candidates are provided, use them and skip retrieval
        if candidates is not None:
            # again, check if candidates is a dict
            if isinstance(candidates, Dict):
                if self.task not in candidates:
                    raise ValueError(
                        f"Task `{self.task}` not found in `candidates`."
                        f"Please choose one of {list(TaskType)}."
                    )
            else:
                candidates = {self.task: candidates}

            for task_type, _candidates in candidates.items():
                if isinstance(_candidates, list):
                    _candidates = [
                        [
                            c if isinstance(c, Document) else Document(c)
                            for c in _candidates[w.doc_id]
                        ]
                        for w in windows
                    ]
                windows_candidates[task_type] = _candidates

        else:
            # retrieve candidates first
            if self._retriever is None:
                raise ValueError(
                    "No retriever was provided, please provide a retriever or candidates."
                )
            start_retr = time.time()
            # retrieve for each task type
            for task_type, retriever in self._retriever.items():
                retriever_out = retriever.retrieve(
                    [w.text for w in windows],
                    text_pair=[
                        (
                            w.doc_topic
                            if (w.doc_topic is not None and use_doc_topic)
                            else None
                        )
                        for w in windows
                    ],
                    k=top_k,
                    batch_size=retriever_batch_size,
                    progress_bar=progress_bar,
                    **kwargs,
                )
                windows_candidates[task_type] = [
                    [p.document for p in predictions] for predictions in retriever_out
                ]
            end_retr = time.time()
            # logger.debug(f"Retrieval took {end_retr - start_retr} seconds.")

        # clean up None's
        windows_candidates = {
            t: c for t, c in windows_candidates.items() if c is not None
        }

        # add passage to the windows
        for task_type, task_candidates in windows_candidates.items():
            for window, candidates in zip(windows, task_candidates):
                # construct the candidates for the reader
                formatted_candidates = []
                for candidate in candidates:
                    window_candidate_text = candidate.text
                    # the metadata fields are concatenated to the text to be used by the reader
                    # by default, the reader uses just the text as the candidate
                    # but this behavior can be changed by the user by providing a list of metadata fields
                    for field in self.metadata_fields:
                        window_candidate_text += f"{candidate.metadata.get(field, '')}"
                    formatted_candidates.append(window_candidate_text)
                # create a member for the windows that is named like the task
                window._d[f"{task_type.value}_candidates"] = formatted_candidates

        for task_type, task_candidates in windows_candidates.items():
            for window in blank_windows:
                window._d[f"{task_type.value}_candidates"] = []
                window._d["predicted_spans"] = []
                window._d["predicted_triples"] = []

        if self.reader is not None:
            start_read = time.time()
            windows = self.reader.read(
                samples=windows,
                max_batch_size=reader_batch_size,
                annotation_type=annotation_type,
                progress_bar=progress_bar,
                **kwargs,
            )
            end_read = time.time()
            # logger.debug(f"Reading took {end_read - start_read} seconds.")

            # replace the reader "text" candidates with the full Document ones
            for task_type, task_candidates in windows_candidates.items():
                for i, task_candidate in enumerate(task_candidates):
                    # if f"{task_type.value}_candidates" in windows[i]._d:
                    windows[i]._d[f"{task_type.value}_candidates"] = task_candidate

            # TODO: check merging behavior without a reader
            # do we want to merge windows if there is no reader? I don't think so :)
            if self.window_size is not None and self.window_size not in [
                "sentence",
                "none",
            ]:
                start_w = time.time()
                windows = windows + blank_windows
                windows.sort(key=lambda x: (x.doc_id, x.offset))
                merged_windows = self.window_manager.merge_windows(windows)
                end_w = time.time()
                # logger.debug(f"Merging took {end_w - start_w} seconds.")
            else:
                merged_windows = windows
        else:
            # if there is no reader, just return the windows
            windows = windows + blank_windows
            windows.sort(key=lambda x: (x.doc_id, x.offset))
            merged_windows = windows

        # transform predictions into RelikOutput objects
        output = []
        for w in merged_windows:
            span_labels = []
            triples_labels = []
            # span extraction should always be present
            if getattr(w, "predicted_spans", None) is not None:
                span_labels = sorted(
                    [
                        (
                            Span(start=ss, end=se, label=sl, text=text[w.doc_id][ss:se])
                            if annotation_type == AnnotationType.CHAR
                            else Span(start=ss, end=se, label=sl, text=w.words[ss:se])
                        )
                        for ss, se, sl in w.predicted_spans
                    ],
                    key=lambda x: x.start,
                )
                # triple extraction is optional, if here add it
                if getattr(w, "predicted_triples", None) is not None:
                    triples_labels = [
                        Triples(
                            subject=span_labels[subj],
                            label=label,
                            object=span_labels[obj],
                            confidence=conf,
                        )
                        for subj, label, obj, conf in w.predicted_triples
                    ]
            # we also want to add the candidates to the output
            candidates_labels = defaultdict(list)
            for task_type, _ in windows_candidates.items():
                if f"{task_type.value}_candidates" in w._d:
                    candidates_labels[task_type].append(
                        w._d[f"{task_type.value}_candidates"]
                    )

            sample_output = RelikOutput(
                text=w.text,
                tokens=w.words,
                spans=span_labels,
                triples=triples_labels,
                candidates=Candidates(
                    span=candidates_labels.get(TaskType.SPAN, []),
                    triplet=candidates_labels.get(TaskType.TRIPLET, []),
                ),
            )
            output.append(sample_output)

        # add windows to the output if requested
        # do we want to force windows to be returned if there is no reader?
        if return_windows:
            for i, sample_output in enumerate(output):
                sample_output.windows = [w.to_dict() for w in windows if w.doc_id == i]

        # if only one text was provided, return a single RelikOutput object
        if len(output) == 1:
            return output[0]

        return output

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_dir: Union[str, os.PathLike],
        config_file_name: str = CONFIG_NAME,
        *args,
        **kwargs,
    ) -> "Relik":
        """
        Instantiate a `Relik` from a pretrained model.

        Args:
            model_name_or_dir (`str` or `os.PathLike`):
                The name or path of the model to load.
            config_file_name (`str`, `optional`, defaults to `config.yaml`):
                The name of the configuration file to load.
            *args:
                Additional positional arguments to pass to `OmegaConf.merge`.
            **kwargs:
                Additional keyword arguments to pass to `OmegaConf.merge`.

        Returns:
            `Relik`:
                The instantiated `Relik`.

        """

        print_relik_text_art()

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)

        if "retriever" in kwargs and kwargs["retriever"] is None:
            # if retriever is None, we don't want to load the index
            kwargs["index"] = None

        model_dir = from_cache(
            model_name_or_dir,
            filenames=[config_file_name],
            cache_dir=cache_dir,
            force_download=force_download,
        )

        config_path = model_dir / config_file_name
        if not config_path.exists():
            raise FileNotFoundError(
                f"Model configuration file not found at {config_path}."
            )

        # overwrite config with config_kwargs
        config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, OmegaConf.create(kwargs))
        # do we want to print the config? I like it
        logger.info(f"Loading Relik from {model_name_or_dir}")
        logger.info(pformat(OmegaConf.to_container(config)))

        # load relik from config
        relik = hydra.utils.instantiate(config, _recursive_=False, *args)

        return relik

    def save_pretrained(
        self,
        output_dir: Union[str, os.PathLike],
        config: Optional[Dict[str, Any]] = None,
        config_file_name: Optional[str] = None,
        save_weights: bool = False,
        push_to_hub: bool = False,
        model_id: Optional[str] = None,
        organization: Optional[str] = None,
        repo_name: Optional[str] = None,
        retriever_model_id: Optional[str] = None,
        reader_model_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Save the configuration of Relik to the specified directory as a YAML file.

        Args:
            output_dir (`str`):
                The directory to save the configuration file to.
            config (`Optional[Dict[str, Any]]`, `optional`):
                The configuration to save. If `None`, the current configuration will be
                saved. Defaults to `None`.
            config_file_name (`Optional[str]`, `optional`):
                The name of the configuration file. Defaults to `config.yaml`.
            save_weights (`bool`, `optional`):
                Whether to save the weights of the model. Defaults to `False`.
            push_to_hub (`bool`, `optional`):
                Whether to push the saved model to the hub. Defaults to `False`.
            model_id (`Optional[str]`, `optional`):
                The id of the model to push to the hub. If `None`, the name of the
                directory will be used. Defaults to `None`.
            organization (`Optional[str]`, `optional`):
                The organization to push the model to. Defaults to `None`.
            repo_name (`Optional[str]`, `optional`):
                The name of the repository to push the model to. Defaults to `None`.
            retriever_model_id (`Optional[str]`, `optional`):
                The id of the retriever model to push to the hub. If `None`, the name of the
                directory will be used. Defaults to `None`.
            reader_model_id (`Optional[str]`, `optional`):
                The id of the reader model to push to the hub. If `None`, the name of the
                directory will be used. Defaults to `None`.
            **kwargs:
                Additional keyword arguments to pass to `OmegaConf.save`.
        """
        # create the output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        retrievers_names: Dict[TaskType, Dict | None] = {
            TaskType.SPAN: {
                "question_encoder_name": None,
                "passage_encoder_name": None,
            },
            TaskType.TRIPLET: {
                "question_encoder_name": None,
                "passage_encoder_name": None,
            },
        }
        index_names = {
            TaskType.SPAN: None,
            TaskType.TRIPLET: None,
        }

        if save_weights:
            # save weights
            # retriever
            model_id = model_id or output_dir.name
            retriever_model_id = retriever_model_id or f"retriever-{model_id}"
            for task_type, retriever in self._retriever.items():
                if retriever is None:
                    continue
                task_retriever_model_id = f"{retriever_model_id}-{task_type.value}"
                question_encoder_name = f"{task_retriever_model_id}-question-encoder"
                passage_encoder_name = f"{task_retriever_model_id}-passage-encoder"
                logger.info(
                    f"Saving retriever to {output_dir / task_retriever_model_id}"
                )
                retriever.save_pretrained(
                    output_dir / task_retriever_model_id,
                    question_encoder_name=question_encoder_name,
                    passage_encoder_name=passage_encoder_name,
                    push_to_hub=push_to_hub,
                    organization=organization,
                    **kwargs,
                )
                retrievers_names[task_type] = {
                    # "reader_model_id": task_retriever_model_id,
                    "question_encoder_name": question_encoder_name,
                    "passage_encoder_name": passage_encoder_name,
                }

            for task_type, index in self._index.items():
                index_name = f"{retriever_model_id}-{task_type.value}-index"
                logger.info(f"Saving index to {output_dir / index_name}")
                index.save_pretrained(
                    output_dir / index_name,
                    push_to_hub=push_to_hub,
                    organization=organization,
                    **kwargs,
                )
                index_names[task_type] = index_name

            # reader
            reader_model_id = reader_model_id or f"reader-{model_id}"
            logger.info(f"Saving reader to {output_dir / reader_model_id}")
            self.reader.save_pretrained(
                output_dir / reader_model_id,
                push_to_hub=push_to_hub,
                organization=organization,
                **kwargs,
            )

            if push_to_hub:
                user = organization or get_logged_in_username()
                # we need to update the config with the model ids that will
                # result from the push to hub
                for task_type, retriever_names in retrievers_names.items():
                    retriever_names["question_encoder_name"] = (
                        f"{user}/{retriever_names['question_encoder_name']}"
                    )
                    retriever_names["passage_encoder_name"] = (
                        f"{user}/{retriever_names['passage_encoder_name']}"
                    )
                    # retriever_names["document_index_name"] = (
                    #     f"{user}/{retriever_names['document_index_name']}"
                    # )
                for task_type, index_name in index_names.items():
                    if index_name is not None:
                        index_names[task_type] = f"{user}/{index_name}"
                # question_encoder_name = f"{user}/{question_encoder_name}"
                # passage_encoder_name = f"{user}/{passage_encoder_name}"
                # document_index_name = f"{user}/{document_index_name}"
                reader_model_id = f"{user}/{reader_model_id}"
            else:
                for task_type, retriever_names in retrievers_names.items():
                    retriever_names["question_encoder_name"] = (
                        output_dir / retriever_names["question_encoder_name"]
                    )
                    retriever_names["passage_encoder_name"] = (
                        output_dir / retriever_names["passage_encoder_name"]
                    )
                    # retriever_names["document_index_name"] = (
                    #     output_dir / retriever_names["document_index_name"]
                    # )
                reader_model_id = output_dir / reader_model_id
        else:
            # save config only
            for task_type, retriever_names in retrievers_names.items():
                retriever = self._retriever.get(task_type, None)
                if retriever is None:
                    continue
                retriever_names["question_encoder_name"] = (
                    retriever.question_encoder.name_or_path
                )
                retriever_names["passage_encoder_name"] = (
                    retriever.passage_encoder.name_or_path
                )
                # retriever_names["document_index_name"] = (
                #     retriever.document_index.name_or_path
                # )

            for task_type, index in self._index.items():
                index_names[task_type] = index.name_or_path

            reader_model_id = self.reader.name_or_path

        if config is None:
            # create a default config
            config = {
                "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}"
            }
            if self._retriever is not None:
                config["retriever"] = {}
                for task_type, retriever in self._retriever.items():
                    if retriever is None:
                        continue
                    config["retriever"][task_type.value] = {
                        "_target_": f"{retriever.__class__.__module__}.{retriever.__class__.__name__}",
                    }
                    if retriever.question_encoder is not None:
                        config["retriever"][task_type.value]["question_encoder"] = (
                            retrievers_names[task_type]["question_encoder_name"]
                        )
                    if (
                        retriever.passage_encoder is not None
                        and not retriever.passage_encoder_is_question_encoder
                    ):
                        config["retriever"][task_type.value]["passage_encoder"] = (
                            retrievers_names[task_type]["passage_encoder_name"]
                        )
                    # if retriever.document_index is not None:
                    #     config["retriever"][task_type.value]["document_index"] = (
                    #         retrievers_names[task_type]["document_index_name"]
                    #     )
            if self._index is not None:
                config["index"] = {}
                for task_type, index in self._index.items():
                    if index is None:
                        continue
                    config["index"][task_type.value] = {
                        "_target_": f"{index.__class__.__module__}.{index.__class__.__name__}.from_pretrained",
                    }
                    config["index"][task_type.value][
                        "name_or_path"
                    ] = index.name_or_path
            if self.reader is not None:
                config["reader"] = {
                    "_target_": f"{self.reader.__class__.__module__}.{self.reader.__class__.__name__}",
                    "transformer_model": reader_model_id,
                }

            # these are model-specific and should be saved
            config["task"] = self.task
            config["metadata_fields"] = self.metadata_fields
            config["top_k"] = self.top_k
            config["window_size"] = self.window_size
            config["window_stride"] = self.window_stride

        config_file_name = config_file_name or CONFIG_NAME

        logger.info(f"Saving relik config to {output_dir / config_file_name}")
        # pretty print the config
        logger.info(pformat(config))
        OmegaConf.save(config, output_dir / config_file_name)

        if push_to_hub:
            # push to hub
            logger.info("Pushing to hub")
            model_id = model_id or output_dir.name
            upload(
                output_dir,
                model_id,
                filenames=[config_file_name],
                organization=organization,
                repo_name=repo_name,
            )
