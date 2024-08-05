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
    Triplets,
)
from relik.inference.data.splitters.base_sentence_splitter import BaseSentenceSplitter
from relik.inference.data.splitters.blank_sentence_splitter import BlankSentenceSplitter
from relik.inference.data.splitters.spacy_sentence_splitter import SpacySentenceSplitter
from relik.inference.data.splitters.window_based_splitter import WindowSentenceSplitter
from relik.inference.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from relik.inference.data.window.manager import WindowManager
from relik.inference.utils import load_index, load_reader, load_retriever
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
        retriever: GoldenRetriever | None = None,
        index: BaseDocumentIndex | None = None,
        reader: RelikReaderBase | None = None,
        task: TaskType | str = TaskType.SPAN,
        metadata_fields: list[str] | None = None,
        top_k: int | None = None,
        window_size: int | str | None = None,
        window_stride: int | None = None,
        **kwargs,
    ) -> None:
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

        # retriever section
        self._retriever: Dict[TaskType, GoldenRetriever] = {
            TaskType.SPAN: None,
            TaskType.TRIPLET: None,
        }
        if retriever:
            if isinstance(retriever, GoldenRetriever):
                if self.task in [TaskType.SPAN, TaskType.BOTH]:
                    self._retriever[TaskType.SPAN] = retriever
                if self.task in [TaskType.TRIPLET, TaskType.BOTH]:
                    self._retriever[TaskType.TRIPLET] = retriever
                # check if both retrievers are the same
                if self._retriever[TaskType.SPAN] == self._retriever[TaskType.TRIPLET]:
                    logger.warning("The retriever is the same for both tasks.")
            elif isinstance(retriever, (Dict, DictConfig)):
                for task_type, r in retriever.items():
                    # convert task_type to TaskType
                    if isinstance(task_type, str):
                        try:
                            task_type = TaskType(task_type.lower())
                        except ValueError:
                            raise ValueError(
                                f"Task `{task_type}` not recognized. "
                                f"Please choose one of {list(TaskType)}."
                            )
                    self._retriever[task_type] = r
            else:
                raise ValueError(
                    f"Invalid retriever type {type(retriever)}. "
                    f"Please provide a `GoldenRetriever` or a dictionary of retrievers."
                )

        self._retriever = {
            task_type: r for task_type, r in self._retriever.items() if r is not None
        }
        self._index: Dict[TaskType, GoldenRetriever] = {
            TaskType.SPAN: None,
            TaskType.TRIPLET: None,
        }

        if index is None:
            # check if the retriever has an index
            for task_type, r in self._retriever.items():
                if r is not None:
                    if r.document_index is None:
                        raise ValueError(
                            f"No index found for task `{task_type}` in the retriever."
                        )
                    self._index[task_type] = r.document_index
        elif isinstance(index, BaseDocumentIndex):
            if self.task in [TaskType.SPAN, TaskType.BOTH]:
                self._index[TaskType.SPAN] = index
            if self.task in [TaskType.TRIPLET, TaskType.BOTH]:
                self._index[TaskType.TRIPLET] = index

            # check if both retrievers are the same
            if self._index[TaskType.SPAN] == self._index[TaskType.TRIPLET]:
                logger.warning("The index is the same for both tasks.")

        elif isinstance(index, (Dict, DictConfig)):
            for task_type, i in index.items():
                # convert task_type to TaskType
                if isinstance(task_type, str):
                    try:
                        task_type = TaskType(task_type.lower())
                    except ValueError:
                        raise ValueError(
                            f"Task `{task_type}` not recognized. "
                            f"Please choose one of {list(TaskType)}."
                        )
                self._index[task_type] = i
        else:
            raise ValueError(
                f"Invalid index type {type(index)}. "
                f"Please provide a `BaseDocumentIndex` or a dictionary of indexes."
            )
        self._index = {
            task_type: i for task_type, i in self._index.items() if i is not None
        }

        # if isinstance(index, BaseDocumentIndex):
        #     index = {self.task: index}

        # self._retriever = load_retriever(retriever)
        # self._index = load_index(index)
        # self._index = index or {}
        self.reader = reader

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
            use_doc_topic (`bool`, `optional`, defaults to `False`):
                Whether to use the document topic for the retriever in each window. Used by some EL systems where the first word of the document as the topic.
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
            else:
                if isinstance(self.sentence_splitter, WindowSentenceSplitter):
                    if not isinstance(window_size, int):
                        logger.warning(
                            "With WindowSentenceSplitter the window_size must be an integer. "
                            f"Using the default window size {self.window_manager.window_size}."
                            f"If you want to change the window size to `sentence` or `none`, "
                            f"please create a new Relik instance."
                        )
                        window_size = self.window_manager.window_size
                        window_stride = self.window_manager.window_stride
                if isinstance(self.sentence_splitter, SpacySentenceSplitter):
                    if window_size != "sentence":
                        logger.warning(
                            "With SpacySentenceSplitter the window_size must be `sentence`. "
                            f"Using the default window size {self.window_manager.window_size}."
                            f"If you want to change the window size to an integer or `none`, "
                            f"please create a new Relik instance."
                        )
                        window_size = "sentence"
                        window_stride = None
                if isinstance(self.sentence_splitter, BlankSentenceSplitter):
                    if window_size != "none" or window_stride is not None:
                        logger.warning(
                            "With BlankSentenceSplitter the window_size must be `none`. "
                            f"Using the default window size {self.window_manager.window_size}."
                            f"If you want to change the window size to an integer or `sentence`, "
                            f"please create a new Relik instance."
                        )
                        window_size = "none"
                        window_stride = None

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
            (
                windows,
                blank_windows,
                documents_tokens,
            ) = self.window_manager.create_windows(
                text,
                window_size,
                window_stride,
                is_split_into_words=is_split_into_words,
                mentions=mentions,
                annotation_type=annotation_type,
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
        # candidates are provided, use them and skip retrieval if they exist for the specific task
        if candidates is not None:
            if isinstance(candidates, Dict):
                for task_type in windows_candidates.keys():
                    if task_type in candidates:
                        _candidates = candidates[task_type]
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
                        if self._retriever is None:
                            raise ValueError(
                                "No retriever was provided, please provide a retriever or candidates."
                            )
                        # retrieve candidates for this task
                        retriever = self._retriever.get(task_type)
                        if retriever is not None:
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
                                [p.document for p in predictions]
                                for predictions in retriever_out
                            ]
            else:
                # check if the candidates are a list of lists
                if not isinstance(candidates[0], list):
                    candidates = [candidates]
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

        # retrieve candidates for tasks without provided candidates
        for task_type, retriever in self._retriever.items():
            if windows_candidates[task_type] is None:
                if retriever is not None:
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
                        [p.document for p in predictions]
                        for predictions in retriever_out
                    ]

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
                window._d["predicted_triplets"] = []

        if self.reader is not None:
            # start_read = time.time()
            windows = self.reader.read(
                samples=windows,
                max_batch_size=reader_batch_size,
                annotation_type=annotation_type,
                progress_bar=progress_bar,
                use_predefined_spans=mentions is not None,
                **kwargs,
            )
            # end_read = time.time()
            # logger.debug(f"Reading took {end_read - start_read} seconds.")

        # replace the reader "text" candidates with the full Document ones
        for task_type, task_candidates in windows_candidates.items():
            for i, task_candidate in enumerate(task_candidates):
                windows[i]._d[f"{task_type.value}_candidates"] = task_candidate

        windows = windows + blank_windows
        windows.sort(key=lambda x: (x.doc_id, x.offset))

        # if there is no reader, just return the windows
        if self.reader is None:
            # normalize window candidates to be a list of lists, like when the reader is used
            merged_windows = [
                self.window_manager._normalize_single_window(w) for w in windows
            ]
        else:
            merged_windows = self.window_manager.merge_windows(windows)

        # transform predictions into RelikOutput objects
        output = []
        for w in merged_windows:
            span_labels = []
            triplets_labels = []
            # span extraction should always be present
            if getattr(w, "predicted_spans", None) is not None:
                span_labels = sorted(
                    [
                        (
                            Span(start=ss, end=se, label=sl, text=text[w.doc_id][ss:se])
                            if annotation_type == AnnotationType.CHAR
                            else Span(
                                start=ss,
                                end=se,
                                label=sl,
                                text=documents_tokens[w.doc_id][ss:se],
                            )
                        )
                        for ss, se, sl in w.predicted_spans
                    ],
                    key=lambda x: x.start,
                )
                # triple extraction is optional, if here add it
                if getattr(w, "predicted_triplets", None) is not None:
                    triplets_labels = [
                        Triplets(
                            subject=span_labels[subj],
                            label=label,
                            object=span_labels[obj],
                            confidence=conf,
                        )
                        for subj, label, obj, conf in w.predicted_triplets
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
                tokens=documents_tokens[w.doc_id],
                id=w.doc_id,
                spans=span_labels,
                triplets=triplets_labels,
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

        device = kwargs.pop("device", None)
        retriever_device = kwargs.pop("retriever_device", device)
        index_device = kwargs.pop("index_device", device)
        reader_device = kwargs.pop("reader_device", device)

        precision = kwargs.pop("precision", None)
        retriever_precision = kwargs.pop("retriever_precision", precision)
        index_precision = kwargs.pop("index_precision", precision)
        reader_precision = kwargs.pop("reader_precision", precision)

        retriever_kwargs = kwargs.pop("retriever_kwargs", {})
        index_kwargs = kwargs.pop("index_kwargs", {})
        reader_kwargs = kwargs.pop("reader_kwargs", {})

        # notable parameters
        if "use_faiss" in kwargs:
            index_kwargs["use_faiss"] = kwargs.pop("use_faiss")
        if "skip_metadata" in kwargs:
            index_kwargs["skip_metadata"] = kwargs.pop("skip_metadata")
        if "use_nme" in kwargs:
            reader_kwargs["use_nme"] = kwargs.pop("use_nme")

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

        config = OmegaConf.load(config_path)
        # do we want to print the config? I like it
        logger.info(f"Loading Relik from {model_name_or_dir}")
        logger.info(pformat(OmegaConf.to_container(config)))

        task = config.task
        # parse task into a TaskType
        if isinstance(task, str):
            try:
                task = TaskType(task.lower())
            except ValueError:
                raise ValueError(
                    f"Task `{task}` not recognized. "
                    f"Please choose one of {list(TaskType)}."
                )
        task = task

        # retriever section
        if "retriever" in kwargs:
            retriever = kwargs.pop("retriever")
        elif hasattr(config, "retriever"):
            retriever = config.pop("retriever")
        else:
            retriever = None

        if retriever:
            retriever = load_retriever(
                retriever,
                retriever_device,
                retriever_precision,
                task,
                **retriever_kwargs,
            )

        # index section
        if retriever is None:
            logger.warning("No retriever was provided, ignoring the index.")
            index = None
        else:
            if "index" in kwargs:
                index = kwargs.pop("index")
            elif hasattr(config, "index"):
                index = config.pop("index")
            else:
                index = None

        if index:
            index = load_index(
                index, index_device, index_precision, task, **index_kwargs
            )

        # link each index to the retriever
        if retriever and index:
            for task_type, r in retriever.items():
                if task_type in index:
                    r.document_index = index[task_type]

        # reader section
        if "reader" in kwargs:
            reader = kwargs.pop("reader")
        elif hasattr(config, "reader"):
            reader = config.pop("reader")
        else:
            reader = None

        if reader:
            reader = load_reader(
                reader, reader_device, reader_precision, **reader_kwargs
            )

        # overwrite config with config_kwargs
        config = OmegaConf.merge(config, OmegaConf.create(kwargs))
        # do we want to print the config? I like it
        # logger.info(f"Loading Relik from {model_name_or_dir}")
        # logger.info(pformat(OmegaConf.to_container(config)))

        # load relik from config
        relik = hydra.utils.instantiate(
            config, _recursive_=False, retriever=retriever, index=index, reader=reader
        )

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
                    "question_encoder_name": question_encoder_name,
                    "passage_encoder_name": passage_encoder_name,
                }

            # index
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
                for task_type, index_name in index_names.items():
                    if index_name is not None:
                        index_names[task_type] = f"{user}/{index_name}"
                reader_model_id = f"{user}/{reader_model_id}"
            else:
                for task_type, retriever_names in retrievers_names.items():
                    retriever_names["question_encoder_name"] = (
                        output_dir / retriever_names["question_encoder_name"]
                    )
                    retriever_names["passage_encoder_name"] = (
                        output_dir / retriever_names["passage_encoder_name"]
                    )
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
