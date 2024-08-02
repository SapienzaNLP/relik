from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from relik.common.log import get_logger
from relik.inference.data.objects import TaskType
from relik.reader.pytorch_modules.base import RelikReaderBase
from relik.retriever.indexers.base import BaseDocumentIndex
from relik.retriever.pytorch_modules import PRECISION_MAP
from relik.retriever.pytorch_modules.model import GoldenRetriever

logger = get_logger(__name__)


def _instantiate_retriever(
    retriever: GoldenRetriever | DictConfig | Dict,
    device: str | None,
    precision: int | str | torch.dtype | None,
    **kwargs: Any,
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
            device=device,
            precision=precision,
            **kwargs,
        )
    else:
        if device is not None:
            logger.info(f"Moving retriever to `{device}`.")
            retriever.to(device)
        if precision is not None:
            logger.info(
                f"Setting precision of retriever to `{PRECISION_MAP[precision]}`."
            )
            retriever.to(PRECISION_MAP[precision])
    retriever.training = False
    retriever.eval()
    return retriever


def load_retriever(
    retriever: GoldenRetriever | DictConfig | Dict | str,
    device: str | None | torch.device | int = None,
    precision: int | str | torch.dtype | None = None,
    task: TaskType | str | None = None,
    compile: bool = False,
    **kwargs,
) -> Dict[TaskType, GoldenRetriever]:
    """
    Load and instantiate retrievers for a given task.

    Args:
        retriever (GoldenRetriever | DictConfig | Dict):
            The retriever object or configuration.
        device (str | None):
            The device to load the retriever on.
        precision (int | str | torch.dtype | None):
            The precision of the retriever.
        task (TaskType):
            The task type for the retriever.
        compile (bool, optional):
            Whether to compile the retriever. Defaults to False.
        **kwargs:
            Additional keyword arguments to be passed to the retriever instantiation.

    Returns:
        Dict[TaskType, GoldenRetriever]:
            A dictionary containing the instantiated retrievers.

    Raises:
        ValueError: If the `retriever` argument is not of type `GoldenRetriever`, `DictConfig`, or `Dict`.
        ValueError: If the `retriever` argument is a `DictConfig` without the `_target_` key.
        ValueError: If the task type is not valid for each retriever in the `DictConfig`.

    """

    # retriever section
    _retriever: Dict[TaskType, GoldenRetriever] = {
        TaskType.SPAN: None,
        TaskType.TRIPLET: None,
    }

    # check retriever type, it can be a GoldenRetriever, a DictConfig or a Dict
    if not isinstance(retriever, (GoldenRetriever, DictConfig, Dict, str)):
        raise ValueError(
            f"`retriever` must be a `GoldenRetriever`, a `DictConfig`, "
            f"a `Dict`, or a `str`, got `{type(retriever)}`."
        )
    if isinstance(retriever, str):
        logger.warning(
            "Using a string to instantiate the retriever. "
            f"We will use the same model `{retriever}` for both query and passage encoder. "
            "If you want to use different models, please provide a dictionary with keys `question_encoder` and `passage_encoder`."
        )
        retriever = {
            "question_encoder": retriever,
            "_target_": "relik.retriever.pytorch_modules.model.GoldenRetriever",
        }
    # we need to check weather the DictConfig is a DictConfig for an instance of GoldenRetriever
    # or a primitive Dict
    # if isinstance(retriever, (DictConfig, Dict)):
    #     # then it is probably a primitive Dict
    #     if "_target_" not in retriever:
    #         retriever["_target_"] = "relik.retriever.pytorch_modules.model.GoldenRetriever"
    #         retriever = OmegaConf.to_container(retriever, resolve=True)
    #         # convert the key to TaskType
    #         try:
    #             retriever = {TaskType(k.lower()): v for k, v in retriever.items()}
    #         except ValueError as e:
    #             raise ValueError(
    #                 f"Please choose a valid task type (one of {list(TaskType)}) for each retriever."
    #             ) from e

    try:
        # convert the key to TaskType
        retriever = {TaskType(k): v for k, v in retriever.items()}
    except ValueError as e:
        retriever = {task: retriever}

    # instantiate each retriever
    if task in [TaskType.SPAN, TaskType.BOTH]:
        _retriever[TaskType.SPAN] = _instantiate_retriever(
            retriever[TaskType.SPAN],
            device,
            precision,
            **kwargs,
        )
    if task in [TaskType.TRIPLET, TaskType.BOTH]:
        _retriever[TaskType.TRIPLET] = _instantiate_retriever(
            retriever[TaskType.TRIPLET],
            device,
            precision,
            **kwargs,
        )

    # clean up None retrievers from the dictionary
    _retriever = {task_type: r for task_type, r in _retriever.items() if r is not None}
    if compile:
        # torch compile
        _retriever = {
            task_type: torch.compile(r) for task_type, r in _retriever.items()
        }

    return _retriever


def _instantiate_index(
    index: BaseDocumentIndex | DictConfig | Dict,
    device: str | None | torch.device | int = None,
    precision: int | str | torch.dtype | None = None,
    **kwargs: Dict[str, Any],
) -> BaseDocumentIndex:
    """
    Instantiate a document index.

    Args:
        index (`BaseDocumentIndex`, `DictConfig` or `Dict`):
            The document index to instantiate.
        device (`str`, `optional`):
            The device to use for the document index.
        precision (`int`, `str` or `torch.dtype`, `optional`):
            The precision to use for the document index.
        kwargs (`Dict[str, Any]`, `optional`):
            Additional keyword arguments to pass to the document index.

    Returns:
        `BaseDocumentIndex`:
            The instantiated document index.
    """
    if not isinstance(index, BaseDocumentIndex):
        index = OmegaConf.create(index)
        # use_faiss = kwargs.get("use_faiss", False)
        if "use_faiss" not in kwargs:
            kwargs["use_faiss"] = False
        # if use_faiss:
        #     index = OmegaConf.merge(
        #         index,
        #         {
        #             "_target_": "relik.retriever.indexers.faissindex.FaissDocumentIndex.from_pretrained",
        #         },
        #     )
        # else:
        # index = OmegaConf.merge(
        #     index,
        #     {
        #         "_target_": "relik.retriever.indexers.base.BaseDocumentIndex.from_pretrained",
        #     },
        # )
        if device is not None:
            kwargs["device"] = device
        if precision is not None:
            kwargs["precision"] = precision

        if "_target_" not in index:
            index["_target_"] = (
                "relik.retriever.indexers.base.BaseDocumentIndex.from_pretrained"
            )

        # merge the kwargs
        index = OmegaConf.merge(index, OmegaConf.create(kwargs))
        index: BaseDocumentIndex = hydra.utils.instantiate(index)
    else:
        index = index
        if device is not None:
            logger.info(f"Moving index to `{device}`.")
            index.to(device)
        if precision is not None:
            logger.info(f"Setting precision of index to `{PRECISION_MAP[precision]}`.")
            index.to(PRECISION_MAP[precision])
    return index


def load_index(
    index: BaseDocumentIndex | DictConfig | Dict | str,
    device: str | None,
    precision: int | str | torch.dtype | None,
    task: TaskType,
    **kwargs,
) -> Dict[TaskType, BaseDocumentIndex]:
    """
    Load the document index based on the specified parameters.

    Args:
        index (BaseDocumentIndex | DictConfig | Dict):
            The document index to load. It can be an instance of `BaseDocumentIndex`, a `DictConfig`, or a `Dict`.
        device (str | None):
            The device to use for loading the index. If `None`, the default device will be used.
        precision (int | str | torch.dtype | None):
            The precision of the index. If `None`, the default precision will be used.
        task (TaskType):
            The type of task for the index.
        **kwargs:
            Additional keyword arguments to be passed to the index instantiation.

    Returns:
        Dict[TaskType, BaseDocumentIndex]:
            A dictionary containing the loaded document index for each task type.

    Raises:
        ValueError: If the `index` parameter is not of type `BaseDocumentIndex`, `DictConfig`, or `Dict`.
        ValueError: If the `index` parameter is a `DictConfig` without a `_target_` key.
        ValueError: If the task type specified in the `index` parameter is not valid.
    """

    # index
    _index: Dict[TaskType, BaseDocumentIndex] = {
        TaskType.SPAN: None,
        TaskType.TRIPLET: None,
    }

    # check retriever type, it can be a BaseDocumentIndex, a DictConfig or a Dict
    if not isinstance(index, (BaseDocumentIndex, DictConfig, Dict, str)):
        raise ValueError(
            f"`index` must be a `BaseDocumentIndex`, a `DictConfig`, "
            f"a `Dict`, or a `str`, got `{type(index)}`."
        )
    # we need to check weather the DictConfig is a DictConfig for an instance of BaseDocumentIndex
    # or a primitive Dict
    if isinstance(index, str):
        # use_faiss = kwargs.get("use_faiss", False)
        index = {"name_or_path": index}
        # if use_faiss:
        #     index["_target_"] = "relik.retriever.indexers.faissindex.FaissDocumentIndex.from_pretrained"

    # if isinstance(index, DictConfig):
    #     # then it is probably a primitive Dict
    #     if "_target_" not in index:
    #         index = OmegaConf.to_container(index, resolve=True)
    #         # convert the key to TaskType
    #         try:
    #             index = {TaskType(k.lower()): v for k, v in index.items()}
    #         except ValueError as e:
    #             raise ValueError(
    #                 f"Please choose a valid task type (one of {list(TaskType)}) for each index."
    #             ) from e

    try:
        # convert the key to TaskType
        index = {TaskType(k): v for k, v in index.items()}
    except ValueError as e:
        index = {task: index}

    # instantiate each retriever
    if task in [TaskType.SPAN, TaskType.BOTH]:
        _index[TaskType.SPAN] = _instantiate_index(
            index[TaskType.SPAN],
            device,
            precision,
            **kwargs,
        )
    if task in [TaskType.TRIPLET, TaskType.BOTH]:
        _index[TaskType.TRIPLET] = _instantiate_index(
            index[TaskType.TRIPLET],
            device,
            precision,
            **kwargs,
        )
    # clean up None retrievers from the dictionary
    _index = {task_type: i for task_type, i in _index.items() if i is not None}
    return _index


def load_reader(
    reader: RelikReaderBase,
    device: str | None,
    precision: int | str | torch.dtype | None,
    compile: bool = False,
    **kwargs: Dict[str, Any],
) -> RelikReaderBase:
    """
    Load a reader model for inference.

    Args:
        reader (RelikReaderBase):
            The reader model to load.
        device (str | None):
            The device to move the reader model to.
        precision (int | str | torch.dtype | None):
            The precision to set for the reader model.
        compile (bool, optional):
            Whether to compile the reader model. Defaults to False.
        **kwargs (Dict[str, Any]):
            Additional keyword arguments to pass to the reader model.

    Returns:
        RelikReaderBase: The loaded reader model.
    """

    if not isinstance(reader, (RelikReaderBase, DictConfig, Dict, str)):
        raise ValueError(
            f"`reader` must be a `RelikReaderBase`, a `DictConfig`, "
            f"a `Dict`, or a `str`, got `{type(reader)}`."
        )

    if isinstance(reader, str):
        reader = {
            "_target_": "relik.reader.pytorch_modules.base.RelikReaderBase.from_pretrained",
            "model_name_or_dir": reader,
        }

    if not isinstance(reader, DictConfig):
        # then it is probably a primitive Dict
        # if "_target_" not in reader:
        #     reader = OmegaConf.to_container(reader, resolve=True)
        # reader = OmegaConf.to_container(reader, resolve=True)
        # if not isinstance(reader, DictConfig):
        reader = OmegaConf.create(reader)

    reader = (
        hydra.utils.instantiate(
            reader,
            device=device,
            precision=precision,
            **kwargs,
        )
        if isinstance(reader, DictConfig)
        else reader
    )
    reader.training = False
    reader.eval()
    if device is not None:
        logger.info(f"Moving reader to `{device}`.")
        reader.to(device)
    if precision is not None and reader.precision != PRECISION_MAP[precision]:
        logger.info(f"Setting precision of reader to `{PRECISION_MAP[precision]}`.")
        reader.to(PRECISION_MAP[precision])

    if compile:
        reader = torch.compile(reader)
    return reader
