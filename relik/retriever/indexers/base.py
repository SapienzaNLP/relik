import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import hydra
import numpy
import torch
from omegaconf import OmegaConf
from pprintpp import pformat

from relik.common.log import get_logger
from relik.common.upload import upload
from relik.common.utils import (
    from_cache,
    is_package_available,
    is_str_a_path,
    relative_to_absolute_path,
    to_config,
)
from relik.retriever.indexers.document import Document, DocumentStore

logger = get_logger(__name__)


@dataclass
class IndexerOutput:
    indices: Union[torch.Tensor, numpy.ndarray]
    distances: Union[torch.Tensor, numpy.ndarray]


class BaseDocumentIndex:
    """
    Base class for document indexes.

    Args:
        documents (:obj:`str`, :obj:`List[str]`, :obj:`os.PathLike`, :obj:`List[os.PathLike]`, :obj:`DocumentStore`, `optional`):
            The documents to index. If `None`, an empty document store will be created. Defaults to `None`.
        embeddings (:obj:`torch.Tensor`, `optional`):
            The embeddings of the documents. If `None`, the documents will not be indexed. Defaults to `None`.
        name_or_path (:obj:`str`, :obj:`os.PathLike`, `optional`):
            The name or directory of the retriever.
    """

    CONFIG_NAME = "config.yaml"
    DOCUMENTS_FILE_NAME = "documents.jsonl"
    EMBEDDINGS_FILE_NAME = "embeddings.pt"
    INDEX_FILE_NAME = "index.faiss"

    def __init__(
        self,
        documents: (
            str | List[str] | os.PathLike | List[os.PathLike] | DocumentStore | None
        ) = None,
        embeddings: torch.Tensor | None = None,
        metadata_fields: List[str] | None = None,
        separator: str | None = None,
        name_or_path: str | os.PathLike | None = None,
        device: str = "cpu",
    ) -> None:
        if metadata_fields is None:
            metadata_fields = []

        if device is None:
            device = "cpu"

        self.metadata_fields = metadata_fields
        self.separator = separator

        self.document_path: List[str | os.PathLike] = []

        if documents is not None:
            if isinstance(documents, DocumentStore):
                self.documents = documents
            else:
                documents_are_paths = False

                # normalize the documents to list if not already
                if not isinstance(documents, list):
                    documents = [documents]

                # now check if the documents are a list of paths (either str or os.PathLike)
                if isinstance(documents[0], str) or isinstance(
                    documents[0], os.PathLike
                ):
                    # check if the str is a path
                    documents_are_paths = is_str_a_path(documents[0])

                # if the documents are a list of paths, then we load them
                if documents_are_paths:
                    logger.info("Loading documents from paths")
                    _documents = []
                    for doc in documents:
                        with open(relative_to_absolute_path(doc)) as f:
                            self.document_path.append(doc)
                            _documents += [
                                Document.from_dict(json.loads(line))
                                for line in f.readlines()
                            ]
                    # remove duplicates
                    documents = _documents

                self.documents = DocumentStore(documents)
        else:
            self.documents = DocumentStore()

        self.embeddings = embeddings
        self.name_or_path = name_or_path

        # store the device in case embeddings are not provided
        self.device_in_init = device

    def __iter__(self):
        # make this class iterable
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.get_passage_from_index(index)

    def to(
        self, device_or_precision: str | torch.device | torch.dtype
    ) -> "BaseDocumentIndex":
        """
        Move the retriever to the specified device or precision.

        Args:
            device_or_precision (`str` | `torch.device` | `torch.dtype`):
                The device or precision to move the retriever to.

        Returns:
            `BaseDocumentIndex`: The retriever.
        """
        if self.embeddings is not None:
            if isinstance(device_or_precision, torch.dtype) and self.device != "cpu":
                # if the device is a dtype, then we need to move the embeddings to cpu
                # first before converting to the dtype to avoid OOM
                previous_device = self.embeddings.device
                self.embeddings = self.embeddings.cpu()
                self.embeddings = self.embeddings.to(device_or_precision)
                self.embeddings = self.embeddings.to(previous_device)
            else:
                if isinstance(device_or_precision, torch.device):
                    self.embeddings = self.embeddings.to(device_or_precision)
                else:
                    if (
                        device_or_precision != self.embeddings.dtype
                        and self.device != "cpu"
                    ):
                        self.embeddings = self.embeddings.to(device_or_precision)
                # self.embeddings = self.embeddings.to(device_or_precision)
        return self

    @property
    def device(self):
        return (
            self.embeddings.device
            if self.embeddings is not None
            else self.device_in_init
        )

    @property
    def config(self) -> Dict[str, Any]:
        """
        The configuration of the document index.

        Returns:
            `Dict[str, Any]`: The configuration of the retriever.
        """

        config = {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "metadata_fields": self.metadata_fields,
            "separator": self.separator,
            "name_or_path": self.name_or_path,
        }
        if len(self.document_path) > 0:
            config["documents"] = self.document_path
        return config

    def index(
        self,
        retriever,
        *args,
        **kwargs,
    ) -> "BaseDocumentIndex":
        raise NotImplementedError

    def search(self, query: Any, k: int = 1, *args, **kwargs) -> List:
        raise NotImplementedError

    def get_document_from_passage(self, passage: str) -> Document | None:
        """
        Get the document label from the passage.

        Args:
            passage (`str`):
                The document to get the label for.

        Returns:
            `str`: The document label.
        """
        # get the text from the document
        if self.separator:
            text = passage.split(self.separator)[0]
        else:
            text = passage
        return self.documents.get_document_from_text(text)

    def get_index_from_passage(self, passage: str) -> int:
        """
        Get the index of the passage.

        Args:
            passage (`str`):
                The document to get the index for.

        Returns:
            `int`: The index of the document.
        """
        # get the text from the document
        doc = self.get_document_from_passage(passage)
        if doc is None:
            raise ValueError(f"Document `{passage}` not found.")
        return doc.id

    def get_document_from_index(self, index: int) -> Document | None:
        """
        Get the document from the index.

        Args:
            index (`int`):
                The index of the document.

        Returns:
            `str`: The document.
        """
        return self.documents.get_document_from_id(index)

    def get_passage_from_index(self, index: int) -> str:
        """
        Get the document from the index.

        Args:
            index (`int`):
                The index of the document.

        Returns:
            `str`: The document.
        """
        document = self.get_document_from_index(index)
        # build the passage using the metadata fields
        passage = document.text
        for field in self.metadata_fields:
            passage += f"{self.separator}{document.metadata[field]}"
        return passage

    def get_passage_from_document(self, document: Document) -> str:
        passage = document.text
        for field in self.metadata_fields:
            passage += f"{self.separator}{document.metadata[field]}"
        return passage

    def get_embeddings_from_index(self, index: int) -> torch.Tensor:
        """
        Get the document vector from the index.

        Args:
            index (`int`):
                The index of the document.

        Returns:
            `torch.Tensor`: The document vector.
        """
        if self.embeddings is None:
            raise ValueError(
                "The documents must be indexed before they can be retrieved."
            )
        if index >= self.embeddings.shape[0]:
            raise ValueError(
                f"The index {index} is out of bounds. The maximum index is {len(self.embeddings) - 1}."
            )
        return self.embeddings[index]

    def get_embeddings_from_passage(self, document: str) -> torch.Tensor:
        """
        Get the document vector from the document label.

        Args:
            document (`str`):
                The document to get the vector for.

        Returns:
            `torch.Tensor`: The document vector.
        """
        if self.embeddings is None:
            raise ValueError(
                "The documents must be indexed before they can be retrieved."
            )
        return self.get_embeddings_from_index(self.get_index_from_passage(document))

    def get_embeddings_from_document(self, document: str) -> torch.Tensor:
        """
        Get the document vector from the document label.

        Args:
            document (`str`):
                The document to get the vector for.

        Returns:
            `torch.Tensor`: The document vector.
        """
        if self.embeddings is None:
            raise ValueError(
                "The documents must be indexed before they can be retrieved."
            )
        return self.get_embeddings_from_index(self.get_index_from_document(document))

    def get_passages(self, documents: DocumentStore | None = None) -> List[str]:
        """
        Get the passages from the document store.

        Returns:
            `List[str]`: The passages.
        """
        documents = documents or self.documents
        # construct the passages from the documents
        # return [self.get_passage_from_index(i) for i in range(len(documents))]
        return [self.get_passage_from_document(doc) for doc in documents]

    def save_pretrained(
        self,
        output_dir: Union[str, os.PathLike],
        config: Optional[Dict[str, Any]] = None,
        config_file_name: str | None = None,
        document_file_name: str | None = None,
        embedding_file_name: str | None = None,
        push_to_hub: bool = False,
        model_id: str | None = None,
        **kwargs,
    ):
        """
        Save the retriever to a directory.

        Args:
            output_dir (`str`):
                The directory to save the retriever to.
            config (`Optional[Dict[str, Any]]`, `optional`):
                The configuration to save. If `None`, the current configuration of the retriever will be
                saved. Defaults to `None`.
            config_file_name (`str | None`, `optional`):
                The name of the configuration file. Defaults to `config.yaml`.
            document_file_name (`str | None`, `optional`):
                The name of the document file. Defaults to `documents.json`.
            embedding_file_name (`str | None`, `optional`):
                The name of the embedding file. Defaults to `embeddings.pt`.
            push_to_hub (`bool`, `optional`):
                Whether to push the saved retriever to the hub. Defaults to `False`.
            model_id (`str | None`, `optional`):
                The id of the model to push to the hub. If `None`, the name of the output
                directory will be used. Defaults to `None`.
            **kwargs:
                Additional keyword arguments to pass to `upload`.
        """
        if config is None:
            # create a default config
            config = self.config

        config_file_name = config_file_name or self.CONFIG_NAME
        document_file_name = document_file_name or self.DOCUMENTS_FILE_NAME
        embedding_file_name = embedding_file_name or self.EMBEDDINGS_FILE_NAME

        # create the output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving retriever to {output_dir}")
        logger.info(f"Saving config to {output_dir / config_file_name}")
        # pretty print the config
        OmegaConf.save(config, output_dir / config_file_name)
        logger.info(pformat(config))

        # save the current state of the retriever
        embedding_path = output_dir / embedding_file_name
        logger.info(f"Saving retriever state to {output_dir / embedding_path}")
        torch.save(self.embeddings, embedding_path)

        # save the passage index
        documents_path = output_dir / document_file_name
        logger.info(f"Saving passage index to {documents_path}")
        self.documents.save(documents_path)

        logger.info("Saving document index to disk done.")

        if push_to_hub:
            # push to hub
            logger.info("Pushing to hub")
            model_id = model_id or output_dir.name
            upload(output_dir, model_id, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: Union[str, os.PathLike],
        device: str = "cpu",
        precision: str | None = None,
        config_file_name: str | None = None,
        document_file_name: str | None = None,
        embedding_file_name: str | None = None,
        index_file_name: str | None = None,
        *args,
        **kwargs,
    ) -> "BaseDocumentIndex":
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)

        config_file_name = config_file_name or cls.CONFIG_NAME
        document_file_name = document_file_name or cls.DOCUMENTS_FILE_NAME
        embedding_file_name = embedding_file_name or cls.EMBEDDINGS_FILE_NAME
        index_file_name = index_file_name or cls.INDEX_FILE_NAME

        model_dir = from_cache(
            name_or_path,
            filenames=[config_file_name, document_file_name, embedding_file_name],
            cache_dir=cache_dir,
            force_download=force_download,
        )

        config_path = model_dir / config_file_name
        if not config_path.exists():
            raise FileNotFoundError(
                f"Model configuration file not found at {config_path}."
            )

        config = OmegaConf.load(config_path)
        # add the actual cls class to the config in place of the _target_ if cls is not BaseDocumentIndex
        if cls.__name__ != "BaseDocumentIndex":
            kwargs["_target_"] = f"{cls.__module__}.{cls.__name__}"
        # override the config with the kwargs
        config = OmegaConf.merge(config, OmegaConf.create(kwargs))
        logger.info("Loading Index from config:")
        logger.info(pformat(OmegaConf.to_container(config)))

        # load the documents
        documents_path = model_dir / document_file_name

        if not documents_path.exists():
            raise ValueError(f"Document file `{documents_path}` does not exist.")
        logger.info(f"Loading documents from {documents_path}")
        documents = DocumentStore.from_file(documents_path)
        # TODO: probably is better to do the opposite and iterate over the config
        # check for each possible attribute ind DocumentStore
        for attr in dir(documents):
            if attr.startswith("__"):
                continue
            if attr not in config:
                continue
            # set the attribute
            setattr(documents, attr, config[attr])

        # base variables
        index = None
        embeddings = None
        index_path = model_dir / index_file_name
        embedding_path = model_dir / embedding_file_name
        # boolean variables to check if the index and embeddings exist
        index_exists = index_path.exists()
        embeddings_exists = embedding_path.exists()
        use_faiss = "FaissDocumentIndex" in cls.__name__

        if use_faiss:
            if index_exists:
                logger.info(f"Loading index from {index_path}")
                if is_package_available("faiss"):
                    import faiss

                    index = faiss.read_index(str(index_path))
                else:
                    raise ImportError(
                        "To load a FAISS index, the `faiss` package must be installed."
                        "You can install it with `pip install relik[faiss]`."
                        "Otherwise, you can load the index with a different class."
                    )
            else:
                if embeddings_exists:
                    logger.warning(
                        "FAISS index file does not exist, but a torch embeddings file was found. "
                        "We will create a new index from the embeddings."
                    )
        
        if embeddings_exists:
            if index is None:
                logger.info(f"Loading embeddings from {embedding_path}")
                embeddings = torch.load(embedding_path, map_location="cpu")
        else:
            if index_exists and not use_faiss:
                logger.warning(
                    "An embeddings file was not found, but a FAISS index file was found instead. "
                    "If you want to use it, try the `FaissDocumentIndex` class."
                )

        if not index_exists and not embeddings_exists:
            logger.warning(
                "No index or embeddings found in the directory. Remember to index the documents before using the retriever."
            )

        document_index = hydra.utils.instantiate(
            config,
            documents=documents,
            embeddings=embeddings,
            index=index,
            device=device,
            precision=precision,
            name_or_path=name_or_path,
            _convert_="partial",
            *args,
            **kwargs,
        )

        return document_index
