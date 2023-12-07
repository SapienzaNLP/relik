import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy
import psutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from relik.common.log import get_logger
from relik.common.utils import is_package_available
from relik.retriever.common.model_inputs import ModelInputs
from relik.retriever.data.base.datasets import BaseDataset
from relik.retriever.indexers.base import BaseDocumentIndex
from relik.retriever.indexers.document import Document, DocumentStore
from relik.retriever.pytorch_modules import PRECISION_MAP, RetrievedSample
from relik.retriever.pytorch_modules.model import GoldenRetriever

if is_package_available("faiss"):
    import faiss
    import faiss.contrib.torch_utils

logger = get_logger(__name__, level=logging.INFO)


@dataclass
class FaissOutput:
    indices: Union[torch.Tensor, numpy.ndarray]
    distances: Union[torch.Tensor, numpy.ndarray]


class FaissDocumentIndex(BaseDocumentIndex):
    DOCUMENTS_FILE_NAME = "documents.json"
    EMBEDDINGS_FILE_NAME = "embeddings.pt"
    INDEX_FILE_NAME = "index.faiss"

    def __init__(
        self,
        documents: str
        | List[str]
        | os.PathLike
        | List[os.PathLike]
        | DocumentStore
        | None = None,
        embeddings: torch.Tensor | numpy.ndarray | None = None,
        metadata_fields: List[str] | None = None,
        separator: str = "<def>",
        name_or_path: str | os.PathLike | None = None,
        device: str = "cpu",
        index=None,
        index_type: str = "Flat",
        nprobe: int = 1,
        metric: int = faiss.METRIC_INNER_PRODUCT,
        normalize: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            documents, embeddings, metadata_fields, separator, name_or_path, device
        )

        if embeddings is not None and documents is not None:
            logger.info("Both documents and embeddings are provided.")
            if len(documents) != embeddings.shape[0]:
                raise ValueError(
                    "The number of documents and embeddings must be the same."
                )

        faiss.omp_set_num_threads(psutil.cpu_count(logical=False))

        # params
        self.index_type = index_type
        self.metric = metric
        self.normalize = normalize

        if index is not None:
            self.embeddings = index
            if self.device == "cuda":
                # use a single GPU
                faiss_resource = faiss.StandardGpuResources()
                self.embeddings = faiss.index_cpu_to_gpu(
                    faiss_resource, 0, self.embeddings
                )
        else:
            if embeddings is not None:
                # build the faiss index
                logger.info("Building the index from the embeddings.")
                self.embeddings = self._build_faiss_index(
                    embeddings=embeddings,
                    index_type=index_type,
                    nprobe=nprobe,
                    normalize=normalize,
                    metric=metric,
                )

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
        if isinstance(device_or_precision, torch.dtype):
            # raise ValueError(
            #     "FaissDocumentIndex does not support precision conversion."
            # )
            logger.warning(
                "FaissDocumentIndex does not support precision conversion. Ignoring."
            )
        if device_or_precision == "cuda" and self.device == "cpu":
            # use a single GPU
            faiss_resource = faiss.StandardGpuResources()
            self.embeddings = faiss.index_cpu_to_gpu(faiss_resource, 0, self.embeddings)
        elif device_or_precision == "cpu" and self.device == "cuda":
            # move faiss index to CPU
            self.embeddings = faiss.index_gpu_to_cpu(self.embeddings)
        else:
            logger.warning(
                f"Provided device `{device_or_precision}` is the same as the current device `{self.device}`."
            )
        return self

    @property
    def device(self):
        # check if faiss index is on GPU
        if faiss.get_num_gpus() > 0:
            return "cuda"
        return "cpu"

    def _build_faiss_index(
        self,
        embeddings: Optional[Union[torch.Tensor, numpy.ndarray]],
        index_type: str,
        nprobe: int,
        normalize: bool,
        metric: int,
    ):
        # build the faiss index
        self.normalize = (
            normalize
            and metric == faiss.METRIC_INNER_PRODUCT
            and not isinstance(embeddings, torch.Tensor)
        )
        if self.normalize:
            index_type = f"L2norm,{index_type}"
        faiss_vector_size = embeddings.shape[1]
        # if self.device == "cpu":
        #     index_type = index_type.replace("x,", "x_HNSW32,")
        # nlist = math.ceil(math.sqrt(faiss_vector_size)) * 4
        # # nlist = 8
        # index_type = index_type.replace(
        #     "x", str(nlist)
        # )
        # print("Current nlist:", nlist)
        self.embeddings = faiss.index_factory(faiss_vector_size, index_type, metric)

        # convert to GPU
        if self.device == "cuda":
            # use a single GPU
            faiss_resource = faiss.StandardGpuResources()
            self.embeddings = faiss.index_cpu_to_gpu(faiss_resource, 0, self.embeddings)
        else:
            # move to CPU if embeddings is a torch.Tensor
            embeddings = (
                embeddings.cpu() if isinstance(embeddings, torch.Tensor) else embeddings
            )

        # convert to float32 if embeddings is a torch.Tensor and is float16
        if isinstance(embeddings, torch.Tensor) and embeddings.dtype == torch.float16:
            embeddings = embeddings.float()

        logger.info("Training the index.")
        self.embeddings.train(embeddings)

        logger.info("Adding the embeddings to the index.")
        self.embeddings.add(embeddings)

        self.embeddings.nprobe = nprobe

        # save parameters for saving/loading
        self.index_type = index_type
        self.metric = metric

        # clear the embeddings to free up memory
        embeddings = None

        return self.embeddings

    @torch.no_grad()
    @torch.inference_mode()
    def index(
        self,
        retriever: GoldenRetriever,
        documents: Optional[List[Document]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        encoder_precision: Optional[Union[str, int]] = None,
        compute_on_cpu: bool = False,
        force_reindex: bool = False,
        *args,
        **kwargs,
    ) -> "FaissDocumentIndex":
        """
        Index the documents using the encoder.

        Args:
            retriever (:obj:`torch.nn.Module`):
                The encoder to be used for indexing.
            documents (:obj:`List[Document]`, `optional`, defaults to None):
                The documents to be indexed.
            batch_size (:obj:`int`, `optional`, defaults to 32):
                The batch size to be used for indexing.
            num_workers (:obj:`int`, `optional`, defaults to 4):
                The number of workers to be used for indexing.
            max_length (:obj:`int`, `optional`, defaults to None):
                The maximum length of the input to the encoder.
            collate_fn (:obj:`Callable`, `optional`, defaults to None):
                The collate function to be used for batching.
            encoder_precision (:obj:`Union[str, int]`, `optional`, defaults to None):
                The precision to be used for the encoder.
            compute_on_cpu (:obj:`bool`, `optional`, defaults to False):
                Whether to compute the embeddings on CPU.
            force_reindex (:obj:`bool`, `optional`, defaults to False):
                Whether to force reindexing.

        Returns:
            :obj:`InMemoryIndexer`: The indexer object.
        """

        if self.embeddings is not None and not force_reindex:
            logger.log(
                "Embeddings are already present and `force_reindex` is `False`. Skipping indexing."
            )
            if documents is None:
                return self

        # release the memory
        if collate_fn is None:
            tokenizer = retriever.passage_tokenizer

            def collate_fn(x):
                return ModelInputs(
                    tokenizer(
                        x,
                        padding=True,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length or tokenizer.model_max_length,
                    )
                )

        if force_reindex:
            if documents is not None:
                self.documents.add_document(documents)
            data = [k for k in self.get_passages()]

        else:
            if documents is not None:
                data = [k for k in self.get_passages(DocumentStore(documents))]
            else:
                return self

        dataloader = DataLoader(
            BaseDataset(name="passage", data=data),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        encoder = retriever.passage_encoder

        # Create empty lists to store the passage embeddings and passage index
        passage_embeddings: List[torch.Tensor] = []

        encoder_device = "cpu" if compute_on_cpu else self.device

        # fucking autocast only wants pure strings like 'cpu' or 'cuda'
        # we need to convert the model device to that
        device_type_for_autocast = str(encoder_device).split(":")[0]
        # autocast doesn't work with CPU and stuff different from bfloat16
        autocast_pssg_mngr = (
            contextlib.nullcontext()
            if device_type_for_autocast == "cpu"
            else (
                torch.autocast(
                    device_type=device_type_for_autocast,
                    dtype=PRECISION_MAP[encoder_precision],
                )
            )
        )
        with autocast_pssg_mngr:
            # Iterate through each batch in the dataloader
            for batch in tqdm(dataloader, desc="Indexing"):
                # Move the batch to the device
                batch: ModelInputs = batch.to(encoder_device)
                # Compute the passage embeddings
                passage_outs = encoder(**batch)
                # Append the passage embeddings to the list
                if self.device == "cpu":
                    passage_embeddings.extend([c.detach().cpu() for c in passage_outs])
                else:
                    passage_embeddings.extend([c for c in passage_outs])

        # move the passage embeddings to the CPU if not already done
        passage_embeddings = [c.detach().cpu() for c in passage_embeddings]
        # stack it
        passage_embeddings: torch.Tensor = torch.stack(passage_embeddings, dim=0)
        # convert to float32 for faiss
        passage_embeddings.to(PRECISION_MAP["float32"])

        # index the embeddings
        self.embeddings = self._build_faiss_index(
            embeddings=passage_embeddings,
            index_type=self.index_type,
            normalize=self.normalize,
            metric=self.metric,
        )
        # free up memory from the unused variable
        del passage_embeddings

        return self

    @torch.no_grad()
    @torch.inference_mode()
    def search(self, query: torch.Tensor, k: int = 1) -> list[list[RetrievedSample]]:
        k = min(k, self.embeddings.ntotal)

        if self.normalize:
            faiss.normalize_L2(query)
        if isinstance(query, torch.Tensor) and self.device == "cpu":
            query = query.detach().cpu()
        # Retrieve the indices of the top k passage embeddings
        retriever_out = self.embeddings.search(query, k)

        # get int values (second element of the tuple)
        batch_top_k: List[List[int]] = retriever_out[1].detach().cpu().tolist()
        # get float values (first element of the tuple)
        batch_scores: List[List[float]] = retriever_out[0].detach().cpu().tolist()
        # Retrieve the passages corresponding to the indices
        batch_docs = [
            [self.documents.get_document_from_id(i) for i in indices if i != -1]
            for indices in batch_top_k
        ]
        # build the output object
        # build the output object
        batch_retrieved_samples = [
            [
                RetrievedSample(document=doc, score=score)
                for doc, score in zip(docs, scores)
            ]
            for docs, scores in zip(batch_docs, batch_scores)
        ]
        return batch_retrieved_samples

    # def save(self, saving_dir: Union[str, os.PathLike]):
    #     """
    #     Save the indexer to the disk.

    #     Args:
    #         saving_dir (:obj:`Union[str, os.PathLike]`):
    #             The directory where the indexer will be saved.
    #     """
    #     saving_dir = Path(saving_dir)
    #     # save the passage embeddings
    #     index_path = saving_dir / self.INDEX_FILE_NAME
    #     logger.info(f"Saving passage embeddings to {index_path}")
    #     faiss.write_index(self.embeddings, str(index_path))
    #     # save the passage index
    #     documents_path = saving_dir / self.DOCUMENTS_FILE_NAME
    #     logger.info(f"Saving passage index to {documents_path}")
    #     self.documents.save(documents_path)

    # @classmethod
    # def load(
    #     cls,
    #     loading_dir: Union[str, os.PathLike],
    #     device: str = "cpu",
    #     document_file_name: Optional[str] = None,
    #     embedding_file_name: Optional[str] = None,
    #     index_file_name: Optional[str] = None,
    #     **kwargs,
    # ) -> "FaissDocumentIndex":
    #     loading_dir = Path(loading_dir)

    #     document_file_name = document_file_name or cls.DOCUMENTS_FILE_NAME
    #     embedding_file_name = embedding_file_name or cls.EMBEDDINGS_FILE_NAME
    #     index_file_name = index_file_name or cls.INDEX_FILE_NAME

    #     # load the documents
    #     documents_path = loading_dir / document_file_name

    #     if not documents_path.exists():
    #         raise ValueError(f"Document file `{documents_path}` does not exist.")
    #     logger.info(f"Loading documents from {documents_path}")
    #     documents = Labels.from_file(documents_path)

    #     index = None
    #     embeddings = None
    #     # try to load the index directly
    #     index_path = loading_dir / index_file_name
    #     if not index_path.exists():
    #         # try to load the embeddings
    #         embedding_path = loading_dir / embedding_file_name
    #         # run some checks
    #         if embedding_path.exists():
    #             logger.info(f"Loading embeddings from {embedding_path}")
    #             embeddings = torch.load(embedding_path, map_location="cpu")
    #         logger.warning(
    #             f"Index file `{index_path}` and embedding file `{embedding_path}` do not exist."
    #         )
    #     else:
    #         logger.info(f"Loading index from {index_path}")
    #         index = faiss.read_index(str(embedding_path))

    #     return cls(
    #         documents=documents,
    #         embeddings=embeddings,
    #         index=index,
    #         device=device,
    #         **kwargs,
    #     )

    def get_embeddings_from_index(
        self, index: int
    ) -> Union[torch.Tensor, numpy.ndarray]:
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
        if index >= self.embeddings.ntotal:
            raise ValueError(
                f"The index {index} is out of bounds. The maximum index is {self.embeddings.ntotal}."
            )
        return self.embeddings.reconstruct(index)
