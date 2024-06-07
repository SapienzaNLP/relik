import contextlib
import logging
import os
import tempfile
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers as tr

from relik.common.log import get_logger
from relik.common.torch_utils import get_autocast_context
from relik.retriever.common.model_inputs import ModelInputs
from relik.retriever.data.base.datasets import BaseDataset
from relik.retriever.indexers.base import BaseDocumentIndex
from relik.retriever.indexers.document import Document, DocumentStore
from relik.retriever.pytorch_modules import PRECISION_MAP, RetrievedSample


# check if ORT is available
# if is_package_available("onnxruntime"):

logger = get_logger(__name__, level=logging.INFO)


class MatrixMultiplicationModule(torch.nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = torch.nn.Parameter(embeddings, requires_grad=False)

    def forward(self, query):
        return torch.matmul(query, self.embeddings.T)


class InMemoryDocumentIndex(BaseDocumentIndex):

    def __init__(
        self,
        documents: str
        | List[str]
        | os.PathLike
        | List[os.PathLike]
        | DocumentStore
        | None = None,
        embeddings: torch.Tensor | None = None,
        metadata_fields: List[str] | None = None,
        separator: str | None = None,
        name_or_path: str | os.PathLike | None = None,
        device: str = "cpu",
        precision: str | int | torch.dtype = 32,
        *args,
        **kwargs,
    ) -> None:
        """
        An in-memory indexer based on PyTorch.

        Args:
            documents (:obj:`Union[List[str]]`):
                The documents to be indexed.
            embeddings (:obj:`Optional[torch.Tensor]`, `optional`, defaults to :obj:`None`):
                The embeddings of the documents.
            device (:obj:`str`, `optional`, defaults to "cpu"):
                The device to be used for storing the embeddings.
        """

        super().__init__(
            documents, embeddings, metadata_fields, separator, name_or_path, device
        )

        if embeddings is not None and documents is not None:
            logger.info("Both documents and embeddings are provided.")
            if len(documents) != embeddings.shape[0]:
                raise ValueError(
                    "The number of documents and embeddings must be the same. "
                    f"Got {len(documents)} documents and {embeddings.shape[0]} embeddings."
                )

        # # embeddings of the documents
        # self.embeddings = embeddings
        # does this do anything?
        del embeddings
        # convert the embeddings to the desired precision
        if precision is not None:
            if self.embeddings is not None and self.device == "cpu":
                if PRECISION_MAP[precision] == PRECISION_MAP[16]:
                    logger.info(
                        f"Precision `{precision}` is not supported on CPU. "
                        f"Using `{PRECISION_MAP[32]}` instead."
                    )
                precision = 32

            if (
                self.embeddings is not None
                and self.embeddings.dtype != PRECISION_MAP[precision]
            ):
                logger.info(
                    f"Index vectors are of type {self.embeddings.dtype}. "
                    f"Converting to {PRECISION_MAP[precision]}."
                )
                self.embeddings = self.embeddings.to(PRECISION_MAP[precision])
        else:
            # TODO: a bit redundant, fix this eventually
            if (
                # here we trust the device_in_init, since we don't know yet 
                # the device of the embeddings
                (self.device_in_init == "cpu" or self.device_in_init == torch.device("cpu"))
                and self.embeddings is not None
                and self.embeddings.dtype != torch.float32
            ):
                logger.info(
                    f"Index vectors are of type {self.embeddings.dtype} but the device is CPU. "
                    f"Converting to {PRECISION_MAP[32]}."
                )
                self.embeddings = self.embeddings.to(PRECISION_MAP[32])

        # move the embeddings to the desired device
        if self.embeddings is not None and not self.embeddings.device == self.device_in_init:
            self.embeddings = self.embeddings.to(self.device_in_init)

        # TODO: check interactions with the embeddings
        # self.mm = MatrixMultiplicationModule(embeddings=self.embeddings)
        # self.mm.eval()

        # precision to be used for the embeddings
        self.precision = precision

    @torch.no_grad()
    @torch.inference_mode()
    def index(
        self,
        retriever,
        documents: Optional[List[Document]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        max_length: int | None = None,
        collate_fn: Optional[Callable] = None,
        encoder_precision: Optional[Union[str, int]] = None,
        compute_on_cpu: bool = False,
        force_reindex: bool = False,
    ) -> "InMemoryDocumentIndex":
        """
        Index the documents using the encoder.

        Args:
            retriever (:obj:`torch.nn.Module`):
                The encoder to be used for indexing.
            documents (:obj:`List[Document]`, `optional`, defaults to :obj:`None`):
                The documents to be indexed. If not provided, the documents provided at the initialization will be used.
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

        if documents is None and self.documents is None:
            raise ValueError("Documents must be provided.")

        if self.embeddings is not None and not force_reindex and documents is None:
            logger.info(
                "Embeddings are already present and `force_reindex` is `False`. Skipping indexing."
            )
            return self

        if force_reindex:
            if documents is not None:
                self.documents.add_documents(documents)
            data = [k for k in self.get_passages()]

        else:
            if documents is not None:
                data = [k for k in self.get_passages(DocumentStore(documents))]
                # add the documents to the actual document store
                self.documents.add_documents(documents)
            else:
                if self.embeddings is None:
                    data = [k for k in self.get_passages()]

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

        encoder_device = "cpu" if compute_on_cpu else encoder.device

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
                passage_outs = encoder(**batch).pooler_output
                # Append the passage embeddings to the list
                if self.device == "cpu":
                    passage_embeddings.extend([c.detach().cpu() for c in passage_outs])
                else:
                    passage_embeddings.extend([c for c in passage_outs])

        # move the passage embeddings to the CPU if not already done
        # the move to cpu and then to gpu is needed to avoid OOM when using mixed precision
        if not self.device == "cpu":  # this if is to avoid unnecessary moves
            passage_embeddings = [c.detach().cpu() for c in passage_embeddings]
        # stack it
        passage_embeddings: torch.Tensor = torch.stack(passage_embeddings, dim=0)
        # move the passage embeddings to the gpu if needed
        if not self.device == "cpu":
            passage_embeddings = passage_embeddings.to(PRECISION_MAP[self.precision])
            passage_embeddings = passage_embeddings.to(self.device)
        self.embeddings = passage_embeddings
        # update the matrix multiplication module
        # self.mm = MatrixMultiplicationModule(embeddings=self.embeddings)

        # free up memory from the unused variable
        del passage_embeddings

        return self

    @torch.no_grad()
    @torch.inference_mode()
    def search(self, query: torch.Tensor, k: int = 1) -> list[list[RetrievedSample]]:
        """
        Search the documents using the query.

        Args:
            query (:obj:`torch.Tensor`):
                The query to be used for searching.
            k (:obj:`int`, `optional`, defaults to 1):
                The number of documents to be retrieved.

        Returns:
            :obj:`List[RetrievedSample]`: The retrieved documents.
        """

        with get_autocast_context(self.device, self.embeddings.dtype):
            # move query to the same device as embeddings
            query = query.to(self.embeddings.device)
            if query.dtype != self.embeddings.dtype:
                query = query.to(self.embeddings.dtype)
            similarity = torch.matmul(query, self.embeddings.T)
            # similarity = self.mm(query)
            # Retrieve the indices of the top k passage embeddings
            retriever_out: torch.return_types.topk = torch.topk(
                similarity, k=min(k, similarity.shape[-1]), dim=1
            )

        # get int values
        batch_top_k: List[List[int]] = retriever_out.indices.detach().cpu().tolist()
        # get float values
        batch_scores: List[List[float]] = retriever_out.values.detach().cpu().tolist()
        # Retrieve the passages corresponding to the indices
        batch_docs = [
            [self.documents.get_document_from_id(i) for i in indices]
            for indices in batch_top_k
        ]
        # build the output object
        batch_retrieved_samples = [
            [
                RetrievedSample(document=doc, score=score)
                for doc, score in zip(docs, scores)
            ]
            for docs, scores in zip(batch_docs, batch_scores)
        ]
        return batch_retrieved_samples
