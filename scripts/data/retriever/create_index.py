import argparse
import csv
import os
from pathlib import Path
from typing import Optional, Union

import torch

from relik.retriever import GoldenRetriever
from relik.common.utils import get_logger, get_callable_from_string
from relik.retriever.indexers.document import DocumentStore

logger = get_logger(__name__)


@torch.no_grad()
def build_index(
    question_encoder_name_or_path: Union[str, os.PathLike],
    document_path: Union[str, os.PathLike],
    output_folder: Union[str, os.PathLike],
    document_file_type: str = "jsonl",
    passage_encoder_name_or_path: Optional[Union[str, os.PathLike]] = None,
    indexer_class: str = "relik.retriever.indexers.inmemory.InMemoryDocumentIndex",
    batch_size: int = 512,
    num_workers: int = 4,
    passage_max_length: int = 64,
    device: str = "cuda",
    index_device: str = "cpu",
    precision: str = "fp32",
):
    logger.info("Loading documents")
    if document_file_type == "jsonl":
        documents = DocumentStore.from_file(document_path)
    elif document_file_type == "csv":
        documents = DocumentStore.from_tsv(
            document_path, delimiter=",", quoting=csv.QUOTE_NONE, ingore_case=True
        )
    elif document_file_type == "tsv":
        documents = DocumentStore.from_tsv(
            document_path, delimiter="\t", quoting=csv.QUOTE_NONE, ingore_case=True
        )
    else:
        raise ValueError(
            f"Unknown document file type: {document_file_type}, must be one of jsonl, csv, tsv"
        )

    logger.info("Loading document index")
    # document_index = InMemoryDocumentIndex(
    #     documents=documents,
    #     # metadata_fields=["title"],
    #     # separator=" <title> ",
    #     device="cuda",
    #     precision="16",
    # )
    # retriever.document_index = document_index
    logger.info(f"Loaded {len(documents)} documents")
    indexer = get_callable_from_string(indexer_class)(
        documents, device=index_device, precision=precision
    )

    retriever = GoldenRetriever(
        question_encoder=question_encoder_name_or_path,
        passage_encoder=passage_encoder_name_or_path,
        document_index=indexer,
        device=device,
        precision=precision,
    )
    retriever.eval()

    retriever.index(
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=passage_max_length,
        force_reindex=True,
        precision=precision,
    )

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    retriever.save_pretrained(output_folder)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--question_encoder_name_or_path", type=str, required=True)
    arg_parser.add_argument("--document_path", type=str, required=True)
    arg_parser.add_argument("--passage_encoder_name_or_path", type=str)
    arg_parser.add_argument(
        "--indexer_class",
        type=str,
        default="relik.retriever.indexers.inmemory.InMemoryDocumentIndex",
    )
    arg_parser.add_argument("--document_file_type", type=str, default="jsonl")
    arg_parser.add_argument("--output_folder", type=str, required=True)
    arg_parser.add_argument("--batch_size", type=int, default=128)
    arg_parser.add_argument("--passage_max_length", type=int, default=64)
    arg_parser.add_argument("--device", type=str, default="cuda")
    arg_parser.add_argument("--index_device", type=str, default="cpu")
    arg_parser.add_argument("--precision", type=str, default="fp32")

    build_index(**vars(arg_parser.parse_args()))
