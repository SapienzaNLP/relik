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
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
):
    if push_to_hub:
        if not repo_id:
            raise ValueError("`repo_id` must be provided when `push_to_hub=True`")

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
    retriever.document_index.save_pretrained(
        output_folder, push_to_hub=push_to_hub, model_id=repo_id
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Create retriever index.")
    arg_parser.add_argument("--question-encoder-name-or-path", type=str, required=True)
    arg_parser.add_argument("--document-path", type=str, required=True)
    arg_parser.add_argument("--passage-encoder-name-or-path", type=str)
    arg_parser.add_argument(
        "--indexer_class",
        type=str,
        default="relik.retriever.indexers.inmemory.InMemoryDocumentIndex",
    )
    arg_parser.add_argument("--document-file-type", type=str, default="jsonl")
    arg_parser.add_argument("--output-folder", type=str, required=True)
    arg_parser.add_argument("--batch-size", type=int, default=128)
    arg_parser.add_argument("--passage-max-length", type=int, default=64)
    arg_parser.add_argument("--device", type=str, default="cuda")
    arg_parser.add_argument("--index-device", type=str, default="cpu")
    arg_parser.add_argument("--precision", type=str, default="fp32")
    arg_parser.add_argument("--num-workers", type=int, default=4)
    arg_parser.add_argument("--push-to-hub", action="store_true")
    arg_parser.add_argument("--repo-id", type=str)

    build_index(**vars(arg_parser.parse_args()))
