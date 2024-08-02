import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import hydra
import torch
import tqdm
import typer

from relik.cli.utils import resolve_config
from relik.common.log import get_logger, print_relik_text_art
from relik.common.utils import get_callable_from_string
from relik.retriever import GoldenRetriever
from relik.retriever.common.model_inputs import ModelInputs
from relik.retriever.data.base.datasets import BaseDataset
from relik.retriever.indexers.document import DocumentStore
from relik.retriever.trainer.train import train_hydra as retriever_train

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def train():
    """
    Trains the retriever model.

    This function prints the Relik text art, resolves the configuration file path,
    and then calls the `_retriever_train` function to train the retriever model.

    Args:
        None

    Returns:
        None
    """
    print_relik_text_art()
    config_dir, config_name, overrides = resolve_config("retriever")

    @hydra.main(
        config_path=str(config_dir),
        config_name=str(config_name),
        version_base="1.3",
    )
    def _retriever_train(conf):
        retriever_train(conf)

    # clean sys.argv for hydra
    sys.argv = sys.argv[:1]
    # add the overrides to sys.argv
    sys.argv.extend(overrides)

    _retriever_train()


@torch.no_grad()
@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def create_index(
    question_encoder_name_or_path: str,
    document_path: str,
    output_folder: str,
    document_file_type: str = "jsonl",
    passage_encoder_name_or_path: Optional[str] = None,
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
    """
    Builds an index for document retrieval.

    Args:
        question_encoder_name_or_path (str):
            The name or path of the question encoder model.
        document_path (str):
            The path to the document file.
        output_folder (str):
            The folder where the index will be saved.
        document_file_type (str, optional):
            The type of the document file. Defaults to "jsonl".
        passage_encoder_name_or_path (str, optional):
            The name or path of the passage encoder model. Defaults to None.
        indexer_class (str, optional):
            The class of the document indexer. Defaults to "relik.retriever.indexers.inmemory.InMemoryDocumentIndex".
        batch_size (int, optional):
            The batch size for indexing. Defaults to 512.
        num_workers (int, optional):
            The number of workers for indexing. Defaults to 4.
        passage_max_length (int, optional):
            The maximum length of a passage. Defaults to 64.
        device (str, optional):
            The device to use for indexing. Defaults to "cuda".
        index_device (str, optional):
            The device to use for indexing the document index. Defaults to "cpu".
        precision (str, optional):
            The precision for indexing. Defaults to "fp32".
        push_to_hub (bool, optional):
             Whether to push the index to the Hugging Face Model Hub. Defaults to False.
        repo_id (str, optional):
            The ID of the repository in the Hugging Face Model Hub. Required if push_to_hub is True.

    Raises:
        ValueError: If `repo_id` is not provided when `push_to_hub=True`.

    Returns:
        None
    """

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


@torch.no_grad()
@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def add_candidates(
    question_encoder_name_or_path: str,
    document_name_or_path: str,
    input_path: str,
    output_path: str,
    passage_encoder_name_or_path: Optional[str] = None,
    relations: bool = False,
    top_k: int = 100,
    batch_size: int = 128,
    num_workers: int = 4,
    device: str = "cuda",
    index_device: str = "cpu",
    precision: str = "fp32",
    use_doc_topics: bool = False,
    log_recall: bool = False,
):
    """
    Adds candidates to the input samples based on retrieval from a document index.

    Args:
        question_encoder_name_or_path (str):
            The name or path of the question encoder model.
        document_name_or_path (str):
            The name or path of the document index.
        input_path (str):
            The path to the input file containing samples.
        output_path (str):
            The path to the output file where the samples with candidates will be saved.
        passage_encoder_name_or_path (Optional[str]):
            The name or path of the passage encoder model. Defaults to None.
        relations (bool):
            Whether to add the candidates as relations. Defaults to False.
        top_k (int):
            The number of candidates to retrieve for each sample. Defaults to 100.
        batch_size (int):
            The batch size for retrieval. Defaults to 128.
        num_workers (int):
            The number of worker processes for data loading. Defaults to 4.
        device (str):
            The device to use for retrieval. Defaults to "cuda".
        index_device (str):
            The device to use for the document index. Defaults to "cpu".
        precision (str):
            The precision to use for retrieval. Defaults to "fp32".
        use_doc_topics (bool):
            Whether to use document topics for retrieval. Defaults to False.
        log_recall (bool):
            Whether to log the recall of the retrieval. Defaults to False.

    Raises:
        ValueError: If the dataset does not contain topics but `use_doc_topics` is set to True.

    Returns:
        None
    """
    retriever = GoldenRetriever(
        question_encoder=question_encoder_name_or_path,
        passage_encoder=passage_encoder_name_or_path,
        document_index=document_name_or_path,
        device=device,
        index_device=index_device,
        index_precision=precision,
    )
    retriever.eval()

    logger.info(f"Loading from {input_path}")
    with open(input_path) as f:
        samples = [json.loads(line) for line in f.readlines()]

    if use_doc_topics and "doc_topic" not in samples[0]:
        raise ValueError(
            "Dataset does not contain topics, but --use-doc-topics was passed"
        )
    use_doc_topics = use_doc_topics and "doc_topic" in samples[0]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct, total = 0, 0

    with open(output_path, "w") as f_out:
        # get tokenizer
        tokenizer = retriever.question_tokenizer

        def collate_fn(batch):
            return ModelInputs(
                tokenizer(
                    [b["text"] for b in batch],
                    text_pair=(
                        [b["doc_topic"] for b in batch] if use_doc_topics else None
                    ),
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                )
            )

        logger.info(
            f"Creating dataloader with batch size {batch_size} and {num_workers} workers"
        )
        dataloader = torch.utils.data.DataLoader(
            BaseDataset(name="passage", data=samples),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        # we also dump the candidates to a file after a while
        retrieved_accumulator = []
        with torch.inference_mode():
            num_completed_docs = 0

            start = time.time()
            for documents_batch in tqdm.tqdm(dataloader):
                retrieve_kwargs = {
                    **documents_batch,
                    "k": top_k,
                    "precision": precision,
                }
                batch_out = retriever.retrieve(**retrieve_kwargs)
                retrieved_accumulator.extend(batch_out)

                if len(retrieved_accumulator) % 1_000 == 0:
                    output_data = []
                    # get the correct document from the original dataset
                    # the dataloader is not shuffled, so we can just count the number of
                    # documents we have seen so far
                    for sample, retrieved in zip(
                        samples[
                            num_completed_docs : num_completed_docs
                            + len(retrieved_accumulator)
                        ],
                        retrieved_accumulator,
                    ):
                        candidate_titles = [
                            c.document.text
                            for c in retrieved  # TODO: add metadata if needed
                        ]
                        # TODO: compatibility shit
                        if relations:
                            sample["triplet_candidates"] = candidate_titles
                            sample["triplet_candidates_scores"] = [
                                c.score for c in retrieved
                            ]
                            if log_recall:
                                for triplet in sample["window_triplet_labels"]:
                                    relation = triplet["relation"]
                                    if relation.lower() in candidate_titles:
                                        correct += 1
                                    else:
                                        logger.debug(
                                            f"Did not find `{relation.lower()}` in candidates"
                                        )
                                    total += 1
                        else:
                            sample["span_candidates"] = candidate_titles
                            # sample["window_candidates"] = candidate_titles
                            sample["span_candidates_scores"] = [
                                c.score for c in retrieved
                            ]
                            if log_recall:
                                candidate_titles_lower = [
                                    candidate.lower() for candidate in candidate_titles
                                ]
                                for ss, se, label in sample["window_labels"]:
                                    if label == "--NME--":
                                        continue
                                    if (
                                        label.replace("_", " ").lower()
                                        in candidate_titles_lower
                                    ):
                                        correct += 1
                                    else:
                                        logger.debug(
                                            f"Did not find `{label.replace('_', ' ').lower()}` in candidates"
                                        )
                                    total += 1
                        output_data.append(sample)

                    for sample in output_data:
                        f_out.write(json.dumps(sample) + "\n")

                    num_completed_docs += len(retrieved_accumulator)
                    retrieved_accumulator = []

            if len(retrieved_accumulator) > 0:
                output_data = []
                # get the correct document from the original dataset
                # the dataloader is not shuffled, so we can just count the number of
                # documents we have seen so far
                for sample, retrieved in zip(
                    samples[
                        num_completed_docs : num_completed_docs
                        + len(retrieved_accumulator)
                    ],
                    retrieved_accumulator,
                ):
                    candidate_titles = [
                        c.document.text
                        for c in retrieved  # TODO: add metadata if needed
                    ]
                    # TODO: compatibility shit
                    if relations:
                        sample["triplet_candidates"] = candidate_titles
                        sample["triplet_candidates_scores"] = [
                            c.score for c in retrieved
                        ]
                        if log_recall:
                            for triplet in sample["window_triplet_labels"]:
                                relation = triplet["relation"]
                                if relation.lower() in candidate_titles:
                                    correct += 1
                                else:
                                    logger.debug(
                                        f"Did not find `{relation.lower()}` in candidates"
                                    )
                                total += 1
                    else:
                        sample["span_candidates"] = candidate_titles
                        # sample["window_candidates"] = candidate_titles
                        sample["span_candidates_scores"] = [c.score for c in retrieved]
                        if log_recall:
                            for ss, se, label in sample["window_labels"]:
                                if label == "--NME--":
                                    continue
                                if label.replace("_", " ").lower() in candidate_titles:
                                    correct += 1
                                else:
                                    logger.debug(
                                        f"Did not find `{label.replace('_', ' ').lower()}` in candidates"
                                    )
                                total += 1
                    output_data.append(sample)

                for sample in output_data:
                    f_out.write(json.dumps(sample) + "\n")

                num_completed_docs += len(retrieved_accumulator)
                retrieved_accumulator = []

            end = time.time()
            logger.info(f"Retrieval took {end - start} seconds")
            if log_recall:
                recall = correct / total
                logger.info(f"Recall@{top_k}: {recall}")
