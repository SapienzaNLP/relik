import argparse
import json
import logging
import os
from pathlib import Path
import time
from typing import Optional, Union

import torch
import tqdm

from relik.retriever import GoldenRetriever
from relik.common.log import get_logger
from relik.retriever.common.model_inputs import ModelInputs
from relik.retriever.data.base.datasets import BaseDataset

logger = get_logger(level=logging.INFO)


def compute_retriever_stats(dataset, top_k) -> None:
    correct, total = 0, 0
    for sample in dataset:
        window_candidates = sample["span_candidates"]
        window_candidates = [c.replace("_", " ").lower() for c in window_candidates]

        for ss, se, label in sample["window_labels"]:
            if label == "--NME--":
                continue
            if label.replace("_", " ").lower() in window_candidates:
                correct += 1
            else:
                logger.debug(f"Did not find `{label.replace('_', ' ').lower()}` in candidates")
            total += 1

    recall = correct / total
    logger.info(f"Recall@{top_k}: {recall}")


@torch.no_grad()
def add_candidates(
    question_encoder_name_or_path: Union[str, os.PathLike],
    document_name_or_path: Union[str, os.PathLike],
    input_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    passage_encoder_name_or_path: Optional[Union[str, os.PathLike]] = None,
    top_k: int = 100,
    batch_size: int = 128,
    num_workers: int = 4,
    device: str = "cuda",
    index_device: str = "cpu",
    precision: str = "fp32",
    use_doc_topics: bool = False,
    log_recall: bool = True,
):
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
        raise ValueError("Dataset does not contain topics, but --use-doc-topics was passed")
    use_doc_topics = use_doc_topics and "doc_topic" in samples[0]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f_out:
        # get tokenizer
        tokenizer = retriever.question_tokenizer

        def collate_fn(batch):
            return ModelInputs(
                tokenizer(
                    [b["text"] for b in batch],
                    text_pair=[b["doc_topic"] for b in batch] if use_doc_topics else None,
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                )
            )

        logger.info(f"Creating dataloader with batch size {batch_size}")
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

                if len(retrieved_accumulator) % 300_000 == 0:
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
                            c.document.text for c in retrieved # TODO: add metadata if needed
                        ]
                        # TODO: compatibility shit
                        sample["span_candidates"] = candidate_titles
                        sample["window_candidates"] = candidate_titles
                        sample["window_candidates_scores"] = [
                            c.score for c in retrieved
                        ]
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
                        c.document.text for c in retrieved # TODO: add metadata if needed
                    ]
                    # TODO: compatibility shit
                    sample["span_candidates"] = candidate_titles
                    sample["window_candidates"] = candidate_titles
                    sample["window_candidates_scores"] = [c.score for c in retrieved]
                    output_data.append(sample)

                for sample in output_data:
                    f_out.write(json.dumps(sample) + "\n")

                num_completed_docs += len(retrieved_accumulator)
                retrieved_accumulator = []

            end = time.time()
            logger.info(f"Retrieval took {end - start} seconds")

    if log_recall:
        with open(output_path) as f:
            annotated_samples = [json.loads(line) for line in f.readlines()]
        compute_retriever_stats(annotated_samples, top_k)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Add the candidates to the windowized dataset")
    arg_parser.add_argument("--question-encoder-name-or-path", type=str, required=True)
    arg_parser.add_argument("--document-name-or-path", type=str, required=True)
    arg_parser.add_argument("--passage-encoder-name-or-path", type=str)
    arg_parser.add_argument("--input-path", type=str, required=True)
    arg_parser.add_argument("--output-path", type=str, required=True)
    arg_parser.add_argument("--top-k", type=int, default=100)
    arg_parser.add_argument("--batch-size", type=int, default=128)
    arg_parser.add_argument("--device", type=str, default="cuda")
    arg_parser.add_argument("--index-device", type=str, default="cpu")
    arg_parser.add_argument("--precision", type=str, default="fp32")
    arg_parser.add_argument("--use-doc-topics", action="store_true")
    arg_parser.add_argument("--num-workers", type=int, default=4)
    arg_parser.add_argument("--log-recall", action="store_true")

    add_candidates(**vars(arg_parser.parse_args()))
