import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Union

from tqdm import tqdm

from relik.common.log import get_logger
from relik.inference.data.splitters.base_sentence_splitter import BaseSentenceSplitter
from relik.inference.data.splitters.blank_sentence_splitter import BlankSentenceSplitter
from relik.inference.data.splitters.spacy_sentence_splitter import SpacySentenceSplitter
from relik.inference.data.splitters.window_based_splitter import WindowSentenceSplitter
from relik.inference.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from relik.inference.data.window.manager import WindowManager

logger = get_logger()


def create_windows(
    input_file: Union[str, os.PathLike],
    output_file: Union[str, os.PathLike],
    window_size: int = 32,
    window_stride: int = 16,
    title_mapping: str = None,
    language: str = "en",
    tokenizer_device: str = "cpu",
    is_split_into_words: bool = False,
    write_batch_size: int = 10_000,
):
    # windowization stuff
    tokenizer = SpacyTokenizer(language=language, use_gpu=tokenizer_device == "cuda")
    if window_size == "none":
        sentence_splitter = BlankSentenceSplitter()
    elif window_size == "sentence":
        sentence_splitter = SpacySentenceSplitter()
    else:
        sentence_splitter = WindowSentenceSplitter(
            window_size=window_size, window_stride=window_stride
        )
    window_manager = WindowManager(tokenizer, sentence_splitter)

    if title_mapping is not None:
        with open(title_mapping) as f:
            title_mapping = json.load(f)

    output_file_path = Path(output_file)

    # check if file exists
    continue_from_id = None
    if output_file_path.exists():
        # we should not overwrite the file
        # open last line of the file using tail command
        try:
            last_line = subprocess.check_output(f"tail -n 1 {output_file}", shell=True)
            continue_from_id = json.loads(last_line)["doc_id"]
        except Exception as e:
            logger.error(f"Error getting last line of the file: {e}")
        logger.info(
            f"Output file {output_file} already exists. Continuing from doc id {continue_from_id}"
        )
    else:
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving windowized data to {output_file}")

    logger.info(f"Loading data from {input_file}")
    batched_data = []
    # get number of lines in the file
    # run bash command to get the number of lines in the file
    try:
        total_lines = int(
            subprocess.check_output(
                f"wc -l {input_file} | awk '{{print $1}}'", shell=True
            )
        )
    except Exception as e:
        logger.error(f"Error getting number of lines in the file: {e}")
        total_lines = None
    progress_bar = tqdm(total=total_lines)
    write_mode = "a" if continue_from_id is not None else "w"
    with open(input_file) as f_in, open(output_file, write_mode) as f_out:
        for line in f_in:
            if continue_from_id is not None:
                # we need to skip until we reach the last written line
                current_id = json.loads(line)["doc_id"]
                if current_id != continue_from_id:
                    progress_bar.update(1)
                    continue
                else:
                    continue_from_id = None
            batched_data.append(json.loads(line))
            if len(batched_data) == write_batch_size:
                windowized_data = _process_batch(
                    batched_data,
                    window_manager,
                    window_size,
                    window_stride,
                    is_split_into_words,
                    title_mapping,
                )
                for wd in windowized_data:
                    f_out.write(wd.to_jsons() + "\n")
                progress_bar.update(len(batched_data))
                batched_data = []
            
        if len(batched_data) > 0:
            windowized_data = _process_batch(
                batched_data,
                window_manager,
                window_size,
                window_stride,
                is_split_into_words,
                title_mapping,
            )
            for wd in windowized_data:
                f_out.write(wd.to_jsons() + "\n")
            progress_bar.update(len(batched_data))
            batched_data = []


def _process_batch(
    data, window_manager, window_size, window_stride, is_split_into_words, title_mapping
):
    # build a doc_id to doc mapping
    doc_id_to_doc = {int(document["doc_id"]): document for document in data}

    windowized_data = window_manager.create_windows(
        [document["doc_text"] for document in data],
        window_size,
        window_stride,
        is_split_into_words=is_split_into_words,
        doc_ids=[int(document["doc_id"]) for document in data],
        # doc_topic=doc_topic,
    )

    for window in windowized_data:
        try:
            # we need to add the labels
            doc_level_labels = doc_id_to_doc[window._d["doc_id"]]["doc_annotations"]
            # if we have a title mapping, we need to map the labels to the
            # new titles
            if title_mapping is not None:
                # compute the missing labels
                # missing_labels |= set(title_mapping.keys()) - set(
                #     [label for _, _, label in doc_level_labels]
                # )
                doc_level_labels = [
                    [start, end, title_mapping.get(label, label)]
                    for start, end, label in doc_level_labels
                ]

            # these are the labels for the whole document, we need add them to the correct window
            # for window in windowized_document:
            window_level_labels = []
            for doc_level_label in doc_level_labels:
                start_char, end_char, label_text = doc_level_label
                if start_char >= window.offset and end_char <= window.offset + len(
                    window.text
                ):
                    window_level_labels.append(doc_level_label)
            window._d["window_labels"] = window_level_labels

            # now we need to map the labels to the tokens
            window_level_labels_but_for_tokens = []
            for label in window_level_labels:
                start_char, end_char, label_text = label
                start_token = None
                end_token = None
                for token_id, (start, end) in enumerate(
                    zip(
                        window._d["token2char_start"].values(),
                        window._d["token2char_end"].values(),
                    )
                ):
                    if start_char == start:
                        start_token = token_id
                    if end_char == end:
                        end_token = token_id + 1
                if start_token is None or end_token is None:
                    raise ValueError(
                        f"Could not find token for label: {label} in window: {window}"
                    )
                window_level_labels_but_for_tokens.append(
                    [start_token, end_token, label_text]
                )
            window._d["window_labels_tokens"] = window_level_labels_but_for_tokens

        except Exception as e:
            logger.error(
                f"Error processing document {window._d['doc_id']} window {window._d['window_id']}: {e}"
            )

    return windowized_data


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Create windows from ReLiK data.")
    arg_parser.add_argument("input_file", type=str)
    arg_parser.add_argument("output_file", type=str)
    arg_parser.add_argument("--window-size", type=int, default=32)
    arg_parser.add_argument("--window-stride", type=int, default=16)
    arg_parser.add_argument("--title-mapping", type=str)
    arg_parser.add_argument("--language", type=str, default="en")
    arg_parser.add_argument("--tokenizer-device", type=str, default="cpu")
    arg_parser.add_argument("--is-split-into-words", action="store_true")
    arg_parser.add_argument("--write-batch-size", type=int, default=10_000)

    create_windows(**vars(arg_parser.parse_args()))
