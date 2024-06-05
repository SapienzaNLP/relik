import argparse
import json
import os
from pathlib import Path
from typing import Union

from tqdm import tqdm
from relik.inference.data.splitters.base_sentence_splitter import BaseSentenceSplitter
from relik.inference.data.splitters.blank_sentence_splitter import BlankSentenceSplitter
from relik.inference.data.splitters.spacy_sentence_splitter import SpacySentenceSplitter
from relik.inference.data.splitters.window_based_splitter import WindowSentenceSplitter
from relik.inference.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from relik.inference.data.window.manager import WindowManager


def create_windows(
    input_file: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    window_size: int = 32,
    window_stride: int = 16,
    title_mapping: str = None,
    language: str = "en",
    tokenizer_device: str = "cpu",
    is_split_into_words: bool = False,
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

    data = []
    with open(input_file) as f:
        for line in f:
            data.append(json.loads(line))

    windowized_data_train = []
    windowized_data_dev = []
    windowized_data_test = []
    for document in tqdm(data, desc="Windowizing documents"):
        doc_info = document["doc_id"]

        # clean doc_info, e.g. "-DOCSTART- (1 EU)"
        doc_info = (
            doc_info.replace("-DOCSTART-", "").replace("(", "").replace(")", "").strip()
        )
        doc_id, doc_topic = doc_info.split(" ")

        if "testa" in doc_id:
            split = "dev"
        elif "testb" in doc_id:
            split = "test"
        else:
            split = "train"

        doc_id = doc_id.replace("testa", "").replace("testb", "").strip()
        doc_id = int(doc_id)

        windowized_document = window_manager.create_windows(
            document["doc_text"],
            window_size,
            window_stride,
            is_split_into_words=is_split_into_words,
            doc_ids=doc_id,
            doc_topic=doc_topic,
        )

        # we need to add the labels
        doc_level_labels = document["doc_annotations"]
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
        for window in windowized_document:
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

        if split == "train":
            windowized_data_train.extend(windowized_document)
        elif split == "dev":
            windowized_data_dev.extend(windowized_document)
        elif split == "test":
            windowized_data_test.extend(windowized_document)
        else:
            raise ValueError(f"Unknown split: {split}")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    with open(output_dir_path / "train_windowed.jsonl", "w") as f:
        for window in windowized_data_train:
            # f.write(json.dumps(window) + "\n")
            f.write(window.to_jsons() + "\n")
    with open(output_dir_path / "testa_windowed.jsonl", "w") as f:
        for window in windowized_data_dev:
            # f.write(json.dumps(window) + "\n")
            f.write(window.to_jsons() + "\n")
    with open(output_dir_path / "testb_windowed.jsonl", "w") as f:
        for window in windowized_data_test:
            # f.write(json.dumps(window) + "\n")
            f.write(window.to_jsons() + "\n")

    # print(f"Missing labels: {missing_labels}")
    # print(f"Total number of missing labels: {len(missing_labels)}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_file", type=str, required=True)
    arg_parser.add_argument("--output_dir", type=str, required=True)
    arg_parser.add_argument("--window_size", type=int, default=32)
    arg_parser.add_argument("--window_stride", type=int, default=16)
    arg_parser.add_argument("--title_mapping", type=str)
    arg_parser.add_argument("--language", type=str, default="en")
    arg_parser.add_argument("--tokenizer_device", type=str, default="cpu")
    arg_parser.add_argument("--is_split_into_words", action="store_true")

    create_windows(**vars(arg_parser.parse_args()))
