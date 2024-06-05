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

    windowized_data = []

    for document in tqdm(data, desc="Windowizing documents"):
        doc_id = document["doc_id"] if "doc_id" in document else document["id"]

        windowized_document = window_manager.create_windows(
            document["words"],
            window_size,
            window_stride,
            is_split_into_words=is_split_into_words,
            doc_id=doc_id,
        )

        # we need to add the labels
        doc_level_labels = document["entities"]
        doc_level_labels_triplets = document["triplets"]
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
                if doc_level_label[0] >= window.char2token_start[str(window.offset)] and doc_level_label[0] <= window.char2token_start[str(window.offset)] + len(
                    window.words
                ):
                    window_level_labels.append(doc_level_label)
            window._d["entities"] = window_level_labels
            del window._d["doc_topic"]
            del window._d["spans"]
            # now we need to map the labels to the chars
            window_level_labels_but_for_chars = []
            for label in window_level_labels:
                while str(label[0]) not in window.token2char_start and label[0] > window.char2token_start[str(window.offset)]:
                    label[0] -= 1
                while str(label[1]-1) not in window.token2char_end and label[1]-1 < window.char2token_start[str(window.offset)] + len(
                    window.words
                ):
                    label[1] += 1
                start_char = window.token2char_start[str(label[0])] - window.offset if str(label[0]) in window.token2char_start else None
                end_char = window.token2char_end[str(label[1]-1)] - window.offset if str(label[1]-1) in window.token2char_end else None
                if start_char is None or end_char is None:
                    raise ValueError(
                        f"Could not find token for label: {label} in window: {window}"
                    )
                window_level_labels_but_for_chars.append(
                    [start_char, end_char] + label[2:]
                )

            window._d["entities_chars"] = window_level_labels_but_for_chars
            window_level_labels_triplets = []
            for triplet in doc_level_labels_triplets:
                if triplet["subject"]["start"] >= window.char2token_start[str(window.offset)] and triplet["subject"]["start"] <= window.char2token_start[str(window.offset)] + len(
                    window.words
                ) and triplet["object"]["start"] >= window.char2token_start[str(window.offset)] and triplet["object"]["start"] <= window.char2token_start[str(window.offset)] + len(
                    window.words
                ):
                    window_level_labels_triplets.append(triplet)
            window._d["triplets"] = window_level_labels_triplets
            window_level_labels_triplets_but_for_chars = []
            for triplet in window_level_labels_triplets:
                subject_start_char = window.token2char_start[str(triplet["subject"]["start"])] - window.offset if str(triplet["subject"]["start"]) in window.token2char_start else None
                subject_end_char = window.token2char_end[str(triplet["subject"]["end"]-1)] - window.offset if str(triplet["subject"]["end"]-1) in window.token2char_end else None
                object_start_char = window.token2char_start[str(triplet["object"]["start"])] - window.offset if str(triplet["object"]["start"]) in window.token2char_start else None
                object_end_char = window.token2char_end[str(triplet["object"]["end"]-1)] - window.offset if str(triplet["object"]["end"]-1) in window.token2char_end else None
                if subject_start_char is None or subject_end_char is None or object_start_char is None or object_end_char is None:
                    raise ValueError(
                        f"Could not find token for triplet: {triplet} in window: {window}"
                    )
                window_level_labels_triplets_but_for_chars.append(
                    {
                        "subject": {
                            "start": subject_start_char,
                            "end": subject_end_char,
                            "name": triplet["subject"]["name"],
                            "type": triplet["subject"]["type"],
                            "ner_type": triplet["subject"]["ner_type"] if "ner_type" in triplet["subject"] else None,
                            "uri": triplet["subject"]["uri"] if "uri" in triplet["subject"] else None,
                        },
                        "relation": triplet["relation"],
                        "object": {
                            "start": object_start_char,
                            "end": object_end_char,
                            "name": triplet["object"]["name"],
                            "type": triplet["object"]["type"],
                            "ner_type": triplet["object"]["ner_type"] if "ner_type" in triplet["object"] else None,
                            "uri": triplet["object"]["uri"] if "uri" in triplet["object"] else None,
                        },
                    }
                )
            window._d["triplets_chars"] = window_level_labels_triplets_but_for_chars
            if "triplet_candidates" in document:
                window._d["triplet_candidates"] = document["triplet_candidates"]
            if "triplet_candidates_scores" in document:
                window._d["triplet_candidates_scores"] = document["triplet_candidates_scores"]
        windowized_data.extend(windowized_document)


    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    with open(output_dir_path / input_file.split("/")[-1].replace(".jsonl", "_windowed.jsonl"), "w") as f:
        for window in windowized_data:
            # f.write(json.dumps(window) + "\n")
            f.write(window.to_jsons() + "\n")


    # print(f"Missing labels: {missing_labels}")
    # print(f"Total number of missing labels: {len(missing_labels)}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_file", type=str, required=True)
    arg_parser.add_argument("--output_dir", type=str, required=True)
    arg_parser.add_argument("--window_size", type=str, default=32)
    arg_parser.add_argument("--window_stride", type=int, default=16)
    arg_parser.add_argument("--title_mapping", type=str)
    arg_parser.add_argument("--language", type=str, default="en")
    arg_parser.add_argument("--tokenizer_device", type=str, default="cpu")
    arg_parser.add_argument("--is_split_into_words", action="store_true")

    create_windows(**vars(arg_parser.parse_args()))
