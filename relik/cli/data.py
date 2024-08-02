import ast
import csv
import json
import os
import subprocess
from pathlib import Path

import typer
from tqdm import tqdm

from relik.common.log import get_logger
from relik.inference.data.splitters.blank_sentence_splitter import BlankSentenceSplitter
from relik.inference.data.splitters.spacy_sentence_splitter import SpacySentenceSplitter
from relik.inference.data.splitters.window_based_splitter import WindowSentenceSplitter
from relik.inference.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from relik.inference.data.window.manager import WindowManager
from relik.retriever.indexers.document import DocumentStore

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def create_windows(
    input_file: str,
    output_file: str,
    window_size: int = 32,
    window_stride: int = 16,
    title_mapping: str = None,
    relation_mapping: str = None,
    language: str = "en",
    tokenizer_device: str = "cpu",
    is_split_into_words: bool = False,
    labels_are_tokens: bool = False,
    write_batch_size: int = 10_000,
    overwrite: bool = False,
):
    """
    Create windows from input documents and save them to an output file.

    Args:
        input_file (str):
            Path to the input file containing the documents.
        output_file (str):
            Path to the output file to save the windowized data.
        window_size (int, optional):
            Size of the window. Defaults to 32.
        window_stride (int, optional):
            Stride of the window. Defaults to 16.
        title_mapping (str, optional):
            Path to a JSON file containing a mapping of labels. Defaults to None.
        language (str, optional):
            Language of the documents. Defaults to "en".
        tokenizer_device (str, optional):
            Device to use for tokenization. Defaults to "cpu".
        is_split_into_words (bool, optional):
            Whether the documents are already split into words. Defaults to False.
        write_batch_size (int, optional):
            Number of windows to process and write at a time. Defaults to 10_000.

    Returns:
        None
    """

    def _process_batch(
        data,
        window_manager,
        window_size,
        window_stride,
        is_split_into_words,
        title_mapping,
        relation_mapping,
        labels_are_tokens,
    ):
        # build a doc_id to doc mapping
        doc_id_to_doc = {int(document["doc_id"]): document for document in data}
        if is_split_into_words:
            text_field = "doc_words"
        else:
            text_field = "doc_text"

        windowized_data, _, _ = window_manager.create_windows(
            [document[text_field] for document in data],
            window_size,
            window_stride,
            is_split_into_words=is_split_into_words,
            doc_ids=[int(document["doc_id"]) for document in data],
            # doc_topic=doc_topic,
        )

        for window in windowized_data:
            # try:
            # we need to add the labels
            doc_level_labels = doc_id_to_doc[window._d["doc_id"]][
                "doc_span_annotations"
            ]
            doc_text = doc_id_to_doc[window._d["doc_id"]][text_field]
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

            if relation_mapping is not None:
                doc_level_labels = [
                    [start, end, relation_mapping.get(label, label)]
                    for start, end, label in doc_level_labels
                ]
            # these are the labels for the whole document, we need add them to the correct window
            # for window in windowized_document:
            window_level_labels = []
            window_level_labels_but_for_tokens = []
            if not labels_are_tokens:
                for doc_level_label in doc_level_labels:
                    start_char, end_char, label_text = doc_level_label
                    while (
                        len(doc_text) > end_char - 1
                        and end_char - 1 >= 0
                        and doc_text[end_char - 1] == " "
                    ):
                        end_char -= 1
                    while (
                        len(doc_text) > start_char
                        and start_char >= 0
                        and doc_text[start_char] == " "
                    ):
                        start_char += 1
                    if start_char >= window.offset and end_char <= window.offset + len(
                        window.text
                    ):
                        if (
                            start_char > end_char
                            or start_char < window.offset
                            or end_char > window.offset + len(window.text)
                        ):
                            print(
                                f"Error in window {window._d['window_id']}, start: {start_char}, end: {end_char}, window start: {window.offset}, window end: {window.offset + len(window.text)}"
                            )
                            continue
                        window_level_labels.append([start_char, end_char, label_text])
                window._d["window_labels"] = window_level_labels
                # now we need to map the labels to the tokens
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
                        # if we found the tokens, we can break
                        if start_token is not None and end_token is not None:
                            break
                    if start_token is None or end_token is None:
                        # raise ValueError(
                        #     f"Could not find token for label: {label} in window: {window}"
                        # )
                        # warnings.warn(
                        #     f"Could not find token for label: {label} in window: {window}. We will snap the label to the word boundaries."
                        # )
                        start_token = None
                        end_token = None
                        for token_id, (start, end) in enumerate(
                            zip(
                                window._d["token2char_start"].values(),
                                window._d["token2char_end"].values(),
                            )
                        ):
                            if start_char >= start and start_char <= end:
                                start_token = token_id
                            if end_char >= start and end_char <= end:
                                end_token = token_id + 1
                            if start_token is not None and end_token is not None:
                                break
                    if start_token is None or end_token is None:
                        print(
                            f"Error in window {window._d['window_id']}, start: {start_char}, end: {end_char}, window start: {window.offset}, window end: {window.offset + len(window.text)}"
                        )
                        continue
                    window_level_labels_but_for_tokens.append(
                        [start_token, end_token, label_text]
                    )
                window._d["window_labels_tokens"] = window_level_labels_but_for_tokens
            else:
                # if the text is split into words, we need to map the labels to the tokens
                for doc_level_label in doc_level_labels:
                    start_token, end_token, label_text = doc_level_label
                    window_token_start = window._d["char2token_start"][
                        str(window.offset)
                    ]
                    window_token_end = window._d["char2token_end"][
                        str(window.offset + len(window.text))
                    ]
                    if (
                        start_token >= window_token_start
                        and end_token <= window_token_end
                    ):
                        window_level_labels_but_for_tokens.append(doc_level_label)
                window._d["window_labels_tokens"] = window_level_labels_but_for_tokens
                # if the text is split into words, we need to map the labels to the characters
                for label in window_level_labels_but_for_tokens:
                    start_token, end_token, label_text = label
                    start_char = window._d["token2char_start"][str(start_token)]
                    end_char = window._d["token2char_end"][str(end_token - 1)]
                    window_level_labels.append([start_char, end_char, label_text])
                window._d["window_labels"] = window_level_labels

            if "doc_triplet_annotations" in doc_id_to_doc[window._d["doc_id"]]:
                doc_level_triplet_labels = doc_id_to_doc[window._d["doc_id"]][
                    "doc_triplet_annotations"
                ]
                if title_mapping is not None:
                    doc_level_triplet_labels = [
                        {
                            "subject": [
                                triplet["subject"][0],
                                triplet["subject"][1],
                                title_mapping.get(
                                    triplet["subject"][2], triplet["subject"][2]
                                ),
                            ],
                            "relation": title_mapping.get(
                                triplet["relation"], triplet["relation"]
                            ),
                            "object": [
                                triplet["object"][0],
                                triplet["object"][1],
                                title_mapping.get(
                                    triplet["object"][2], triplet["object"][2]
                                ),
                            ],
                        }
                        for triplet in doc_level_triplet_labels
                    ]

                window_level_triplet_labels = []
                window_level_triplet_labels_but_for_tokens = []
                if not labels_are_tokens:
                    for doc_level_triplet_label in doc_level_triplet_labels:
                        subject_start_char, subject_end_char, subject_label_text = (
                            doc_level_triplet_label["subject"]
                        )
                        while (
                            doc_text[subject_end_char - 1] == " "
                        ):  # remove trailing spaces
                            subject_end_char -= 1
                        while doc_text[subject_start_char] == " ":
                            subject_start_char += 1
                        relation_label_text = doc_level_triplet_label["relation"]
                        object_start_char, object_end_char, object_label_text = (
                            doc_level_triplet_label["object"]
                        )
                        while (
                            doc_text[object_end_char - 1] == " "
                        ):  # remove trailing spaces
                            object_end_char -= 1
                        while doc_text[object_start_char] == " ":
                            object_start_char += 1
                        if (
                            subject_start_char >= window.offset
                            and subject_end_char <= window.offset + len(window.text)
                            and object_start_char >= window.offset
                            and object_end_char <= window.offset + len(window.text)
                        ):
                            window_level_triplet_labels.append(
                                {
                                    "subject": [
                                        subject_start_char,
                                        subject_end_char,
                                        subject_label_text,
                                    ],
                                    "relation": relation_label_text,
                                    "object": [
                                        object_start_char,
                                        object_end_char,
                                        object_label_text,
                                    ],
                                }
                            )
                    window._d["window_triplet_labels"] = window_level_triplet_labels
                    # now we need to map the labels to the tokens
                    for triplet in window_level_triplet_labels:
                        subject_start_char, subject_end_char, subject_label_text = (
                            triplet["subject"]
                        )
                        relation_label_text = triplet["relation"]
                        object_start_char, object_end_char, object_label_text = triplet[
                            "object"
                        ]
                        subject_start_token = None
                        subject_end_token = None
                        object_start_token = None
                        object_end_token = None
                        for token_id, (start, end) in enumerate(
                            zip(
                                window._d["token2char_start"].values(),
                                window._d["token2char_end"].values(),
                            )
                        ):
                            if subject_start_char == start:
                                subject_start_token = token_id
                            if subject_end_char == end:
                                subject_end_token = token_id + 1
                            if object_start_char == start:
                                object_start_token = token_id
                            if object_end_char == end:
                                object_end_token = token_id + 1
                        if (
                            subject_start_token is None
                            or subject_end_token is None
                            or object_start_token is None
                            or object_end_token is None
                        ):
                            # raise ValueError(
                            #     f"Could not find token for triplet: {triplet} in window: {window}"
                            # )
                            # find entity in window_labels using character offsets, use same index to find it in window_labels_tokens
                            if subject_start_token is None or subject_end_token is None:
                                subject_index = None
                                for idx, label in enumerate(window._d["window_labels"]):
                                    if (
                                        label[0] == subject_start_char
                                        and label[1] == subject_end_char
                                    ):
                                        subject_index = idx
                                        break
                                if subject_index is None:
                                    print(
                                        f"Error in window {window._d['window_id']}, start: {subject_start_char}, end: {subject_end_char}, window start: {window.offset}, window end: {window.offset + len(window.text)}"
                                    )
                                    continue
                                subject_start_token, subject_end_token, _ = window._d[
                                    "window_labels_tokens"
                                ][subject_index]
                            if object_start_token is None or object_end_token is None:
                                object_index = None
                                for idx, label in enumerate(window._d["window_labels"]):
                                    if (
                                        label[0] == object_start_char
                                        and label[1] == object_end_char
                                    ):
                                        object_index = idx
                                        break
                                if object_index is None:
                                    print(
                                        f"Error in window {window._d['window_id']}, start: {object_start_char}, end: {object_end_char}, window start: {window.offset}, window end: {window.offset + len(window.text)}"
                                    )
                                    continue
                                object_start_token, object_end_token, _ = window._d[
                                    "window_labels_tokens"
                                ][object_index]

                        window_level_triplet_labels_but_for_tokens.append(
                            {
                                "subject": [
                                    subject_start_token,
                                    subject_end_token,
                                    subject_label_text,
                                ],
                                "relation": relation_label_text,
                                "object": [
                                    object_start_token,
                                    object_end_token,
                                    object_label_text,
                                ],
                            }
                        )
                    window._d["window_triplet_labels_tokens"] = (
                        window_level_triplet_labels_but_for_tokens
                    )
                else:
                    # if the text is split into words, we need to map the labels to the tokens
                    for doc_level_triplet_label in doc_level_triplet_labels:
                        subject_start_token, subject_end_token, subject_label_text = (
                            doc_level_triplet_label["subject"]
                        )
                        relation_label_text = doc_level_triplet_label["relation"]
                        object_start_token, object_end_token, object_label_text = (
                            doc_level_triplet_label["object"]
                        )
                        window_token_start = window._d["char2token_start"][
                            str(window.offset)
                        ]
                        window_token_end = window._d["char2token_end"][
                            str(window.offset + len(window.text))
                        ]
                        if (
                            subject_start_token >= window_token_start
                            and subject_end_token <= window_token_end
                            and object_end_token >= window_token_start
                            and object_end_token <= window_token_end
                        ):
                            window_level_triplet_labels_but_for_tokens.append(
                                doc_level_triplet_label
                            )
                    window._d["window_triplet_labels_tokens"] = (
                        window_level_triplet_labels_but_for_tokens
                    )
                    # if the text is split into words, we need to map the labels to the characters
                    for triplet in window_level_triplet_labels_but_for_tokens:
                        subject_start_token, subject_end_token, subject_label_text = (
                            triplet["subject"]
                        )
                        relation_label_text = triplet["relation"]
                        object_start_token, object_end_token, object_label_text = (
                            triplet["object"]
                        )
                        subject_start_char = window._d["token2char_start"][
                            str(subject_start_token)
                        ]
                        subject_end_char = window._d["token2char_end"][
                            str(subject_end_token - 1)
                        ]
                        object_start_char = window._d["token2char_start"][
                            str(object_start_token)
                        ]
                        object_end_char = window._d["token2char_end"][
                            str(object_end_token - 1)
                        ]
                        window_level_triplet_labels.append(
                            {
                                "subject": [
                                    subject_start_char,
                                    subject_end_char,
                                    subject_label_text,
                                ],
                                "relation": relation_label_text,
                                "object": [
                                    object_start_char,
                                    object_end_char,
                                    object_label_text,
                                ],
                            }
                        )
                    window._d["window_triplet_labels"] = window_level_triplet_labels

        # except Exception as e:
        #     logger.error(
        #         f"Error processing document {window._d['doc_id']} window {window._d['window_id']}: {e}"
        #     )

        return windowized_data

    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file {input_file} not found.")

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

    if relation_mapping is not None:
        with open(relation_mapping) as f:
            relation_mapping = json.load(f)

    output_file_path = Path(output_file)

    # check if file exists
    continue_from_id = None
    if output_file_path.exists():
        if not overwrite:
            # we should not overwrite the file
            # open last line of the file using tail command
            try:
                last_line = subprocess.check_output(
                    f"tail -n 1 {output_file}", shell=True
                )
                continue_from_id = json.loads(last_line)["doc_id"]
            except Exception as e:
                raise ValueError(
                    f"Could not read the last line of the output file {output_file}: {e}"
                )

            logger.info(
                f"Output file {output_file} already exists. Continuing from doc id {continue_from_id}"
            )
        else:
            logger.info(
                f"{output_file} already exists but `--overwrite` flag is set, overwriting the file."
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
                    relation_mapping,
                    labels_are_tokens,
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
                relation_mapping,
                labels_are_tokens,
            )
            for wd in windowized_data:
                f_out.write(wd.to_jsons() + "\n")
            progress_bar.update(len(batched_data))
            batched_data = []


@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def convert_to_dpr(
    input_path: str,
    output_path: str,
    documents_path: str,
    title_map: str = None,
    label_type: str = "span",
):
    if label_type not in ["span", "triplet"]:
        raise ValueError(
            f"Invalid label type: {label_type}. Supported types are: `span`, `triplet`."
        )
    documents = {}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # read entities definitions
    logger.info(f"Loading documents from {documents_path}")
    # with open(documents_path, "r") as f:
    #     for line in f:
    #         line_data = json.loads(line)
    #         title = line_data["text"].strip()
    #         definition = line_data["metadata"]["definition"].strip()
    #         documents[title] = definition

    # infer document type
    document_file_type = Path(documents_path).suffix[1:]
    if document_file_type == "jsonl":
        documents = DocumentStore.from_file(documents_path)
    elif document_file_type == "csv":
        documents = DocumentStore.from_tsv(
            documents_path, delimiter=",", quoting=csv.QUOTE_NONE, ingore_case=True
        )
    elif document_file_type == "tsv":
        documents = DocumentStore.from_tsv(
            documents_path, delimiter="\t", quoting=csv.QUOTE_NONE, ingore_case=True
        )
    else:
        raise ValueError(
            f"Unknown document file type: {document_file_type}. Supported types are: jsonl, csv, and tsv."
        )

    if title_map is not None:
        with open(title_map, "r") as f:
            title_map = json.load(f)
    else:
        title_map = {}

    # store dpr data
    dpr = []
    # lower case titles
    title_to_lower_map = {doc.text.lower(): doc.text for doc in documents}
    # store missing entities
    missing = set()
    # Read input file
    with open(input_path, "r") as f, open(output_path, "w") as f_out:
        for line in tqdm(f, desc="Processing data"):
            sentence = json.loads(line)
            # for sentence in aida_data:
            question = (
                sentence["doc_text"] if "doc_text" in sentence else sentence["text"]
            )
            positive_pssgs = []
            if label_type == "triplet":
                for idx, triplet in enumerate(sentence["window_triplet_labels"]):
                    relation = triplet["relation"]
                    if not relation:
                        continue
                    # relation = relation.strip().lower()
                    relation = relation.strip()
                    # relation = title_to_lower_map.get(relation, relation)
                    if relation in documents:
                        # def_text = documents[relation]
                        doc = documents.get_document_from_text(relation)
                        doc.metadata["passage_id"] = (
                            f"{sentence['doc_id']}_{sentence['offset']}_{idx}"
                        )
                        positive_pssgs.append(doc.to_dict())
                    else:
                        missing.add(relation)
                        print(f"Relation {relation} not found in definitions")
            else:
                for idx, entity in enumerate(sentence["window_labels"]):
                    entity = entity[2]

                    if not entity:
                        continue

                    # entity = entity.strip().lower().replace("_", " ")
                    entity = entity.strip()

                    if entity == "--NME--":
                        continue

                    # if title_map and entity in title_to_lower_map:
                    # entity = title_to_lower_map.get(entity, entity)
                    entity = title_map.get(entity, entity)
                    if entity in documents:
                        doc = documents.get_document_from_text(entity)
                        # doc.text = mapped_entity
                        doc.metadata["passage_id"] = (
                            f"{sentence['doc_id']}_{sentence['offset']}_{idx}"
                        )
                        positive_pssgs.append(doc.to_dict())
                    else:
                        missing.add(entity)
                        print(f"Entity {entity} not found in definitions")

            if len(positive_pssgs) == 0:
                continue

            dpr_sentence = {
                "id": f"{sentence['doc_id']}_{sentence['offset']}",
                "doc_topic": sentence["doc_topic"],
                "question": question,
                "positive_ctxs": positive_pssgs,
                "negative_ctxs": "",
                "hard_negative_ctxs": "",
            }
            f_out.write(json.dumps(dpr_sentence) + "\n")

        for e in missing:
            print(e)
        print(f"Number of missing entities: {len(missing)}")

    return dpr


if __name__ == "__main__":
    app()

# import ast
# import csv
# import json
# import os
# import subprocess
# from pathlib import Path

# import typer
# from tqdm import tqdm

# from relik.common.log import get_logger
# from relik.inference.data.splitters.blank_sentence_splitter import BlankSentenceSplitter
# from relik.inference.data.splitters.spacy_sentence_splitter import SpacySentenceSplitter
# from relik.inference.data.splitters.window_based_splitter import WindowSentenceSplitter
# from relik.inference.data.tokenizers.spacy_tokenizer import SpacyTokenizer
# from relik.inference.data.window.manager import WindowManager
# from relik.retriever.indexers.document import DocumentStore

# logger = get_logger(__name__)

# app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


# @app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
# def create_windows(
#     input_file: str,
#     output_file: str,
#     window_size: int = typer.Option(
#         32, parser=lambda x: None if x == "None" else int(x)
#     ),
#     window_stride: int = typer.Option(
#         16, parser=lambda x: None if x == "None" else int(x)
#     ),
#     # window_size: int | str = 32,
#     # window_stride: int = 16,
#     title_mapping: str = None,
#     relation_mapping: str = None,
#     language: str = "en",
#     tokenizer_device: str = "cpu",
#     is_split_into_words: bool = False,
#     labels_are_tokens: bool = False,
#     write_batch_size: int = 10_000,
#     overwrite: bool = False,
# ):
#     """
#     Create windows from input documents and save them to an output file.

#     Args:
#         input_file (str):
#             Path to the input file containing the documents.
#         output_file (str):
#             Path to the output file to save the windowized data.
#         window_size (int, optional):
#             Size of the window. Defaults to 32.
#         window_stride (int, optional):
#             Stride of the window. Defaults to 16.
#         title_mapping (str, optional):
#             Path to a JSON file containing a mapping of labels. Defaults to None.
#         language (str, optional):
#             Language of the documents. Defaults to "en".
#         tokenizer_device (str, optional):
#             Device to use for tokenization. Defaults to "cpu".
#         is_split_into_words (bool, optional):
#             Whether the documents are already split into words. Defaults to False.
#         write_batch_size (int, optional):
#             Number of windows to process and write at a time. Defaults to 10_000.

#     Returns:
#         None
#     """

#     def _process_batch(
#         data,
#         window_manager,
#         window_size,
#         window_stride,
#         is_split_into_words,
#         title_mapping,
#         relation_mapping,
#         labels_are_tokens,
#     ):
#         # build a doc_id to doc mapping
#         doc_id_to_doc = {int(document["doc_id"]): document for document in data}
#         if is_split_into_words:
#             text_field = "doc_words"
#         else:
#             text_field = "doc_text"

#         windowized_data, _, _ = window_manager.create_windows(
#             [document[text_field] for document in data],
#             window_size,
#             window_stride,
#             is_split_into_words=is_split_into_words,
#             doc_ids=[int(document["doc_id"]) for document in data],
#             # doc_topic=doc_topic,
#         )

#         for window in windowized_data:
#             # try:
#             # we need to add the labels
#             doc_level_labels = doc_id_to_doc[window._d["doc_id"]][
#                 "doc_span_annotations"
#             ]
#             # if we have a title mapping, we need to map the labels to the
#             # new titles
#             if title_mapping is not None:
#                 # compute the missing labels
#                 # missing_labels |= set(title_mapping.keys()) - set(
#                 #     [label for _, _, label in doc_level_labels]
#                 # )
#                 doc_level_labels = [
#                     [start, end, title_mapping.get(label, label)]
#                     for start, end, label in doc_level_labels
#                 ]

#             if relation_mapping is not None:
#                 doc_level_labels = [
#                     [start, end, relation_mapping.get(label, label)]
#                     for start, end, label in doc_level_labels
#                 ]
#             # these are the labels for the whole document, we need add them to the correct window
#             # for window in windowized_document:
#             window_level_labels = []
#             window_level_labels_but_for_tokens = []
#             if not labels_are_tokens:
#                 for doc_level_label in doc_level_labels:
#                     start_char, end_char, label_text = doc_level_label
#                     if start_char >= window.offset and end_char <= window.offset + len(
#                         window.text
#                     ):
#                         window_level_labels.append(doc_level_label)
#                 window._d["window_labels"] = window_level_labels
#                 # now we need to map the labels to the tokens
#                 for label in window_level_labels:
#                     start_char, end_char, label_text = label
#                     start_token = None
#                     end_token = None
#                     for token_id, (start, end) in enumerate(
#                         zip(
#                             window._d["token2char_start"].values(),
#                             window._d["token2char_end"].values(),
#                         )
#                     ):
#                         if start_char == start:
#                             start_token = token_id
#                         if end_char == end:
#                             end_token = token_id + 1
#                         # if we found the tokens, we can break
#                         if start_token is not None and end_token is not None:
#                             break
#                     if start_token is None or end_token is None:
#                         # raise ValueError(
#                         #     f"Could not find token for label: {label} in window: {window}"
#                         # )
#                         # warnings.warn(
#                         #     f"Could not find token for label: {label} in window: {window}. We will snap the label to the word boundaries."
#                         # )
#                         start_token = None
#                         end_token = None
#                         for token_id, (start, end) in enumerate(
#                             zip(
#                                 window._d["token2char_start"].values(),
#                                 window._d["token2char_end"].values(),
#                             )
#                         ):
#                             if start_char >= start and start_char < end:
#                                 start_token = token_id
#                             if end_char > start and end_char <= end:
#                                 end_token = token_id + 1
#                             if start_token is not None and end_token is not None:
#                                 break
#                     window_level_labels_but_for_tokens.append(
#                         [start_token, end_token, label_text]
#                     )
#                 window._d["window_labels_tokens"] = window_level_labels_but_for_tokens
#             else:
#                 # if the text is split into words, we need to map the labels to the tokens
#                 for doc_level_label in doc_level_labels:
#                     start_token, end_token, label_text = doc_level_label
#                     window_token_start = window._d["char2token_start"][
#                         str(window.offset)
#                     ]
#                     window_token_end = window._d["char2token_end"][
#                         str(window.offset + len(window.text))
#                     ]
#                     if (
#                         start_token >= window_token_start
#                         and end_token <= window_token_end + 1
#                     ):
#                         window_level_labels_but_for_tokens.append(doc_level_label)
#                 window._d["window_labels_tokens"] = window_level_labels_but_for_tokens
#                 # if the text is split into words, we need to map the labels to the characters
#                 for label in window_level_labels_but_for_tokens:
#                     start_token, end_token, label_text = label
#                     start_char = window._d["token2char_start"][str(start_token)]
#                     end_char = window._d["token2char_end"][str(end_token - 1)]
#                     window_level_labels.append([start_char, end_char, label_text])
#                 window._d["window_labels"] = window_level_labels

#             if "doc_triplet_annotations" in doc_id_to_doc[window._d["doc_id"]]:
#                 doc_level_triplet_labels = doc_id_to_doc[window._d["doc_id"]][
#                     "doc_triplet_annotations"
#                 ]
#                 if title_mapping is not None:
#                     doc_level_triplet_labels = [
#                         {
#                             "subject": [
#                                 triplet["subject"][0],
#                                 triplet["subject"][1],
#                                 title_mapping.get(
#                                     triplet["subject"][2], triplet["subject"][2]
#                                 ),
#                             ],
#                             "relation": title_mapping.get(
#                                 triplet["relation"], triplet["relation"]
#                             ),
#                             "object": [
#                                 triplet["object"][0],
#                                 triplet["object"][1],
#                                 title_mapping.get(
#                                     triplet["object"][2], triplet["object"][2]
#                                 ),
#                             ],
#                         }
#                         for triplet in doc_level_triplet_labels
#                     ]

#                 window_level_triplet_labels = []
#                 window_level_triplet_labels_but_for_tokens = []

#                 if not labels_are_tokens:
#                     for doc_level_triplet_label in doc_level_triplet_labels:
#                         subject_start_char, subject_end_char, subject_label_text = (
#                             doc_level_triplet_label["subject"]
#                         )
#                         relation_label_text = doc_level_triplet_label["relation"]
#                         object_start_char, object_end_char, object_label_text = (
#                             doc_level_triplet_label["object"]
#                         )
#                         if (
#                             subject_start_char >= window.offset
#                             and subject_end_char <= window.offset + len(window.text)
#                             and object_start_char >= window.offset
#                             and object_end_char <= window.offset + len(window.text)
#                         ):
#                             window_level_triplet_labels.append(doc_level_triplet_label)
#                     window._d["window_triplet_labels"] = window_level_triplet_labels
#                     # now we need to map the labels to the tokens
#                     for triplet in window_level_triplet_labels:
#                         subject_start_char, subject_end_char, subject_label_text = (
#                             triplet["subject"]
#                         )
#                         relation_label_text = triplet["relation"]
#                         object_start_char, object_end_char, object_label_text = triplet[
#                             "object"
#                         ]
#                         subject_start_token = None
#                         subject_end_token = None
#                         object_start_token = None
#                         object_end_token = None
#                         for token_id, (start, end) in enumerate(
#                             zip(
#                                 window._d["token2char_start"].values(),
#                                 window._d["token2char_end"].values(),
#                             )
#                         ):
#                             if subject_start_char == start:
#                                 subject_start_token = token_id
#                             if subject_end_char == end:
#                                 subject_end_token = token_id + 1
#                             if object_start_char == start:
#                                 object_start_token = token_id
#                             if object_end_char == end:
#                                 object_end_token = token_id + 1
#                         if (
#                             subject_start_token is None
#                             or subject_end_token is None
#                             or object_start_token is None
#                             or object_end_token is None
#                         ):
#                             # raise ValueError(
#                             #     f"Could not find token for triplet: {triplet} in window: {window}"
#                             # )
#                             # find entity in window_labels using character offsets, use same index to find it in window_labels_tokens
#                             if subject_start_token is None or subject_end_token is None:
#                                 subject_index = None
#                                 for idx, label in enumerate(window._d["window_labels"]):
#                                     if (
#                                         label[0] == subject_start_char
#                                         and label[1] == subject_end_char
#                                     ):
#                                         subject_index = idx
#                                         break
#                                 subject_start_token, subject_end_token, _ = window._d[
#                                     "window_labels_tokens"
#                                 ][subject_index]
#                             if object_start_token is None or object_end_token is None:
#                                 object_index = None
#                                 for idx, label in enumerate(window._d["window_labels"]):
#                                     if (
#                                         label[0] == object_start_char
#                                         and label[1] == object_end_char
#                                     ):
#                                         object_index = idx
#                                         break
#                                 object_start_token, object_end_token, _ = window._d[
#                                     "window_labels_tokens"
#                                 ][object_index]

#                         window_level_triplet_labels_but_for_tokens.append(
#                             {
#                                 "subject": [
#                                     subject_start_token,
#                                     subject_end_token,
#                                     subject_label_text,
#                                 ],
#                                 "relation": relation_label_text,
#                                 "object": [
#                                     object_start_token,
#                                     object_end_token,
#                                     object_label_text,
#                                 ],
#                             }
#                         )
#                     window._d["window_triplet_labels_tokens"] = (
#                         window_level_triplet_labels_but_for_tokens
#                     )
#                 else:
#                     # if the text is split into words, we need to map the labels to the tokens
#                     for doc_level_triplet_label in doc_level_triplet_labels:
#                         subject_start_token, subject_end_token, subject_label_text = (
#                             doc_level_triplet_label["subject"]
#                         )
#                         relation_label_text = doc_level_triplet_label["relation"]
#                         object_start_token, object_end_token, object_label_text = (
#                             doc_level_triplet_label["object"]
#                         )
#                         window_token_start = window._d["char2token_start"][
#                             str(window.offset)
#                         ]
#                         window_token_end = window._d["char2token_end"][
#                             str(window.offset + len(window.text))
#                         ]
#                         if (
#                             subject_start_token >= window_token_start
#                             and subject_end_token <= window_token_end + 1
#                             and object_end_token >= window_token_start
#                             and object_end_token <= window_token_end + 1
#                         ):
#                             window_level_triplet_labels_but_for_tokens.append(
#                                 doc_level_triplet_label
#                             )
#                     window._d["window_triplet_labels_tokens"] = (
#                         window_level_triplet_labels_but_for_tokens
#                     )
#                     # if the text is split into words, we need to map the labels to the characters
#                     for triplet in window_level_triplet_labels_but_for_tokens:
#                         subject_start_token, subject_end_token, subject_label_text = (
#                             triplet["subject"]
#                         )
#                         relation_label_text = triplet["relation"]
#                         object_start_token, object_end_token, object_label_text = (
#                             triplet["object"]
#                         )
#                         subject_start_char = window._d["token2char_start"][
#                             str(subject_start_token)
#                         ]
#                         subject_end_char = window._d["token2char_end"][
#                             str(subject_end_token - 1)
#                         ]
#                         object_start_char = window._d["token2char_start"][
#                             str(object_start_token)
#                         ]
#                         object_end_char = window._d["token2char_end"][
#                             str(object_end_token - 1)
#                         ]
#                         window_level_triplet_labels.append(
#                             {
#                                 "subject": [
#                                     subject_start_char,
#                                     subject_end_char,
#                                     subject_label_text,
#                                 ],
#                                 "relation": relation_label_text,
#                                 "object": [
#                                     object_start_char,
#                                     object_end_char,
#                                     object_label_text,
#                                 ],
#                             }
#                         )
#                     window._d["window_triplet_labels"] = window_level_triplet_labels

#         # except Exception as e:
#         #     logger.error(
#         #         f"Error processing document {window._d['doc_id']} window {window._d['window_id']}: {e}"
#         #     )

#         return windowized_data

#     if not Path(input_file).exists():
#         raise FileNotFoundError(f"Input file {input_file} not found.")

#     # windowization stuff
#     tokenizer = SpacyTokenizer(language=language, use_gpu=tokenizer_device == "cuda")
#     if window_size == "none" or window_size is None:
#         sentence_splitter = BlankSentenceSplitter()
#     elif window_size == "sentence":
#         sentence_splitter = SpacySentenceSplitter()
#     else:
#         sentence_splitter = WindowSentenceSplitter(
#             window_size=window_size, window_stride=window_stride
#         )
#     window_manager = WindowManager(tokenizer, sentence_splitter)

#     if title_mapping is not None:
#         with open(title_mapping) as f:
#             title_mapping = json.load(f)

#     if relation_mapping is not None:
#         with open(relation_mapping) as f:
#             relation_mapping = json.load(f)

#     output_file_path = Path(output_file)

#     # check if file exists
#     continue_from_id = None
#     if output_file_path.exists():
#         if not overwrite:
#             # we should not overwrite the file
#             # open last line of the file using tail command
#             try:
#                 last_line = subprocess.check_output(
#                     f"tail -n 1 {output_file}", shell=True
#                 )
#                 continue_from_id = json.loads(last_line)["doc_id"]
#             except Exception as e:
#                 raise ValueError(
#                     f"Could not read the last line of the output file {output_file}: {e}"
#                 )

#             logger.info(
#                 f"Output file {output_file} already exists. Continuing from doc id {continue_from_id}"
#             )
#         else:
#             logger.info(
#                 f"{output_file} already exists but `--overwrite` flag is set, overwriting the file."
#             )
#     else:
#         output_file_path.parent.mkdir(parents=True, exist_ok=True)
#         logger.info(f"Saving windowized data to {output_file}")

#     logger.info(f"Loading data from {input_file}")
#     batched_data = []

#     # get number of lines in the file
#     # run bash command to get the number of lines in the file
#     try:
#         total_lines = int(
#             subprocess.check_output(
#                 f"wc -l {input_file} | awk '{{print $1}}'", shell=True
#             )
#         )
#     except Exception as e:
#         logger.error(f"Error getting number of lines in the file: {e}")
#         total_lines = None

#     progress_bar = tqdm(total=total_lines)
#     write_mode = "a" if continue_from_id is not None else "w"
#     with open(input_file) as f_in, open(output_file, write_mode) as f_out:
#         for line in f_in:
#             if continue_from_id is not None:
#                 # we need to skip until we reach the last written line
#                 current_id = json.loads(line)["doc_id"]
#                 if current_id != continue_from_id:
#                     progress_bar.update(1)
#                     continue
#                 else:
#                     continue_from_id = None
#             batched_data.append(json.loads(line))
#             if len(batched_data) == write_batch_size:
#                 windowized_data = _process_batch(
#                     batched_data,
#                     window_manager,
#                     window_size,
#                     window_stride,
#                     is_split_into_words,
#                     title_mapping,
#                     relation_mapping,
#                     labels_are_tokens,
#                 )
#                 for wd in windowized_data:
#                     f_out.write(wd.to_jsons() + "\n")
#                 progress_bar.update(len(batched_data))
#                 batched_data = []

#         if len(batched_data) > 0:
#             windowized_data = _process_batch(
#                 batched_data,
#                 window_manager,
#                 window_size,
#                 window_stride,
#                 is_split_into_words,
#                 title_mapping,
#                 relation_mapping,
#                 labels_are_tokens,
#             )
#             for wd in windowized_data:
#                 f_out.write(wd.to_jsons() + "\n")
#             progress_bar.update(len(batched_data))
#             batched_data = []


# @app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
# def convert_to_dpr(
#     input_path: str,
#     output_path: str,
#     documents_path: str,
#     title_map: str = None,
#     label_type: str = "span",
# ):
#     if label_type not in ["span", "triplet"]:
#         raise ValueError(
#             f"Invalid label type: {label_type}. Supported types are: `span`, `triplet`."
#         )
#     documents = {}
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     # read entities definitions
#     logger.info(f"Loading documents from {documents_path}")
#     # with open(documents_path, "r") as f:
#     #     for line in f:
#     #         line_data = json.loads(line)
#     #         title = line_data["text"].strip()
#     #         definition = line_data["metadata"]["definition"].strip()
#     #         documents[title] = definition

#     # infer document type
#     document_file_type = Path(documents_path).suffix[1:]
#     if document_file_type == "jsonl":
#         documents = DocumentStore.from_file(documents_path)
#     elif document_file_type == "csv":
#         documents = DocumentStore.from_tsv(
#             documents_path, delimiter=",", quoting=csv.QUOTE_NONE, ingore_case=True
#         )
#     elif document_file_type == "tsv":
#         documents = DocumentStore.from_tsv(
#             documents_path, delimiter="\t", quoting=csv.QUOTE_NONE, ingore_case=True
#         )
#     else:
#         raise ValueError(
#             f"Unknown document file type: {document_file_type}. Supported types are: jsonl, csv, and tsv."
#         )

#     if title_map is not None:
#         with open(title_map, "r") as f:
#             title_map = json.load(f)
#     else:
#         title_map = {}

#     # store dpr data
#     dpr = []
#     # lower case titles
#     # title_to_lower_map = {doc.text.lower(): doc.text for doc in documents}
#     # store missing entities
#     missing = set()
#     # Read input file
#     with open(input_path, "r") as f, open(output_path, "w") as f_out:
#         for line in tqdm(f, desc="Processing data"):
#             sentence = json.loads(line)
#             # for sentence in aida_data:
#             question = sentence["text"]
#             positive_pssgs = []

#             if label_type == "triplet":
#                 for idx, triplet in enumerate(sentence["window_triplet_labels"]):
#                     relation = triplet["relation"]
#                     if not relation:
#                         continue
#                     # relation = relation.strip().lower()
#                     relation = relation.strip()
#                     # relation = title_to_lower_map.get(relation, relation)
#                     if relation in documents:
#                         # def_text = documents[relation]
#                         doc = documents.get_document_from_text(relation)
#                         doc.metadata["passage_id"] = (
#                             f"{sentence['doc_id']}_{sentence['offset']}_{idx}"
#                         )
#                         positive_pssgs.append(doc.to_dict())
#                     else:
#                         missing.add(relation)
#                         print(f"Relation {relation} not found in definitions")
#             else:
#                 for idx, entity in enumerate(sentence["window_labels"]):
#                     entity = entity[2]

#                     if not entity:
#                         continue

#                     # entity = entity.strip().lower().replace("_", " ")
#                     entity = entity.strip()

#                     if entity == "--NME--":
#                         continue

#                     # if title_map and entity in title_to_lower_map:
#                     # entity = title_to_lower_map.get(entity, entity)
#                     entity = title_map.get(entity, entity)
#                     if entity in documents:
#                         doc = documents.get_document_from_text(entity)
#                         # doc.text = mapped_entity
#                         doc.metadata["passage_id"] = (
#                             f"{sentence['doc_id']}_{sentence['offset']}_{idx}"
#                         )
#                         positive_pssgs.append(doc.to_dict())
#                     else:
#                         missing.add(entity)
#                         print(f"Entity {entity} not found in definitions")

#             if len(positive_pssgs) == 0:
#                 continue

#             dpr_sentence = {
#                 "id": f"{sentence['doc_id']}_{sentence['offset']}",
#                 "doc_topic": sentence["doc_topic"],
#                 "question": question,
#                 "positive_ctxs": positive_pssgs,
#                 "negative_ctxs": "",
#                 "hard_negative_ctxs": "",
#             }
#             f_out.write(json.dumps(dpr_sentence) + "\n")

#         for e in missing:
#             print(e)
#         print(f"Number of missing entities: {len(missing)}")

#     return dpr


# if __name__ == "__main__":
#     app()
