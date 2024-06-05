import argparse
import json
import logging
import os
from pathlib import Path
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator, List, Optional, Tuple
from urllib import parse

from relik.inference.annotator import Relik
from relik.inference.data.objects import RelikOutput

# sys.path += ['../']
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


logger = logging.getLogger(__name__)


class GerbilAlbyManager:
    def __init__(
        self,
        annotator: Optional[Relik] = None,
        response_logger_dir: Optional[str] = None,
    ) -> None:
        self.annotator = annotator
        self.response_logger_dir = response_logger_dir
        self.predictions_counter = 0
        self.labels_mapping = None
        self.retriever_batch_size = 32
        self.reader_batch_size = 32
        self.top_k = 100

    def annotate(self, document: str):
        relik_output: RelikOutput = self.annotator(
            document,
            retriever_batch_size=self.retriever_batch_size,
            reader_batch_size=self.reader_batch_size,
            top_k=self.top_k,
        )
        annotations = [(ss, se, l) for ss, se, l, _ in relik_output.spans]
        if self.labels_mapping is not None:
            return [
                (ss, se, self.labels_mapping.get(l, l)) for ss, se, l in annotations
            ]
        return annotations

    def set_mapping_file(self, mapping_file_path: str):
        with open(mapping_file_path) as f:
            labels_mapping = json.load(f)
        self.labels_mapping = {v: k for k, v in labels_mapping.items()}

    def write_response_bundle(
        self,
        document: str,
        new_document: str,
        annotations: list,
        mapped_annotations: list,
    ) -> None:
        if self.response_logger_dir is None:
            return

        if not os.path.isdir(self.response_logger_dir):
            os.mkdir(self.response_logger_dir)

        with open(
            f"{self.response_logger_dir}/{self.predictions_counter}.json", "w"
        ) as f:
            out_json_obj = dict(
                document=document,
                new_document=new_document,
                annotations=annotations,
                mapped_annotations=mapped_annotations,
            )

            out_json_obj["span_annotations"] = [
                (ss, se, document[ss:se], label) for (ss, se, label) in annotations
            ]

            out_json_obj["span_mapped_annotations"] = [
                (ss, se, new_document[ss:se], label)
                for (ss, se, label) in mapped_annotations
            ]

            json.dump(out_json_obj, f, indent=2)

        self.predictions_counter += 1


manager = GerbilAlbyManager()


def preprocess_document(document: str) -> Tuple[str, List[Tuple[int, int]]]:
    pattern_subs = {
        "-LPR- ": " (",
        "-RPR-": ")",
        "\n\n": "\n",
        "-LRB-": "(",
        "-RRB-": ")",
        '","': ",",
    }

    document_acc = document
    curr_offset = 0
    char2offset = []

    matchings = re.finditer("({})".format("|".join(pattern_subs)), document)
    for span_matching in sorted(matchings, key=lambda x: x.span()[0]):
        span_start, span_end = span_matching.span()
        span_start -= curr_offset
        span_end -= curr_offset

        span_text = document_acc[span_start:span_end]
        span_sub = pattern_subs[span_text]
        document_acc = document_acc[:span_start] + span_sub + document_acc[span_end:]

        offset = len(span_text) - len(span_sub)
        curr_offset += offset

        char2offset.append((span_start + len(span_sub), curr_offset))

    return document_acc, char2offset


def map_back_annotations(
    annotations: List[Tuple[int, int, str]], char_mapping: List[Tuple[int, int]]
) -> Iterator[Tuple[int, int, str]]:
    def map_char(char_idx: int) -> int:
        current_offset = 0
        for offset_idx, offset_value in char_mapping:
            if char_idx >= offset_idx:
                current_offset = offset_value
            else:
                break
        return char_idx + current_offset

    for ss, se, label in annotations:
        yield map_char(ss), map_char(se), label


def annotate(document: str) -> List[Tuple[int, int, str]]:
    new_document, mapping = preprocess_document(document)
    logger.info("Mapping: " + str(mapping))
    logger.info("Document: " + str(document))
    annotations = [
        (cs, ce, label.replace(" ", "_"))
        for cs, ce, label in manager.annotate(new_document)
    ]
    logger.info("New document: " + str(new_document))
    mapped_annotations = (
        list(map_back_annotations(annotations, mapping))
        if len(mapping) > 0
        else annotations
    )

    logger.info(
        "Annotations: "
        + str([(ss, se, document[ss:se], ann) for ss, se, ann in mapped_annotations])
    )

    manager.write_response_bundle(
        document, new_document, mapped_annotations, annotations
    )

    if not all(
        [
            new_document[ss:se] == document[mss:mse]
            for (mss, mse, _), (ss, se, _) in zip(mapped_annotations, annotations)
        ]
    ):
        diff_mappings = [
            (new_document[ss:se], document[mss:mse])
            for (mss, mse, _), (ss, se, _) in zip(mapped_annotations, annotations)
        ]
        return None
    assert all(
        [
            document[mss:mse] == new_document[ss:se]
            for (mss, mse, _), (ss, se, _) in zip(mapped_annotations, annotations)
        ]
    ), (mapped_annotations, annotations)

    return [(cs, ce - cs, label) for cs, ce, label in mapped_annotations]


class GetHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()
        doc_text = read_json(post_data)
        # try:
        response = annotate(doc_text)

        self.wfile.write(bytes(json.dumps(response), "utf-8"))
        return


def read_json(post_data):
    data = json.loads(post_data.decode("utf-8"))
    # logger.info("received data:", data)
    text = data["text"]
    # spans = [(int(j["start"]), int(j["length"])) for j in data["spans"]]
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relik-model-name", required=True)
    parser.add_argument("--retriever-device", default="cuda")
    parser.add_argument("--index-device", type=int, default=32)
    parser.add_argument("--reader-device", default="cuda")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--retriever-batch-size", type=int, default=32)
    parser.add_argument("--reader-batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--responses-log-dir")
    parser.add_argument("--log-file", default="experiments/logging.txt")
    parser.add_argument("--mapping-file")
    return parser.parse_args()


def main():
    args = parse_args()

    responses_log_dir = Path(args.responses_log_dir)
    responses_log_dir.mkdir(parents=True, exist_ok=True)

    # init manager
    manager.response_logger_dir = args.responses_log_dir
    manager.annotator = Relik.from_pretrained(
        args.relik_model_name,
        document_index_device=args.retriever_device,
        retriever_device=args.retriever_device,
        reader_device=args.reader_device,
        precision=args.precision,
        dataset_kwargs={"use_nme": True},
    )

    # set global batch sizes
    manager.retriever_batch_size = args.retriever_batch_size
    manager.reader_batch_size = args.reader_batch_size
    manager.top_k = args.top_k

    if args.mapping_file is not None:
        manager.set_mapping_file(args.mapping_file)

    # port = 6654
    port = 5555
    server = HTTPServer(("localhost", port), GetHandler)
    logger.info(f"Starting server at http://localhost:{port}")

    # Create a file handler and set its level
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a log formatter and set it on the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        exit(0)


if __name__ == "__main__":
    main()
