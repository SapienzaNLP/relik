import argparse
import json
from pathlib import Path
import re
from typing import List, Tuple

from tqdm import tqdm

LABELS_REGEX = "[{][^}]+[}] \[[^]]+\]"
REPLACE_PATTERNS = [
    ("BULLET::::- ", ""),
    ("( )", ""),
    ("  ", " "),
    ("   ", " "),
    ("    ", " "),
]
LABELS_FILTERING_FUNCTIONS = [lambda x: x.startswith("List of")]
SUBSTITUTION_PATTERN = "#$#"


def process_annotation(ann_surface_text: str) -> Tuple[str, str]:
    mention, label = ann_surface_text.split("} [")
    mention = mention.replace("{", "").strip()
    label = label.replace("]", "").strip()
    return mention, label


def substitute_annotations(
    annotations: List[str], sub_line: str
) -> Tuple[str, List[Tuple[int, int, str]]]:
    final_annotations_store = []
    for annotation in annotations:
        mention, label = process_annotation(annotation)
        start_char = sub_line.index(SUBSTITUTION_PATTERN)
        end_char = start_char + len(mention)
        sub_line = sub_line.replace(SUBSTITUTION_PATTERN, mention, 1)
        assert sub_line[start_char:end_char] == mention
        if any([fl(label) for fl in LABELS_FILTERING_FUNCTIONS]):
            continue
        final_annotations_store.append((start_char, end_char, label))
    return sub_line, final_annotations_store


def preprocess_line(line: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    for rps, rpe in REPLACE_PATTERNS:
        line = line.replace(rps, rpe)

    annotations = re.findall(LABELS_REGEX, line)
    sub_line = re.sub(LABELS_REGEX, SUBSTITUTION_PATTERN, line)
    return substitute_annotations(annotations, sub_line)


def preprocess_genre_el_file(
    file_path: str, output_path: str, limit_lines: int = -1
) -> None:
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path) as fi, open(output_path, "w") as fo:
        for i, line in tqdm(enumerate(fi)):
            text, annotations = preprocess_line(line.strip())
            fo.write(
                json.dumps(dict(doc_id=i, doc_text=text, doc_annotations=annotations))
                + "\n"
            )
            if limit_lines == i:
                break


def main():

    arg_parser = argparse.ArgumentParser("Preprocess Genre BLINK file.")
    arg_parser.add_argument("input_file", type=str)
    arg_parser.add_argument("output_file", type=str)
    arg_parser.add_argument("--limit-lines", type=int, default=-1)
    args = arg_parser.parse_args()

    preprocess_genre_el_file(args.input_file, args.output_file, args.limit_lines)


if __name__ == "__main__":
    main()
