import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from relik.common.log import get_logger

logger = get_logger()


def aida_to_dpr(
    conll_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    documents_path: Optional[Union[str, os.PathLike]] = None,
    title_map: Optional[Union[str, os.PathLike]] = None,
) -> List[Dict[str, Any]]:
    documents = {}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # read entities definitions
    logger.info(f"Loading documents from {documents_path}")
    with open(documents_path, "r") as f:
        for line in f:
            line_data = json.loads(line)
            title = line_data["text"].strip()
            definition = line_data["metadata"]["definition"].strip()
            documents[title] = definition

    if title_map is not None:
        with open(title_map, "r") as f:
            title_map = json.load(f)

    # store dpr data
    dpr = []
    # lower case titles
    title_to_lower_map = {title.lower(): title for title in documents.keys()}
    # store missing entities
    missing = set()
    # Read AIDA file
    with open(conll_path, "r") as f, open(output_path, "w") as f_out:
        for line in tqdm(f, desc="Processing AIDA data"):
            sentence = json.loads(line)
            # for sentence in aida_data:
            question = sentence["text"]
            positive_pssgs = []
            for idx, entity in enumerate(sentence["window_labels"]):
                entity = entity[2]
                if not entity:
                    continue
                entity = entity.strip().lower().replace("_", " ")
                # if title_map and entity in title_to_lower_map:
                entity = title_to_lower_map.get(entity, entity)
                if entity in documents:
                    def_text = documents[entity]
                    positive_pssgs.append(
                        {
                            "title": title_to_lower_map[entity.lower()],
                            "text": f"{title_to_lower_map[entity.lower()]} <def> {def_text}",
                            "passage_id": f"{sentence['doc_id']}_{sentence['offset']}_{idx}",
                        }
                    )
                else:
                    missing.add(entity)
                    print(f"Entity {entity} not found in definitions")

            if len(positive_pssgs) == 0:
                continue

            dpr_sentence = {
                "id": f"{sentence['doc_id']}_{sentence['offset']}",
                "doc_topic": sentence["doc_topic"],
                "question": question,
                "answers": "",
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to AIDA file")
    parser.add_argument("output", type=str, help="Path to output file")
    parser.add_argument("documents", type=str, help="Path to entities definitions file")
    parser.add_argument("--title_map", type=str, help="Path to title map file")
    args = parser.parse_args()

    # Convert to DPR
    aida_to_dpr(args.input, args.output, args.documents, args.title_map)