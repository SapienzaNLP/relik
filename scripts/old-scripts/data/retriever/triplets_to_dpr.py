import json
import os
from pathlib import Path
from typing import Union, Dict, List, Optional, Any

from tqdm import tqdm

# from transformers import AutoTokenizer, BertTokenizer


def aida_to_dpr(
    conll_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    definitions_path: Optional[Union[str, os.PathLike]] = None,
) -> List[Dict[str, Any]]:
    definitions = {}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # read entities definitions
    with open(definitions_path, "r") as f:
        for line in f:
            line_data = json.loads(line)
            title = line_data["text"].strip()
            definition = line_data["metadata"]["definition"].strip()
            definitions[title] = definition
            # title, definition = line.split(" <def> ")
            # title = title.strip()
            # definition = definition.strip()
            # definitions[title] = definition

    dpr = []

    title_to_lower_map = {title.lower(): title for title in definitions.keys()}
    
    missing = set()

    # Read AIDA file
    with open(conll_path, "r") as f, open(output_path, "w") as f_out:
        for line in tqdm(f):
            sentence = json.loads(line)
            # for sentence in aida_data:
            question = sentence["text"]
            positive_pssgs = []
            for idx, triplet in enumerate(sentence["triplets"]):
                relation = triplet["relation"]["name"]
                if not relation:
                    continue
                if relation in definitions:
                    def_text = definitions[relation]
                    positive_pssgs.append(
                        {
                            "title": title_to_lower_map[relation.lower()],
                            "text": f"{title_to_lower_map[relation.lower()]} <def> {def_text}",
                            "passage_id": f"{sentence['doc_id']}_{sentence['offset']}_{idx}",
                        }
                    )
                else:
                    missing.add(relation)
                    # print(f"Entity {entity} not found in definitions")

            if len(positive_pssgs) == 0:
                continue

            dpr_sentence = {
                "id": f"{sentence['doc_id']}_{sentence['offset']}",
                "doc_topic": sentence["doc_topic"] if "doc_topic" in sentence else "",
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
    parser.add_argument(
        "--definitions", type=str, help="Path to entities definitions file"
    )
    args = parser.parse_args()

    # Convert to DPR
    aida_to_dpr(args.input, args.output, args.definitions)