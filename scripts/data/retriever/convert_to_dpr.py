import json
import os
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from relik.common.log import get_logger
from relik.retriever.indexers.document import DocumentStore

logger = get_logger()


def convert_to_dpr(
    input_path: Union[str, os.PathLike],
    output_path: Union[str, os.PathLike],
    documents_path: Optional[Union[str, os.PathLike]] = None,
    title_map: Optional[Union[str, os.PathLike]] = None,
    label_type: Optional[bool] = False,
) -> List[Dict[str, Any]]:
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

    # store dpr data
    dpr = []
    # lower case titles
    title_to_lower_map = {title.lower(): title for title in documents.keys()}
    # store missing entities
    missing = set()
    # Read input file
    with open(input_path, "r") as f, open(output_path, "w") as f_out:
        for line in tqdm(f, desc="Processing data"):
            sentence = json.loads(line)
            # for sentence in aida_data:
            question = sentence["text"]
            positive_pssgs = []
            if label_type:
                for idx, triplet in enumerate(sentence["window_triplet_labels"]):
                    relation = triplet["relation"]
                    if not relation:
                        continue
                    relation = relation.strip().lower()
                    relation = title_to_lower_map.get(relation, relation)
                    if relation in documents:
                        def_text = documents[relation]
                        positive_pssgs.append(
                            {
                                "title": title_to_lower_map[relation.lower()],
                                "text": f"{title_to_lower_map[relation.lower()]} <def> {def_text}",
                                "passage_id": f"{sentence['doc_id']}_{sentence['offset']}_{idx}",
                            }
                        )
                    else:
                        missing.add(relation)
                        print(f"Relation {relation} not found in definitions")
            else:
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
    parser.add_argument("input", type=str, help="Path to windowized file")
    parser.add_argument("output", type=str, help="Path to output file")
    parser.add_argument("documents", type=str, help="Path to entities definitions file")
    parser.add_argument("--title_map", type=str, help="Path to title map file")
    parser.add_argument("--relations", action="store_true", help="Use relation labels")
    args = parser.parse_args()

    # Convert to DPR
    convert_to_dpr(args.input, args.output, args.documents, args.title_map, args.relations)
