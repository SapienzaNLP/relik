import json
from pathlib import Path
import unicodedata

from relik.common.log import get_logger

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([char for char in nfkd_form if not unicodedata.combining(char)])

logger = get_logger()
mapping_relations = {
    "/location/location/contains": "contains",
    "/location/neighborhood/neighborhood_of": "neighborhood",
    "/people/person/place_lived": "residence",
    "/location/country/capital": "capital",
    "/people/person/place_of_birth": "place of birth",
    "/location/country/administrative_divisions": "contains division",
    "/business/person/company": "company",
    "/location/administrative_division/country": "division",
    "/people/deceased_person/place_of_death": "place of death",
    "/business/company/place_founded": "company location",
    "/people/person/children": "child",
    "/people/person/nationality": "nationality",
    "/people/person/religion": "religion",
    "/people/ethnicity/geographic_distribution": "country of origin",
    "/business/company_shareholder/major_shareholder_of": "shareholder of",
    "/business/company/advisors": "advisor",
    "/sports/sports_team/location": "sports team location",
    "/sports/sports_team_location/teams": "sports team",
    "/business/company/major_shareholders": "shareholders",
    "/business/company/founders": "founded by",
    "/business/company/industry": "industry",
    "/people/person/profession": "occupation",
    "/people/person/ethnicity": "ethnic background",
    "/people/ethnicity/people": "ethnicity",
}

def preprocess_nyt(input_path: str, output_path: str, legacy_format: bool = True):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for set_name in ["train", "valid", "test"]:
        with open(input_path / f"raw_{set_name}.json", "r") as f:
            with open(output_path / f"{set_name}.jsonl", "w") as out:
                for idx, line in enumerate(f):
                    data = json.loads(line)
                    new_data = {}
                    new_data["doc_id"] = idx
                    new_data["doc_words"] = data["sentText"].split()
                    tokens_visited = new_data["doc_words"].copy()
                    new_data["doc_span_annotations"] = []
                    entity_names = []
                    for entity in data["entityMentions"]:
                        new_ent = []
                        type_ent = entity["label"]
                        token_text = entity["text"].split()
                        entity_names.append(entity["text"])
                        # find token position. If repeated, take the next one
                        for idx in range(len(tokens_visited)):
                            if tokens_visited[idx:idx+len(token_text)] == token_text:
                                start = idx
                                end = idx + len(token_text)
                                if not legacy_format:
                                    tokens_visited[idx:idx+len(token_text)] = [""] * len(token_text)
                                break
                        new_ent.append(start)
                        new_ent.append(end)
                        new_ent.append(type_ent)
                        new_data["doc_span_annotations"].append(new_ent)
                    new_data["doc_triplet_annotations"] = []
                    seen = []
                    for relation in data["relationMentions"]:
                        if relation in seen:
                            continue
                        num_rel = sum([1 for x in data["relationMentions"] if x == relation])
                        type_rel = relation["label"]
                        subject_text = remove_accents(relation["em1Text"]) # remove accents 
                        object_text = remove_accents(relation["em2Text"]) # remove accents
                        subject_matches = [i for i, x in enumerate(entity_names) if x == subject_text]
                        object_matches = [i for i, x in enumerate(entity_names) if x == object_text]
                        # take as many as possible up to num_rel
                        count = 0
                        triplet = {}
                        triplet["subject"] = []
                        triplet["object"] = []
                        triplet["relation"] = mapping_relations[type_rel]
                        if legacy_format:
                            # sort subject_matches and object_matches
                            subject_matches.sort()
                            object_matches.sort()
                            subject_matches = [subject_matches[0]] * len(subject_matches)
                            object_matches = [object_matches[0]] * len(object_matches)
                        for idx in subject_matches:
                            for idx2 in object_matches:
                                triplet["subject"].append(new_data["doc_span_annotations"][idx][0])
                                triplet["subject"].append(new_data["doc_span_annotations"][idx][1])
                                triplet["subject"].append(new_data["doc_span_annotations"][idx][2])
                                triplet["object"].append(new_data["doc_span_annotations"][idx2][0])
                                triplet["object"].append(new_data["doc_span_annotations"][idx2][1])
                                triplet["object"].append(new_data["doc_span_annotations"][idx2][2])
                                count += 1
                                new_data["doc_triplet_annotations"].append(triplet)
                                triplet = {"subject": [], "object": [], "relation": mapping_relations[type_rel]}
                                if count == num_rel:
                                    seen.append(relation)
                                    break
                            if count == num_rel:
                                break
                    out.write(json.dumps(new_data) + "\n")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to NYT folder")
    parser.add_argument("output", type=str, help="Path to output folder")
    parser.add_argument("--legacy", default=True, action="store_true", help="Use legacy format")

    args = parser.parse_args()

    preprocess_nyt(args.input, args.output, args.legacy)