import json

from tqdm import tqdm

with open("data/processed/aida_from_edo.jsonl", "r") as f:
    aida = [json.loads(l) for l in f]

aida_train, aida_dev, aida_test = [], [], []

for document in tqdm(aida):
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

    document["doc_id"] = doc_id

    if split == "train":
        aida_train.append(document)
    elif split == "dev":
        aida_dev.append(document)
    else:
        aida_test.append(document)

with open("data/processed/aida_train.jsonl", "w") as f:
    for document in aida_train:
        f.write(json.dumps(document) + "\n")

with open("data/processed/aida_dev.jsonl", "w") as f:
    for document in aida_dev:
        f.write(json.dumps(document) + "\n")

with open("data/processed/aida_test.jsonl", "w") as f:
    for document in aida_test:
        f.write(json.dumps(document) + "\n")
