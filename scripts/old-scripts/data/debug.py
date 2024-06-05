from relik.inference.data.splitters.window_based_splitter import WindowSentenceSplitter
from relik.inference.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from relik.inference.data.window.manager import WindowManager


document = {
    "doc_id": "-DOCSTART- (956testa SOCCER)",
    "doc_text": "SOCCER - RESULTS OF SOUTH KOREAN PRO-SOCCER GAMES . SEOUL 1996-08-30 Results of South Korean pro-soccer games played on Thursday . Pohang 3 Ulsan 2 ( halftime 1-0 ) Puchon 2 Chonbuk 1 ( halftime 1-1 ) Standings after games played on Thursday ( tabulate under - won , drawn , lost , goals for , goals against , points ) : W D L G / F G / A P Puchon 3 1 0 6 1 10 Chonan 3 0 1 13 10 9 Pohang 2 1 1 11 10 7 Suwan 1 3 0 7 3 6 Ulsan 1 0 2 8 9 3 Anyang 0 3 1 6 9 3 Chonnam 0 2 1 4 5 2 Pusan 0 2 1 3 7 2 Chonbuk 0 0 3 3 7 0",
    "doc_annotations": [
        [20, 32, "South Korea"],
        [52, 57, "Seoul"],
        [80, 92, "South Korea"],
        [131, 137, "Pohang Steelers"],
        [140, 145, "Ulsan Hyundai FC"],
        [165, 171, "--NME--"],
        [174, 181, "--NME--"],
        [341, 347, "--NME--"],
        [361, 367, "--NME--"],
        [382, 388, "Pohang Steelers"],
        [403, 408, "--NME--"],
        [421, 426, "Ulsan Hyundai FC"],
        [439, 445, "Anyang LG Cheetahs"],
        [458, 465, "--NME--"],
        [478, 483, "--NME--"],
        [496, 503, "--NME--"],
    ],
}

tokenizer = SpacyTokenizer(language="en")
sentence_splitter = WindowSentenceSplitter(window_size=32, window_stride=16)

window_manager = WindowManager(splitter=sentence_splitter, tokenizer=tokenizer)

doc_info = document["doc_id"]
doc_info = doc_info.replace("-DOCSTART-", "").replace("(", "").replace(")", "").strip()
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
    32,
    16,
    doc_ids=doc_id,
    doc_topic=doc_topic,
)

print(windowized_document)
