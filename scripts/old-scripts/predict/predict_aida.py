import json
from pprintpp import pprint

from relik.inference.annotator import Relik
from relik.inference.data.objects import TaskType
from relik.reader.pytorch_modules.span import RelikReaderForSpanExtraction
from relik.retriever.pytorch_modules.model import GoldenRetriever


def main():
    # retriever = GoldenRetriever(
    #     question_encoder="riccorl/retriever-relik-entity-linking-aida-wikipedia-base-question-encoder",
    #     document_index="riccorl/retriever-relik-entity-linking-aida-wikipedia-base-index",
    #     device="cuda",
    #     index_device="cpu",
    #     precision=16,
    #     index_precision=32,
    # )
    # reader = RelikReaderForSpanExtraction(
    #     "riccorl/reader-relik-entity-linking-aida-wikipedia-small"
    # )

    # relik = Relik(
    #     retriever=retriever,
    #     reader=reader,
    #     top_k=100,
    #     window_size=32,
    #     window_stride=16,
    #     task=TaskType.SPAN,
    # )
    # relik.save_pretrained(
    #     "relik-entity-linking-aida-wikipedia-tiny",
    #     save_weights=False,
    #     push_to_hub=True,
    #     # reader_model_id="reader-relik-entity-linking-aida-wikipedia-small",
    #     # retriever_model_id="retriever-relik-entity-linking-aida-wikipedia-base",
    # )

    reader = RelikReaderForSpanExtraction(
        "riccorl/relik-reader-deberta-base-retriever-relik-entity-linking-aida-wikipedia-large",
        device="cuda",
        precision="fp16",  # , reader_device="cpu", reader_precision="fp32"
        dataset_kwargs={"use_nme": True},
    )

    relik = Relik(reader=reader)

    with open(
        "/home/ric/Projects/relik-sapienzanlp/data/reader/retriever-relik-entity-linking-aida-wikipedia-base-question-encoder/testa_windowed_candidates.jsonl"
    ) as f:
        data = [json.loads(line) for line in f]

    text = [data["text"] for data in data]
    candidates = [data["span_candidates"] for data in data]

    predictions = relik(
        text,
        candidates=candidates,
        window_size="none",
        annotation_type="char",
        progress_bar=True,
        reader_batch_size=32,
    )

    output = []

    for p, s in zip(predictions, data):
        output.append(
            {
                "doc_id": s["doc_id"],
                "window_id": s["window_id"],
                "text": s["text"],
                "window_labels": [
                    [span[0] - s["offset"], span[1] - s["offset"], span[2]]
                    for span in s["window_labels"]
                ],
                "predictions": [[span.start, span.end, span.label] for span in p.spans],
            }
        )

    with open(
        "/home/ric/Projects/relik-sapienzanlp/experiments/predictions/deberta-large/testa.jsonl",
        "w",
    ) as f:
        for line in output:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
