from typing import List

from relik.reader.data.relik_reader_sample import RelikReaderSample
from relik.reader.utils.special_symbols import NME_SYMBOL


def merge_patches_predictions(sample) -> None:
    sample._d["predicted_window_labels"] = dict()
    predicted_window_labels = sample._d["predicted_window_labels"]

    sample._d["span_title_probabilities"] = dict()
    span_title_probabilities = sample._d["span_title_probabilities"]

    span2title = dict()
    for _, patch_info in sorted(sample.patches.items(), key=lambda x: x[0]):
        # selecting span predictions
        for predicted_title, predicted_spans in patch_info[
            "predicted_window_labels"
        ].items():
            for pred_span in predicted_spans:
                pred_span = tuple(pred_span)
                curr_title = span2title.get(pred_span)
                if curr_title is None or curr_title == NME_SYMBOL:
                    span2title[pred_span] = predicted_title
                # else:
                #     print("Merging at patch level")

        # selecting span predictions probability
        for predicted_span, titles_probabilities in patch_info[
            "span_title_probabilities"
        ].items():
            if predicted_span not in span_title_probabilities:
                span_title_probabilities[predicted_span] = titles_probabilities

    for span, title in span2title.items():
        if title not in predicted_window_labels:
            predicted_window_labels[title] = list()
        predicted_window_labels[title].append(span)


def remove_duplicate_samples(
    samples: List[RelikReaderSample],
) -> List[RelikReaderSample]:
    seen_sample = set()
    samples_store = []
    for sample in samples:
        sample_id = f"{sample.doc_id}#{sample.sent_id}#{sample.offset}"
        if sample_id not in seen_sample:
            seen_sample.add(sample_id)
            samples_store.append(sample)
    return samples_store
