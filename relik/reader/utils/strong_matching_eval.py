from typing import Dict, List

from lightning.pytorch.callbacks import Callback

from relik.reader.data.relik_reader_sample import RelikReaderSample
from relik.reader.utils.relik_reader_predictor import RelikReaderPredictor
from relik.reader.utils.metrics import f1_measure, safe_divide
from relik.reader.utils.special_symbols import NME_SYMBOL


class StrongMatching:
    def __call__(self, predicted_samples: List[RelikReaderSample]) -> Dict:
        # accumulators
        correct_predictions = 0
        correct_predictions_at_k = 0
        total_predictions = 0
        total_gold = 0
        correct_span_predictions = 0
        miss_due_to_candidates = 0

        # prediction index stats
        avg_correct_predicted_index = []
        avg_wrong_predicted_index = []
        less_index_predictions = []

        # collect data from samples
        for sample in predicted_samples:
            predicted_annotations = sample.predicted_window_labels_chars
            predicted_annotations_probabilities = sample.probs_window_labels_chars
            gold_annotations = {
                (ss, se, entity)
                for ss, se, entity in sample.window_labels
                if entity != NME_SYMBOL
            }
            total_predictions += len(predicted_annotations)
            total_gold += len(gold_annotations)

            # correct named entity detection
            predicted_spans = {(s, e) for s, e, _ in predicted_annotations}
            gold_spans = {(s, e) for s, e, _ in gold_annotations}
            correct_span_predictions += len(predicted_spans.intersection(gold_spans))

            # correct entity linking
            correct_predictions += len(
                predicted_annotations.intersection(gold_annotations)
            )

            for ss, se, ge in gold_annotations.difference(predicted_annotations):
                if ge not in sample.span_candidates:
                    miss_due_to_candidates += 1
                if ge in predicted_annotations_probabilities.get((ss, se), set()):
                    correct_predictions_at_k += 1

            # indices metrics
            predicted_spans_index = {
                (ss, se): ent for ss, se, ent in predicted_annotations
            }
            gold_spans_index = {(ss, se): ent for ss, se, ent in gold_annotations}

            for pred_span, pred_ent in predicted_spans_index.items():
                gold_ent = gold_spans_index.get(pred_span)

                if pred_span not in gold_spans_index:
                    continue

                # missing candidate
                if gold_ent not in sample.span_candidates:
                    continue

                gold_idx = sample.span_candidates.index(gold_ent)
                if gold_idx is None:
                    continue
                pred_idx = sample.span_candidates.index(pred_ent)

                if gold_ent != pred_ent:
                    avg_wrong_predicted_index.append(pred_idx)

                    if gold_idx is not None:
                        if pred_idx > gold_idx:
                            less_index_predictions.append(0)
                        else:
                            less_index_predictions.append(1)

                else:
                    avg_correct_predicted_index.append(pred_idx)

        # compute NED metrics
        span_precision = safe_divide(correct_span_predictions, total_predictions)
        span_recall = safe_divide(correct_span_predictions, total_gold)
        span_f1 = f1_measure(span_precision, span_recall)

        # compute EL metrics
        precision = safe_divide(correct_predictions, total_predictions)
        recall = safe_divide(correct_predictions, total_gold)
        recall_at_k = safe_divide(
            (correct_predictions + correct_predictions_at_k), total_gold
        )

        f1 = f1_measure(precision, recall)

        wrong_for_candidates = safe_divide(miss_due_to_candidates, total_gold)

        out_dict = {
            "span_precision": span_precision,
            "span_recall": span_recall,
            "span_f1": span_f1,
            "core_precision": precision,
            "core_recall": recall,
            "core_recall-at-k": recall_at_k,
            "core_f1": round(f1, 4),
            "wrong-for-candidates": wrong_for_candidates,
            "index_errors_avg-index": safe_divide(
                sum(avg_wrong_predicted_index), len(avg_wrong_predicted_index)
            ),
            "index_correct_avg-index": safe_divide(
                sum(avg_correct_predicted_index), len(avg_correct_predicted_index)
            ),
            "index_avg-index": safe_divide(
                sum(avg_correct_predicted_index + avg_wrong_predicted_index),
                len(avg_correct_predicted_index + avg_wrong_predicted_index),
            ),
            "index_percentage-favoured-smaller-idx": safe_divide(
                sum(less_index_predictions), len(less_index_predictions)
            ),
        }

        return {k: round(v, 5) for k, v in out_dict.items()}


class ELStrongMatchingCallback(Callback):
    def __init__(self, dataset_path: str, dataset_conf) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_conf = dataset_conf
        self.strong_matching_metric = StrongMatching()

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        relik_reader_predictor = RelikReaderPredictor(pl_module.relik_reader_core_model)
        predicted_samples = relik_reader_predictor.predict(
            self.dataset_path,
            samples=None,
            dataset_conf=self.dataset_conf,
        )
        predicted_samples = list(predicted_samples)
        for k, v in self.strong_matching_metric(predicted_samples).items():
            pl_module.log(f"val_{k}", v)
