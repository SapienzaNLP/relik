import argparse
from relik.reader.data.relik_reader_sample import load_relik_reader_samples
from relik.reader.pytorch_modules.triplet import RelikReaderForTripletExtraction
from relik.reader.utils.relation_matching_eval import StrongMatching
from relik.inference.data.objects import AnnotationType

import torch

import numpy as np
from sklearn.metrics import precision_recall_curve


def find_optimal_threshold(scores, labels):
    # Calculate precision-recall pairs for various threshold values
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # Add the end point for thresholds, which is the maximum score + 1 to ensure completeness
    thresholds = np.append(thresholds, thresholds[-1] + 1)
    # Calculate F1 scores from precision and recall for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    # Handle the case where precision + recall equals zero (to avoid division by zero)
    f1_scores = np.nan_to_num(f1_scores)
    # Find the index of the maximum F1 score
    max_index = np.argmax(f1_scores)
    # Find the threshold and F1 score corresponding to the maximum F1 score
    optimal_threshold = thresholds[max_index]
    best_f1 = f1_scores[max_index]

    return optimal_threshold, best_f1


def eval(
    model_path,
    data_path,
    is_eval,
    output_path=None,
    compute_threshold=False,
    save_threshold=False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device
    print(f"Device: {device}")
    reader = RelikReaderForTripletExtraction(model_path, training=False, device=device)
    samples = list(load_relik_reader_samples(data_path))
    optimal_threshold = None
    if compute_threshold:
        predicted_samples = reader.read(
            samples=samples,
            progress_bar=True,
            annotation_type=AnnotationType.WORD,
            return_threshold_utils=True,
        )
        re_probabilities, re_labels = [], []
        for sample in predicted_samples:
            re_probabilities.extend(sample.re_probabilities.flatten())
            re_labels.extend(sample.re_labels.flatten())
        optimal_threshold, best_f1 = find_optimal_threshold(re_probabilities, re_labels)
        print(f"Optimal threshold: {optimal_threshold}")
        print(f"Best F1: {best_f1}")
        # set the threshold to the optimal threshold
        samples = list(load_relik_reader_samples(data_path))
        if save_threshold:
            reader.relik_reader_model.config.threshold = optimal_threshold
            reader.relik_reader_model.save_pretrained(model_path)

    predicted_samples = reader.read(
        samples=samples,
        progress_bar=True,
        annotation_type=AnnotationType.WORD,
        relation_threshold=optimal_threshold,
    )

    if is_eval:
        strong_matching_metric = StrongMatching()
        predicted_samples = list(predicted_samples)
        for k, v in strong_matching_metric(predicted_samples).items():
            print(f"test_{k}", v)
    if output_path is not None:
        with open(output_path, "w") as f:
            for sample in predicted_samples:
                f.write(sample.to_jsons() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
    )
    parser.add_argument("--is-eval", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--compute-threshold", action="store_true")
    parser.add_argument("--save-threshold", action="store_true")
    args = parser.parse_args()
    eval(
        args.model_path,
        args.data_path,
        args.is_eval,
        args.output_path,
        args.compute_threshold,
        args.save_threshold,
    )


if __name__ == "__main__":
    main()
