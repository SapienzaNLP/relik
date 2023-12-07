def safe_divide(num: float, den: float) -> float:
    if den == 0:
        return 0
    else:
        return num / den


def f1_measure(precision: float, recall: float) -> float:
    if precision == 0 or recall == 0:
        return 0.0
    return safe_divide(2 * precision * recall, (precision + recall))


def compute_metrics(total_correct, total_preds, total_gold):
    precision = safe_divide(total_correct, total_preds)
    recall = safe_divide(total_correct, total_gold)
    f1 = f1_measure(precision, recall)
    return precision, recall, f1
