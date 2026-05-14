import csv
import os


def compute_open_set_metrics(y_true, y_pred, known_labels):
    """
    Compute classifier metrics when test labels may include unseen speakers.

    Classifiers in this project always emit one of the trained speaker labels.
    Test labels outside known_labels are therefore treated as impostor attempts:
    a classifier prediction for them is a false accept unless a future thresholded
    model explicitly rejects them.
    """
    known_labels = set(known_labels)

    total = len(y_true)
    known_total = 0
    unknown_total = 0
    correct_accepts = 0
    correct_rejects = 0
    false_accepts = 0
    false_rejects = 0
    misidentified_known = 0

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in known_labels:
            known_total += 1
            if pred_label == true_label:
                correct_accepts += 1
            elif pred_label is None:
                false_rejects += 1
            else:
                misidentified_known += 1
        else:
            unknown_total += 1
            if pred_label is None:
                correct_rejects += 1
            else:
                false_accepts += 1

    correct = correct_accepts + correct_rejects
    incorrect = misidentified_known + false_accepts
    notmatched = false_rejects

    accuracy = correct / total if total > 0 else 0.0
    fnmr = false_rejects / known_total if known_total > 0 else 0.0

    if unknown_total > 0:
        fmr = false_accepts / unknown_total
    else:
        fmr = incorrect / total if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "notmatched": notmatched,
        "accuracy": accuracy,
        "fnmr": fnmr,
        "fmr": fmr,
        "known_total": known_total,
        "unknown_total": unknown_total,
        "correct_accepts": correct_accepts,
        "correct_rejects": correct_rejects,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "misidentified_known": misidentified_known,
        "open_set": unknown_total > 0,
    }


def compute_metrics(correct, incorrect, not_matched):
    total = correct + incorrect + not_matched
    return {
        "correct":     correct,
        "incorrect":   incorrect,
        "not_matched": not_matched,
        "accuracy":    round(correct / total, 4) if total else 0.0,
        "fnmr":        round(not_matched / total, 4) if total else 0.0,
        "fmr":         round(incorrect / total, 4) if total else 0.0,
    }


def print_metrics(m):
    print(f"Correct:    {m.get('correct', '-')}")
    print(f"Incorrect:  {m.get('incorrect', '-')}")
    print(f"No match:   {m.get('not_matched', '-')}")
    print(f"Accuracy:   {m.get('accuracy', 0):.2%}")
    print(f"FNMR:       {m.get('fnmr', '-')}")
    print(f"FMR:        {m.get('fmr', '-')}")


def save_metrics_csv(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"Results saved to {path}")
