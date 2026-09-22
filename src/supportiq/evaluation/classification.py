"""Classification evaluation metrics: accuracy, macro/weighted F1, and per-class breakdown."""

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    """Calculate comprehensive classification metrics.

    Args:
        y_true: True ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing accuracy, macro F1, weighted F1, and per-label metrics.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    labels = sorted(set(y_true).union(set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            k: {
                "precision": round(v["precision"], 4),
                "recall": round(v["recall"], 4),
                "f1_score": round(v["f1-score"], 4),
                "support": v["support"],
            }
            for k, v in report.items()
            if isinstance(v, dict) and k not in ["macro avg", "weighted avg"]
        },
        "confusion_matrix": cm,
        "labels": labels,
    }
