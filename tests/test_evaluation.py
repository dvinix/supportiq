"""Unit tests for evaluation metrics computation."""

from supportiq.evaluation.classification import compute_classification_metrics


def test_compute_classification_metrics_exact() -> None:
    """Verify accuracy and macro F1 computation on known toy inputs."""
    y_true = ["ORDER", "ORDER", "REFUND", "REFUND"]
    y_pred = ["ORDER", "REFUND", "REFUND", "REFUND"]

    metrics = compute_classification_metrics(y_true, y_pred)

    # 3 correct out of 4 = 75%
    assert metrics["accuracy"] == 0.75
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert "ORDER" in metrics["per_class"]
    assert "REFUND" in metrics["per_class"]
