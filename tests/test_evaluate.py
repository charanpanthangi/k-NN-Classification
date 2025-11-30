import numpy as np

from app.evaluate import evaluate_classification


def test_evaluate_classification_metrics_range():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 0, 0])

    metrics, cm = evaluate_classification(y_true, y_pred)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0

    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_true)
