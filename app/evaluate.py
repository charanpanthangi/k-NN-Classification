"""Evaluation utilities for k-NN classification."""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Compute common classification metrics and the confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    Tuple[Dict[str, float], np.ndarray]
        A dictionary of metrics (accuracy, precision, recall, f1) using macro
        averaging for multi-class problems, and the confusion matrix array.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm


__all__ = ["evaluate_classification"]
