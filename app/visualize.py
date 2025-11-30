"""Visualization utilities for k-NN classification."""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], filename: str) -> None:
    """
    Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    class_names : Sequence[str]
        Class labels for axis ticks.
    filename : str
        Path to save the SVG figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    filename: str,
    h: float = 0.02,
) -> None:
    """
    Plot a 2D decision boundary using PCA for visualization.

    The model is re-fit on the PCA-transformed data to enable plotting in two
    dimensions while respecting scaling. This visualization is an approximation
    of the decision surface.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    pipeline : Pipeline
        Trained pipeline containing ``StandardScaler`` and ``KNeighborsClassifier``.
    filename : str
        Path to save the SVG figure.
    h : float, optional
        Step size in the mesh grid, by default 0.02.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    knn_params = pipeline.named_steps.get("knn", KNeighborsClassifier()).get_params()
    knn_2d = KNeighborsClassifier(**knn_params)
    knn_2d.fit(X_pca, y)

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = knn_2d.predict(grid)
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.get_cmap("Set1", len(np.unique(y)))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap, edgecolor="k", s=40)
    ax.set_title("k-NN Decision Boundary (PCA 2D Projection)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes", loc="upper right")
    ax.add_artist(legend1)

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, format="svg", bbox_inches="tight")
    plt.close(fig)


__all__ = ["plot_confusion_matrix", "plot_decision_boundary"]
