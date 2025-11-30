"""Model creation utilities for k-NN classification."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# We use a Pipeline to ensure that scaling happens inside the cross-validation
# or training workflow, preventing data leakage from the test set.

def create_knn_pipeline(
    n_neighbors: int = 5, weights: str = "distance", metric: str = "euclidean"
) -> Pipeline:
    """
    Build a k-NN classification pipeline with scaling.

    Parameters
    ----------
    n_neighbors : int, optional
        Number of neighbors to use, by default 5.
    weights : str, optional
        Weight function used in prediction (`"uniform"` or `"distance"`),
        by default "distance".
    metric : str, optional
        Distance metric to use, by default "euclidean".

    Returns
    -------
    Pipeline
        A scikit-learn pipeline with ``StandardScaler`` and ``KNeighborsClassifier``.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors, weights=weights, metric=metric
                ),
            ),
        ]
    )


__all__ = ["create_knn_pipeline"]
