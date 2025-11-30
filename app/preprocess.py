"""Data preprocessing utilities for k-NN classification."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# k-NN is a distance-based algorithm, so scaling is essential to
# prevent features with larger ranges from dominating distance calculations.


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.

    The actual scaling is performed inside the model pipeline via
    :class:`sklearn.preprocessing.StandardScaler`, ensuring no data leakage.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    test_size : float, optional
        Fraction of data to allocate to the test set, by default 0.2.
    random_state : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        ``X_train, X_test, y_train, y_test``.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


__all__ = ["split_data"]
