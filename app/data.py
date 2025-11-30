"""Data loading utilities for the k-NN classification tutorial."""

from typing import Tuple

import pandas as pd
from sklearn import datasets


def load_iris_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the classic Iris dataset.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features ``X`` as a DataFrame and labels ``y`` as a Series.
    """
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")
    return X, y


__all__ = ["load_iris_dataset"]
