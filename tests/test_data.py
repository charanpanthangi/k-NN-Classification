import pandas as pd

from app.data import load_iris_dataset


def test_load_iris_dataset_shapes():
    X, y = load_iris_dataset()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (150, 4)
    assert y.shape == (150,)
