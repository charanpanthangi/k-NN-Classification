from app.data import load_iris_dataset
from app.model import create_knn_pipeline
from app.preprocess import split_data


def test_knn_pipeline_trains_and_predicts():
    X, y = load_iris_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = create_knn_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    assert len(predictions) == len(y_test)
    assert predictions.min() >= 0
    assert predictions.max() <= 2
