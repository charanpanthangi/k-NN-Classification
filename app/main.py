"""Run an end-to-end k-NN classification pipeline on the Iris dataset."""

from pathlib import Path

from app.data import load_iris_dataset
from app.evaluate import evaluate_classification
from app.model import create_knn_pipeline
from app.preprocess import split_data
from app.visualize import plot_confusion_matrix, plot_decision_boundary


OUTPUT_DIR = Path("outputs")


def main() -> None:
    # 1. Load data
    X, y = load_iris_dataset()

    # 2. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Build model (includes scaling inside the pipeline)
    pipeline = create_knn_pipeline()

    # 4. Train
    pipeline.fit(X_train, y_train)

    # 5. Predict
    y_pred = pipeline.predict(X_test)

    # 6. Evaluate
    metrics, cm = evaluate_classification(y_test.to_numpy(), y_pred)

    # 7. Visualize
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    plot_confusion_matrix(cm, class_names=["setosa", "versicolor", "virginica"], filename=OUTPUT_DIR / "confusion_matrix.svg")
    plot_decision_boundary(X.to_numpy(), y.to_numpy(), pipeline, filename=OUTPUT_DIR / "decision_boundary.svg")

    # 8. Display metrics
    print("Evaluation metrics (macro-averaged):")
    for name, value in metrics.items():
        print(f"- {name}: {value:.3f}")


if __name__ == "__main__":
    main()
