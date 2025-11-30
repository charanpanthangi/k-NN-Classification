# k-Nearest Neighbors (k-NN) Classification Tutorial & Template

A beginner-friendly, end-to-end example of training, evaluating, and visualizing a k-Nearest Neighbors (k-NN) classifier on the classic Iris dataset using scikit-learn. The repository is organized for clarity and modularity so you can reuse pieces in your own projects.

## What is k-NN Classification?

k-NN is a distance-based, non-parametric algorithm that classifies a new sample by looking at the **k** closest labeled samples (neighbors) in the feature space.

- **Distance**: Typically Euclidean for continuous features. Points that are closer influence the prediction more than distant points.
- **Prediction rule**: Majority vote of neighbors (optionally weighted by inverse distance).
- **Model simplicity**: No explicit training phase beyond storing the data; most work happens at prediction time.

### Why scaling matters

k-NN relies on distances. If one feature has a larger numeric range than others, it can dominate the distance calculation. Standardizing features (zero mean, unit variance) keeps all dimensions comparable. This repo **always scales** features with `StandardScaler`.

### Key hyperparameters

- `n_neighbors`: How many neighbors participate in the vote (default here: 5).
- `weights`: `"uniform"` treats all neighbors equally; `"distance"` gives closer points more influence (default here: `"distance"`).
- `metric`: The distance measure, e.g., `"euclidean"` (used here) or `"manhattan"`.

### When k-NN works well

- Small to medium-sized datasets.
- Low-dimensional, well-scaled numeric features.
- Decision boundaries that can be captured by local neighborhoods.

### When k-NN struggles

- High-dimensional spaces (curse of dimensionality).
- Very large datasets (prediction becomes slow because distances are computed to many points).
- Features on very different scales (unless properly standardized).

## Dataset

The [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) contains 150 flower samples across three species (`setosa`, `versicolor`, `virginica`) with four numeric features (sepal length/width, petal length/width).

## Project Structure

```
app/
├── data.py          # Load Iris dataset
├── preprocess.py    # Train/test split + scaling
├── model.py         # k-NN model & pipeline
├── evaluate.py      # Metrics computation
├── visualize.py     # Confusion matrix & decision boundaries
├── main.py          # End-to-end script
notebooks/
└── demo_knn_classification.ipynb
examples/
└── README_examples.md
requirements.txt
Dockerfile
LICENSE
README.md
```

## Getting Started

### Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/main.py
```

### Jupyter notebook

```bash
jupyter notebook notebooks/demo_knn_classification.ipynb
```

### Running tests

```bash
pytest
```

## Future improvements

- Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`.
- Experiment with different distance metrics (e.g., `manhattan`, `minkowski`).
- Explore weighted k-NN variations or feature selection.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
