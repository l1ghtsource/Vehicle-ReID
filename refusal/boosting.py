from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier


def save_boosting(model, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(destination))
    return destination


def load_boosting(path):
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"CatBoost refusal model not found: {source}")
    model = CatBoostClassifier()
    model.load_model(str(source))
    return model


def fit_boosting(X, y, seed=0, iterations=200, depth=4, learning_rate=0.08):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int).reshape(-1)
    if X.ndim != 2 or len(X) != len(y) or len(X) < 4 or len(np.unique(y)) < 2:
        raise ValueError("Boosting needs a 2D feature matrix with both classes")
    model = CatBoostClassifier(
        iterations=int(iterations),
        depth=int(depth),
        learning_rate=float(learning_rate),
        loss_function="Logloss",
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )
    model.fit(X, y)
    return model


def predict_boosting(model, X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("Expected a 2D feature matrix")
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)
