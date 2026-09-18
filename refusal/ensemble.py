import numpy as np

from .metrics import select_threshold


def score_matrix(scores):
    parts = [np.asarray(channel, dtype=np.float64).reshape(-1) for channel in scores]
    if len(parts) < 2:
        raise ValueError("Need at least two score channels")
    n = len(parts[0])
    if n < 1 or any(len(channel) != n for channel in parts):
        raise ValueError("Score channels must be nonempty and aligned")
    return np.stack(parts, 1)


def rank_scores(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 1:
        raise ValueError("scores must be nonempty")
    if values.size == 1:
        return np.array([0.5], dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks


def _weights(n_channels, weights):
    if weights is None:
        return np.full(n_channels, 1.0 / n_channels, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != (n_channels,) or np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError("weights must be nonnegative and match channels")
    return w / w.sum()


def rank_average(scores, weights=None):
    matrix = score_matrix(scores)
    ranked = np.stack([rank_scores(matrix[:, j]) for j in range(matrix.shape[1])], 1)
    return ranked @ _weights(matrix.shape[1], weights)


def vote_fraction(scores, thresholds):
    matrix = score_matrix(scores)
    t = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if t.shape[0] != matrix.shape[1]:
        raise ValueError("thresholds must match channels")
    return (matrix >= t).mean(axis=1)


def fit_rank_weights(y, scores, top_correct, grid=4):
    matrix = score_matrix(scores)
    y = np.asarray(y, dtype=int).reshape(-1)
    top = np.asarray(top_correct, dtype=bool).reshape(-1)
    if y.shape[0] != matrix.shape[0] or top.shape[0] != matrix.shape[0]:
        raise ValueError("y must align with scores")
    if int(grid) < 2:
        raise ValueError("grid must be >= 2")
    axis = np.linspace(0.0, 1.0, int(grid))
    n = matrix.shape[1]
    best_key = None
    best_w = np.full(n, 1.0 / n)
    best_row = None
    idx = np.zeros(n, dtype=int)
    while True:
        w = axis[idx]
        if float(w.sum()) > 0:
            blended = rank_average([matrix[:, j] for j in range(n)], weights=w)
            picked = select_threshold(y, blended, top, kind="max_f1")
            key = (picked["f1"], picked["tnr"], -picked["threshold"])
            if best_key is None or key > best_key:
                best_key = key
                best_w = w / w.sum()
                best_row = picked
        pos = n - 1
        while pos >= 0:
            idx[pos] += 1
            if idx[pos] < len(axis):
                break
            idx[pos] = 0
            pos -= 1
        if pos < 0:
            break
    return best_w, best_row
