import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def decide(scores, threshold):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return scores >= float(threshold)


def candidate_metrics(y_has_match, scores, threshold):
    y = np.asarray(y_has_match, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != scores.shape or y.size < 1:
        raise ValueError("y_has_match and scores must be nonempty and aligned")
    accept = decide(scores, threshold)
    tp = int((y & accept).sum())
    fp = int((~y & accept).sum())
    tn = int((~y & ~accept).sum())
    fn = int((y & ~accept).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tnr": float(tnr),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": int(y.size),
        "n_open": int((~y).sum()),
        "n_closed": int(y.sum()),
    }


def ranking_metrics(y_has_match, scores):
    y = np.asarray(y_has_match, dtype=int).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != scores.shape or y.size < 1:
        raise ValueError("y_has_match and scores must be nonempty and aligned")
    if len(np.unique(y)) < 2:
        raise ValueError("PR/ROC require both match and open-set queries")
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def sweep_thresholds(y_has_match, scores, thresholds=None):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if thresholds is None:
        lo, hi = float(scores.min()), float(scores.max())
        thresholds = np.linspace(lo, hi, 65)
    rows = [candidate_metrics(y_has_match, scores, t) for t in np.asarray(thresholds, dtype=np.float64)]
    return rows


def select_threshold(y_has_match, scores, kind="max_f1", min_tnr=0.0):
    rows = sweep_thresholds(y_has_match, scores)
    if kind == "max_f1":
        feasible = [row for row in rows if row["tnr"] + 1e-12 >= min_tnr]
        pool = feasible if feasible else rows
        best = max(pool, key=lambda row: (row["f1"], row["tnr"], -row["threshold"]))
    elif kind == "youden":
        best = max(rows, key=lambda row: (row["recall"] + row["tnr"], row["f1"]))
    elif kind == "f1_tnr":
        best = max(rows, key=lambda row: (row["f1"] * row["tnr"], row["f1"]))
    else:
        raise ValueError(f"Unknown threshold rule {kind}")
    return best
