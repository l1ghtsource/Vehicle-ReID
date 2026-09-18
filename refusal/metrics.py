import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

CONTEST_F1_WEIGHT = 0.7
CONTEST_TNR_WEIGHT = 0.3


def contest_score(f1, tnr):
    return CONTEST_F1_WEIGHT * float(f1) + CONTEST_TNR_WEIGHT * float(tnr)


def decide(scores, threshold):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    return scores >= float(threshold)


def _aligned(y_has_match, scores, top_correct):
    y = np.asarray(y_has_match, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    top = np.asarray(top_correct, dtype=bool).reshape(-1)
    if y.size < 1 or y.shape != scores.shape or y.shape != top.shape:
        raise ValueError("y_has_match, scores, and top_correct must be nonempty and aligned")
    return y, scores, top & y


def _confusion(y, accept, top_ok):
    tp = int((accept & top_ok).sum())
    fp = int((accept & ~top_ok).sum())
    tn = int((~accept & ~y).sum())
    fn = int((~accept & y).sum())
    fp_open = int((accept & ~y).sum())
    fp_closed = int((accept & y & ~top_ok).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp_open, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tnr": float(tnr),
        "contest": contest_score(f1, tnr),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fp_open": fp_open,
        "fp_closed": fp_closed,
        "n": int(y.size),
        "n_open": int((~y).sum()),
        "n_closed": int(y.sum()),
        "n_top_hit": int(top_ok.sum()),
    }


def decision_metrics(y_has_match, accept, top_correct):
    y = np.asarray(y_has_match, dtype=bool).reshape(-1)
    accept = np.asarray(accept, dtype=bool).reshape(-1)
    top = np.asarray(top_correct, dtype=bool).reshape(-1)
    if y.size < 1 or y.shape != accept.shape or y.shape != top.shape:
        raise ValueError("y_has_match, accept, and top_correct must be nonempty and aligned")
    return _confusion(y, accept, top & y)


def candidate_metrics(y_has_match, scores, threshold, top_correct):
    y, scores, top_ok = _aligned(y_has_match, scores, top_correct)
    row = _confusion(y, decide(scores, threshold), top_ok)
    row["threshold"] = float(threshold)
    return row


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


def sweep_thresholds(y_has_match, scores, top_correct, thresholds=None):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if thresholds is None:
        uniq = np.unique(scores)
        if uniq.size < 1:
            raise ValueError("y_has_match, scores, and top_correct must be nonempty and aligned")
        thresholds = np.concatenate([uniq, uniq[-1:] + 1.0])
    values = np.asarray(thresholds, dtype=np.float64)
    return [candidate_metrics(y_has_match, scores, t, top_correct) for t in values]


def _rule_key(kind):
    if kind == "max_f1":
        return lambda row: (row["f1"], row["tnr"], -row["threshold"])
    if kind == "youden":
        return lambda row: (row["recall"] + row["tnr"], row["f1"], -row["threshold"])
    if kind == "f1_tnr":
        return lambda row: (row["f1"] * row["tnr"], row["f1"], -row["threshold"])
    if kind == "contest":
        return lambda row: (row["contest"], row["f1"], row["tnr"], -row["threshold"])
    raise ValueError(f"Unknown threshold rule {kind}")


def select_threshold(y_has_match, scores, top_correct, kind="contest", min_tnr=0.0):
    rows = sweep_thresholds(y_has_match, scores, top_correct)
    feasible = [row for row in rows if row["tnr"] + 1e-12 >= min_tnr]
    pool = feasible if feasible else rows
    return max(pool, key=_rule_key(kind))
