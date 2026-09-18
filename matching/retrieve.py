import numpy as np

STAT_KINDS = ("n_matches", "n_inliers", "score_sum", "score_mean")


def minmax(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < 1:
        raise ValueError("values must be nonempty")
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def hybrid_head(order, cosine, match, k, weight=0.5):
    order = np.asarray(order)
    cosine = np.asarray(cosine, dtype=np.float64).reshape(-1)
    match = np.asarray(match, dtype=np.float64).reshape(-1)
    if cosine.shape != match.shape:
        raise ValueError("cosine and match scores must align")
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    if not 0 <= float(weight) <= 1:
        raise ValueError("weight must be in [0, 1]")
    k_eff = min(int(k), len(order))
    head = order[:k_eff]
    if k_eff < 1:
        return np.zeros(0, dtype=np.float64)
    mixed = (1.0 - float(weight)) * minmax(cosine[head]) + float(weight) * minmax(match[head])
    return mixed


def reorder_head(order, gallery_scores, k):
    order = np.asarray(order)
    scores = np.asarray(gallery_scores, dtype=np.float64).reshape(-1)
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    k_eff = min(int(k), len(order))
    head = order[:k_eff]
    if int(head.max(initial=-1)) >= len(scores) or int(head.min(initial=0)) < 0:
        raise ValueError("gallery scores must cover the ranking head")
    reranked = head[np.argsort(-scores[head], kind="stable")]
    return np.concatenate([reranked, order[k_eff:]])


def ranking_row(order, qid, gids, ranks=(1, 5, 10)):
    order = np.asarray(order)
    labels = np.asarray(gids)
    relevant = labels[order] == qid
    if not relevant.any():
        raise ValueError("no gallery positive in the ranking")
    found = np.flatnonzero(relevant)
    row = {
        "ap": float(np.mean((np.arange(len(found)) + 1) / (found + 1))),
        "rank1": bool(relevant[0]),
        "pos_rank": int(found[0]) + 1,
    }
    for rank in ranks:
        if int(rank) < 1:
            raise ValueError("ranks must be >= 1")
        row[f"r{int(rank)}"] = bool(relevant[: int(rank)].any())
    return row


def summarize(rows):
    if len(rows) < 1:
        raise ValueError("no ranking rows")
    keys = [key for key, value in rows[0].items() if isinstance(value, (bool, int, float, np.generic))]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def evidence_vector(stats, kind):
    if kind not in STAT_KINDS:
        raise ValueError(f"unknown match statistic {kind}")
    return np.asarray([row[kind] for row in stats], dtype=np.float64)
