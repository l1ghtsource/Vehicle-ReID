import numpy as np
from scipy.stats import kendalltau


def ranked_indices(similarity, keep=None):
    sim = np.asarray(similarity, dtype=np.float32).reshape(-1)
    if keep is None:
        mask = np.ones(len(sim), dtype=bool)
    else:
        mask = np.asarray(keep, dtype=bool)
        if mask.shape != sim.shape:
            raise ValueError("keep mask must match similarity")
    if not mask.any():
        raise ValueError("keep mask is empty")
    order = np.argsort(-sim, kind="stable")
    return order[mask[order]]


def topk(similarity, k, keep=None):
    if k < 1:
        raise ValueError("k must be >= 1")
    order = ranked_indices(similarity, keep)
    return order[: min(int(k), len(order))]


def average_precision(relevant):
    relevant = np.asarray(relevant, dtype=bool)
    if not relevant.any():
        return 0.0
    found = np.flatnonzero(relevant)
    return float(np.mean((np.arange(len(found)) + 1) / (found + 1)))


def kendall_top(orig_order, corr_order, k):
    k_eff = min(int(k), len(orig_order), len(corr_order))
    if k_eff < 2:
        return 1.0
    universe = np.unique(np.concatenate([orig_order[:k_eff], corr_order[:k_eff]]))
    if len(universe) < 2:
        return 1.0
    orig_rank = {int(idx): rank for rank, idx in enumerate(orig_order)}
    corr_rank = {int(idx): rank for rank, idx in enumerate(corr_order)}
    tau, _ = kendalltau([orig_rank[int(i)] for i in universe], [corr_rank[int(i)] for i in universe])
    return 1.0 if not np.isfinite(tau) else float(tau)


def overlap_curve(orig_order, corr_order, ks=(1, 5, 10, 20)):
    n = min(len(orig_order), len(corr_order))
    if n < 1:
        raise ValueError("rankings are empty")
    out = {}
    for k in ks:
        k_eff = min(int(k), n)
        if k_eff < 1:
            raise ValueError("k must be >= 1")
        inter = np.intersect1d(orig_order[:k_eff], corr_order[:k_eff]).size
        out[int(k)] = inter / k_eff
    return out


def retrieval_stability(orig_sim, corr_sim, k=10, keep=None, qid=None, gids=None):
    if k < 1:
        raise ValueError("k must be >= 1")
    orig_order = ranked_indices(orig_sim, keep)
    corr_order = ranked_indices(corr_sim, keep)
    k_eff = min(int(k), len(orig_order), len(corr_order))
    orig_top = orig_order[:k_eff]
    corr_top = corr_order[:k_eff]
    inter = np.intersect1d(orig_top, corr_top).size
    union = np.union1d(orig_top, corr_top).size
    orig_r1 = int(orig_order[0])
    orig_r1_rank = int(np.flatnonzero(corr_order == orig_r1)[0]) + 1
    out = {
        "k": k_eff,
        "overlap": inter / k_eff,
        "jaccard": inter / max(union, 1),
        "kendall": kendall_top(orig_order, corr_order, k_eff),
        "rank1_image_same": bool(int(corr_order[0]) == orig_r1),
        "orig_r1_rank": orig_r1_rank,
    }
    if qid is None or gids is None:
        return out
    labels = np.asarray(gids)
    orig_rel = labels[orig_order] == qid
    corr_rel = labels[corr_order] == qid
    out["has_positive"] = bool(orig_rel.any())
    out["ap_orig"] = average_precision(orig_rel)
    out["ap_corr"] = average_precision(corr_rel)
    out["delta_ap"] = out["ap_corr"] - out["ap_orig"]
    out["rank1_hit_orig"] = bool(orig_rel[0])
    out["rank1_hit_corr"] = bool(corr_rel[0])
    out["rank1_id_same"] = bool(labels[int(corr_order[0])] == labels[orig_r1])
    if orig_rel.any() and corr_rel.any():
        out["mean_pos_rank_orig"] = float(np.flatnonzero(orig_rel).mean() + 1)
        out["mean_pos_rank_corr"] = float(np.flatnonzero(corr_rel).mean() + 1)
        out["delta_mean_pos_rank"] = out["mean_pos_rank_corr"] - out["mean_pos_rank_orig"]
    else:
        out["mean_pos_rank_orig"] = float("nan")
        out["mean_pos_rank_corr"] = float("nan")
        out["delta_mean_pos_rank"] = float("nan")
    return out
