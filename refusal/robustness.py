import numpy as np

from .metrics import candidate_metrics, decision_metrics, select_threshold
from .protocol import OPEN_SET_FRACTION, open_set_pack, open_set_split, scored_pack

HOLD_SEEDS = tuple(range(8))
TEST_GALLERY_SIZE = 750


def subsample_gallery(qids, gids, n_keep, seed=0, require_positive=True):
    gids = np.asarray(gids)
    qids = np.asarray(qids)
    if n_keep < 1:
        raise ValueError("n_keep must be >= 1")
    if len(gids) < 1:
        raise ValueError("gallery is empty")
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(gids), dtype=bool)
    if require_positive:
        for qid in np.unique(qids):
            idx = np.flatnonzero(gids == qid)
            if idx.size:
                keep[rng.choice(idx)] = True
    remaining = np.flatnonzero(~keep)
    need = min(max(int(n_keep) - int(keep.sum()), 0), remaining.size)
    if need:
        keep[rng.choice(remaining, size=need, replace=False)] = True
    return keep


def open_set_sized_pack(
    query,
    gallery,
    qids,
    gids,
    n_keep=TEST_GALLERY_SIZE,
    k=10,
    with_embeddings=False,
    fraction=OPEN_SET_FRACTION,
    seed=0,
):
    _, gallery_keep, held = open_set_split(qids, gids, fraction=fraction, seed=seed)
    gallery = np.asarray(gallery)[gallery_keep]
    gids = np.asarray(gids)[gallery_keep]
    keep = subsample_gallery(qids, gids, n_keep, seed=seed)
    packed = scored_pack(
        query,
        gallery[keep],
        qids,
        gids[keep],
        k=k,
        with_embeddings=with_embeddings,
    )
    return (*packed, np.asarray(held), keep)


def cosine_holdout_sweep(
    query,
    gallery,
    qids,
    gids,
    seeds=HOLD_SEEDS,
    frozen_threshold=None,
    k=10,
    fraction=OPEN_SET_FRACTION,
):
    rows = []
    for seed in seeds:
        _features, y, cosine, hits, held = open_set_pack(
            query,
            gallery,
            qids,
            gids,
            k=k,
            with_embeddings=False,
            fraction=fraction,
            seed=seed,
        )
        fitted = select_threshold(y, cosine, hits, kind="contest")
        row = {
            "seed": int(seed),
            "n_held": int(len(held)),
            "n": int(len(y)),
            "n_open": int((~np.asarray(y, dtype=bool)).sum()),
            "fit_threshold": float(fitted["threshold"]),
            "fit_f1": float(fitted["f1"]),
            "fit_tnr": float(fitted["tnr"]),
            "fit_contest": float(fitted["contest"]),
        }
        if frozen_threshold is not None:
            frozen = candidate_metrics(y, cosine, frozen_threshold, hits)
            row["frozen_f1"] = float(frozen["f1"])
            row["frozen_tnr"] = float(frozen["tnr"])
            row["frozen_contest"] = float(frozen["contest"])
        rows.append(row)
    return rows


def summarize_sweep(rows, keys):
    out = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return out


def bootstrap_accept_ci(y, accept, top, n_boot=1000, seed=0, alpha=0.05, keys=("f1", "tnr", "contest")):
    y = np.asarray(y, dtype=bool).reshape(-1)
    accept = np.asarray(accept, dtype=bool).reshape(-1)
    top = np.asarray(top, dtype=bool).reshape(-1)
    if y.size < 1 or y.shape != accept.shape or y.shape != top.shape:
        raise ValueError("y, accept, and top must be nonempty and aligned")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rng = np.random.default_rng(seed)
    store = {key: [] for key in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, y.size, y.size)
        row = decision_metrics(y[idx], accept[idx], top[idx])
        for key in keys:
            store[key].append(row[key])
    point = decision_metrics(y, accept, top)
    out = {"n": int(y.size), "n_boot": int(n_boot)}
    for key in keys:
        arr = np.asarray(store[key], dtype=np.float64)
        lo, hi = np.quantile(arr, [alpha / 2, 1.0 - alpha / 2])
        out[key] = {"point": float(point[key]), "lo": float(lo), "hi": float(hi)}
    return out


def paired_accept_delta_ci(y, top, accept_a, accept_b, n_boot=1000, seed=0, alpha=0.05):
    y = np.asarray(y, dtype=bool).reshape(-1)
    top = np.asarray(top, dtype=bool).reshape(-1)
    accept_a = np.asarray(accept_a, dtype=bool).reshape(-1)
    accept_b = np.asarray(accept_b, dtype=bool).reshape(-1)
    if y.size < 1 or len({y.shape, top.shape, accept_a.shape, accept_b.shape}) != 1:
        raise ValueError("paired inputs must be nonempty and aligned")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, y.size, y.size)
        left = decision_metrics(y[idx], accept_a[idx], top[idx])["contest"]
        right = decision_metrics(y[idx], accept_b[idx], top[idx])["contest"]
        deltas.append(left - right)
    point = decision_metrics(y, accept_a, top)["contest"] - decision_metrics(y, accept_b, top)["contest"]
    arr = np.asarray(deltas, dtype=np.float64)
    lo, hi = np.quantile(arr, [alpha / 2, 1.0 - alpha / 2])
    return {
        "point": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "includes_zero": bool(lo <= 0.0 <= hi),
        "n": int(y.size),
        "n_boot": int(n_boot),
    }


def bootstrap_decision_ci(
    y, scores, threshold, top, n_boot=1000, seed=0, alpha=0.05, keys=("f1", "tnr", "contest")
):
    y = np.asarray(y, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    top = np.asarray(top, dtype=bool).reshape(-1)
    if y.size < 1 or y.shape != scores.shape or y.shape != top.shape:
        raise ValueError("y, scores, and top must be nonempty and aligned")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rng = np.random.default_rng(seed)
    store = {key: [] for key in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, y.size, y.size)
        row = candidate_metrics(y[idx], scores[idx], threshold, top[idx])
        for key in keys:
            store[key].append(row[key])
    point = candidate_metrics(y, scores, threshold, top)
    out = {"n": int(y.size), "n_boot": int(n_boot)}
    for key in keys:
        arr = np.asarray(store[key], dtype=np.float64)
        lo, hi = np.quantile(arr, [alpha / 2, 1.0 - alpha / 2])
        out[key] = {"point": float(point[key]), "lo": float(lo), "hi": float(hi)}
    return out


def paired_contest_delta_ci(
    y,
    top,
    scores_a,
    scores_b,
    threshold_a,
    threshold_b,
    n_boot=1000,
    seed=0,
    alpha=0.05,
):
    y = np.asarray(y, dtype=bool).reshape(-1)
    top = np.asarray(top, dtype=bool).reshape(-1)
    scores_a = np.asarray(scores_a, dtype=np.float64).reshape(-1)
    scores_b = np.asarray(scores_b, dtype=np.float64).reshape(-1)
    if y.size < 1 or len({y.shape, top.shape, scores_a.shape, scores_b.shape}) != 1:
        raise ValueError("paired inputs must be nonempty and aligned")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, y.size, y.size)
        left = candidate_metrics(y[idx], scores_a[idx], threshold_a, top[idx])["contest"]
        right = candidate_metrics(y[idx], scores_b[idx], threshold_b, top[idx])["contest"]
        deltas.append(left - right)
    point = (
        candidate_metrics(y, scores_a, threshold_a, top)["contest"]
        - candidate_metrics(y, scores_b, threshold_b, top)["contest"]
    )
    arr = np.asarray(deltas, dtype=np.float64)
    lo, hi = np.quantile(arr, [alpha / 2, 1.0 - alpha / 2])
    return {
        "point": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "includes_zero": bool(lo <= 0.0 <= hi),
        "n": int(y.size),
        "n_boot": int(n_boot),
    }


def percentile_mask(values, q=25, side="low"):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 1:
        raise ValueError("values must be nonempty")
    if side not in {"low", "high"}:
        raise ValueError("side must be low or high")
    cut = float(np.nanpercentile(values, q))
    finite = np.isfinite(values)
    if side == "low":
        return finite & (values <= cut)
    return finite & (values >= cut)


def identity_n_cameras(vehicle_ids, camera_ids):
    counts = {}
    for vid, cam in zip(np.asarray(vehicle_ids), np.asarray(camera_ids), strict=True):
        counts.setdefault(int(vid), set()).add(int(cam))
    return {vid: len(cams) for vid, cams in counts.items()}


def multi_camera_mask(query_ids, camera_counts, min_cameras=3):
    return np.array([camera_counts.get(int(qid), 0) >= min_cameras for qid in query_ids], dtype=bool)


def impostor_max(sim, qids, gids):
    sim = np.asarray(sim, dtype=np.float64)
    qids = np.asarray(qids)
    gids = np.asarray(gids)
    if sim.shape != (len(qids), len(gids)):
        raise ValueError("similarity shape must match query and gallery identities")
    other = gids[None, :] != qids[:, None]
    return np.where(other, sim, -np.inf).max(axis=1)


def lookalike_mask(sim, qids, gids, min_cosine=0.55):
    return impostor_max(sim, qids, gids) >= float(min_cosine)


def aspect_shift_mask(query_w, query_h, gallery_w, gallery_h, qids, gids, ratio=1.5):
    q_aspect = np.asarray(query_w, dtype=np.float64) / np.clip(
        np.asarray(query_h, dtype=np.float64), 1e-6, None
    )
    g_aspect = np.asarray(gallery_w, dtype=np.float64) / np.clip(
        np.asarray(gallery_h, dtype=np.float64), 1e-6, None
    )
    gids = np.asarray(gids)
    flags = []
    for aspect, qid in zip(q_aspect, np.asarray(qids), strict=True):
        same = gids == qid
        if not same.any():
            flags.append(False)
            continue
        med = float(np.median(g_aspect[same]))
        flags.append(bool(med > 0 and max(aspect / med, med / max(aspect, 1e-6)) >= ratio))
    return np.asarray(flags, dtype=bool)


def slice_decision(y, scores, threshold, top, mask, min_n=8):
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    y = np.asarray(y)
    if mask.shape != y.shape:
        raise ValueError("mask must match queries")
    if int(mask.sum()) < int(min_n):
        return None
    return candidate_metrics(y[mask], scores[mask], threshold, top[mask])
