import numpy as np


def query_retrieval_rows(
    distance,
    qids,
    gids,
    qcams=None,
    gcams=None,
    ranks=(1, 5, 10),
    cross_camera=True,
    exclude_all_same_camera=False,
):
    d = np.asarray(distance)
    qids, gids = np.asarray(qids), np.asarray(gids)
    if d.shape != (len(qids), len(gids)) or not np.isfinite(d).all():
        raise ValueError("Invalid retrieval distance matrix")
    ranks = tuple(int(k) for k in ranks)
    rows = []
    for i, row in enumerate(d):
        keep = np.ones(len(gids), dtype=bool)
        if cross_camera:
            if (
                qcams is None
                or gcams is None
                or np.any(np.asarray(qcams) == -1)
                or np.any(np.asarray(gcams) == -1)
            ):
                raise ValueError("Cross-camera metrics require known camera IDs")
            same_cam = np.asarray(gcams) == np.asarray(qcams)[i]
            keep &= ~(same_cam if exclude_all_same_camera else (same_cam & (gids == qids[i])))
        order = np.argsort(row, kind="stable")
        order = order[keep[order]]
        relevant = gids[order] == qids[i]
        item = {"n_pos": int(relevant.sum()), "evaluated": bool(relevant.any())}
        for k in ranks:
            item[f"Rank-{k}"] = float(relevant[:k].any()) if item["evaluated"] else np.nan
        if not item["evaluated"]:
            item["ap"] = np.nan
            item["ap10"] = np.nan
            item["minp"] = np.nan
            rows.append(item)
            continue
        found = np.flatnonzero(relevant)
        item["ap"] = float(np.mean((np.arange(len(found)) + 1) / (found + 1)))
        hits10 = found[found < 10]
        cap = min(len(found), 10)
        if hits10.size == 0:
            item["ap10"] = 0.0
        else:
            item["ap10"] = float(np.sum((np.arange(len(hits10)) + 1) / (hits10 + 1)) / cap)
        item["minp"] = float(len(found) / (found[-1] + 1))
        rows.append(item)
    return rows


def retrieval_metrics(
    distance,
    qids,
    gids,
    qcams=None,
    gcams=None,
    ranks=(1, 5, 10),
    cross_camera=True,
    exclude_all_same_camera=False,
):
    ranks = tuple(int(k) for k in ranks)
    rows = query_retrieval_rows(
        distance,
        qids,
        gids,
        qcams=qcams,
        gcams=gcams,
        ranks=ranks,
        cross_camera=cross_camera,
        exclude_all_same_camera=exclude_all_same_camera,
    )
    evaluated = [row for row in rows if row["evaluated"]]
    if not evaluated:
        raise ValueError("No query has a valid gallery positive; cannot report mAP")
    return {
        "mAP": float(np.mean([row["ap"] for row in evaluated])),
        "mAP@10": float(np.mean([row["ap10"] for row in evaluated])),
        "mINP": float(np.mean([row["minp"] for row in evaluated])),
        **{f"Rank-{k}": float(np.mean([row[f"Rank-{k}"] for row in evaluated])) for k in ranks},
        "evaluated_queries": len(evaluated),
        "queries_without_positive": len(rows) - len(evaluated),
    }


def bootstrap_mean_ci(values, n_boot=1000, seed=0, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 1:
        raise ValueError("bootstrap requires at least one finite value")
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1.0 - alpha / 2])
    return {
        "n": int(values.size),
        "point": float(values.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": int(n_boot),
    }
