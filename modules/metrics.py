import numpy as np


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
    d = np.asarray(distance)
    qids, gids = np.asarray(qids), np.asarray(gids)
    if d.shape != (len(qids), len(gids)) or not np.isfinite(d).all():
        raise ValueError("Invalid retrieval distance matrix")
    aps, aps10, inps, hits = [], [], [], {k: [] for k in ranks}
    no_positive = 0
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
        if not relevant.any():
            no_positive += 1
            continue
        found = np.flatnonzero(relevant)
        aps.append(np.mean((np.arange(len(found)) + 1) / (found + 1)))
        hits10 = found[found < 10]
        cap = min(len(found), 10)
        if hits10.size == 0:
            aps10.append(0.0)
        else:
            aps10.append(float(np.sum((np.arange(len(hits10)) + 1) / (hits10 + 1)) / cap))
        inps.append(len(found) / (found[-1] + 1))
        for k in ranks:
            hits[k].append(float(relevant[:k].any()))
    if not aps:
        raise ValueError("No query has a valid gallery positive; cannot report mAP")
    return {
        "mAP": float(np.mean(aps)),
        "mAP@10": float(np.mean(aps10)),
        "mINP": float(np.mean(inps)),
        **{f"Rank-{k}": float(np.mean(v)) for k, v in hits.items()},
        "evaluated_queries": len(aps),
        "queries_without_positive": no_positive,
    }
