from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATE_COLUMNS = ("query_id", "gallery_id", "confidence")
SKIP_GALLERY = frozenset({"", "nan", "none", "null", "na", "-"})


def _gallery_text(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if isinstance(value, str) or not hasattr(value, "item"):
        text = str(value).strip()
    else:
        text = str(value.item()).strip()
    if text.lower() in SKIP_GALLERY:
        return ""
    return text


def candidate_frame(query_ids, gallery_ids, confidences, accept=None):
    query_ids = np.asarray(query_ids).reshape(-1)
    gallery_ids = np.asarray(gallery_ids)
    confidences = np.asarray(confidences, dtype=np.float64)
    if gallery_ids.ndim != 2 or confidences.shape != gallery_ids.shape:
        raise ValueError("gallery_ids and confidences must be a matching (N, K) matrix")
    if len(query_ids) != len(gallery_ids):
        raise ValueError("query_ids must match gallery rows")
    if accept is None:
        keep = np.ones(len(query_ids), dtype=bool)
    else:
        keep = np.asarray(accept, dtype=bool).reshape(-1)
        if keep.shape != query_ids.shape:
            raise ValueError("accept must match query_ids")
    rows = []
    for i, qid in enumerate(query_ids):
        if not keep[i]:
            continue
        for j in range(gallery_ids.shape[1]):
            gid = _gallery_text(gallery_ids[i, j])
            conf = confidences[i, j]
            if not gid or not np.isfinite(conf):
                continue
            rows.append((str(qid), gid, float(conf)))
    frame = pd.DataFrame(rows, columns=pd.Index(list(CANDIDATE_COLUMNS)))
    if len(frame):
        frame = frame.sort_values(
            ["query_id", "confidence"],
            ascending=[True, False],
            kind="mergesort",
        ).reset_index(drop=True)
    return frame


def write_candidates(path, query_ids, gallery_ids, confidences, accept=None):
    frame = candidate_frame(query_ids, gallery_ids, confidences, accept=accept)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame
