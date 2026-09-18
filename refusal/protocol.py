import numpy as np

from .features import example_vector
from .threshold import max_cosine


def open_set_split(qids, gids, fraction=0.2, seed=0):
    qids = np.asarray(qids)
    gids = np.asarray(gids)
    if not 0 < fraction < 1:
        raise ValueError("open-set fraction must be in (0, 1)")
    identities = np.unique(qids)
    if len(identities) < 2:
        raise ValueError("Need at least two query identities for an open-set split")
    n_hold = int(np.clip(round(fraction * len(identities)), 1, len(identities) - 1))
    rng = np.random.default_rng(seed)
    held = rng.choice(identities, n_hold, replace=False)
    query_open = np.isin(qids, held)
    gallery_keep = ~np.isin(gids, held)
    if not gallery_keep.any():
        raise ValueError("Open-set split removed the entire gallery")
    return query_open, gallery_keep, np.asarray(held)


def has_match(qids, gids, gallery_keep=None):
    gids = np.asarray(gids)
    keep = np.ones(len(gids), dtype=bool) if gallery_keep is None else np.asarray(gallery_keep, dtype=bool)
    if keep.shape != gids.shape:
        raise ValueError("gallery_keep must match gallery identities")
    kept = gids[keep]
    return np.array([np.any(kept == qid) for qid in np.asarray(qids)], dtype=bool)


def mask_gallery(gallery, keep):
    gallery = np.asarray(gallery)
    keep = np.asarray(keep, dtype=bool)
    if keep.shape[0] != len(gallery):
        raise ValueError("keep must match gallery rows")
    if not keep.any():
        raise ValueError("gallery mask is empty")
    return gallery[keep]


def balanced_pack(query, gallery, qids, gids, k=10, with_embeddings=True, edge=0.5):
    query = np.asarray(query, dtype=np.float32)
    if query.ndim == 1:
        query = query[None]
    qids = np.asarray(qids).reshape(-1)
    gids = np.asarray(gids).reshape(-1)
    gallery = np.asarray(gallery, dtype=np.float32)
    if len(query) != len(qids):
        raise ValueError("query rows must match identities")
    if len(gallery) != len(gids):
        raise ValueError("gallery rows must match identities")
    rows, labels, cosine = [], [], []
    for item, qid in zip(query, qids, strict=True):
        present = bool(np.any(gids == qid))
        rows.append(example_vector(item, gallery, k=k, with_embeddings=with_embeddings, edge=edge))
        labels.append(int(present))
        cosine.append(float(max_cosine(item[None], gallery)[0]))
        keep = gids != qid
        if present and keep.any():
            stripped = gallery[keep]
            rows.append(example_vector(item, stripped, k=k, with_embeddings=with_embeddings, edge=edge))
            labels.append(0)
            cosine.append(float(max_cosine(item[None], stripped)[0]))
    return np.stack(rows), np.asarray(labels, dtype=int), np.asarray(cosine, dtype=np.float64)


def balanced_pairs(query, gallery, qids, gids, k=10, with_embeddings=True, edge=0.5):
    features, labels, _ = balanced_pack(
        query, gallery, qids, gids, k=k, with_embeddings=with_embeddings, edge=edge
    )
    return features, labels
