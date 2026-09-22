import numpy as np

from .features import example_vector, similarities

OPEN_SET_FRACTION = 0.2


def open_set_split(qids, gids, fraction=OPEN_SET_FRACTION, seed=0):
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


def top_hit(query, gallery, qids, gids):
    sim = similarities(query, gallery)
    qids = np.asarray(qids).reshape(-1)
    gids = np.asarray(gids).reshape(-1)
    if len(qids) != len(sim):
        raise ValueError("query identities must match query rows")
    if len(gids) != sim.shape[1]:
        raise ValueError("gallery identities must match gallery rows")
    return gids[np.argmax(sim, axis=1)] == qids


def mask_gallery(gallery, keep):
    gallery = np.asarray(gallery)
    keep = np.asarray(keep, dtype=bool)
    if keep.shape[0] != len(gallery):
        raise ValueError("keep must match gallery rows")
    if not keep.any():
        raise ValueError("gallery mask is empty")
    return gallery[keep]


def _pair_inputs(query, gallery, qids, gids):
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
    return query, gallery, qids, gids


def scored_pack(query, gallery, qids, gids, k=10, with_embeddings=True, edge=0.5):
    query, gallery, qids, gids = _pair_inputs(query, gallery, qids, gids)
    rows, labels, cosine, hits = [], [], [], []
    for item, qid in zip(query, qids, strict=True):
        present = bool(np.any(gids == qid))
        rows.append(example_vector(item, gallery, k=k, with_embeddings=with_embeddings, edge=edge))
        labels.append(int(present))
        cosine.append(float(similarities(item[None], gallery).max()))
        hits.append(bool(present and top_hit(item[None], gallery, np.array([qid]), gids)[0]))
    return (
        np.stack(rows),
        np.asarray(labels, dtype=int),
        np.asarray(cosine, dtype=np.float64),
        np.asarray(hits, dtype=bool),
    )


def balanced_pack(query, gallery, qids, gids, k=10, with_embeddings=True, edge=0.5):
    query, gallery, qids, gids = _pair_inputs(query, gallery, qids, gids)
    rows, labels, cosine, hits = [], [], [], []
    for item, qid in zip(query, qids, strict=True):
        features, y, score, hit = scored_pack(
            item, gallery, np.array([qid]), gids, k=k, with_embeddings=with_embeddings, edge=edge
        )
        rows.append(features[0])
        labels.append(int(y[0]))
        cosine.append(float(score[0]))
        hits.append(bool(hit[0]))
        keep = gids != qid
        if y[0] == 1 and keep.any():
            stripped = scored_pack(
                item,
                gallery[keep],
                np.array([qid]),
                gids[keep],
                k=k,
                with_embeddings=with_embeddings,
                edge=edge,
            )
            rows.append(stripped[0][0])
            labels.append(0)
            cosine.append(float(stripped[2][0]))
            hits.append(False)
    return (
        np.stack(rows),
        np.asarray(labels, dtype=int),
        np.asarray(cosine, dtype=np.float64),
        np.asarray(hits, dtype=bool),
    )


def balanced_pairs(query, gallery, qids, gids, k=10, with_embeddings=True, edge=0.5):
    features, labels, *_ = balanced_pack(
        query, gallery, qids, gids, k=k, with_embeddings=with_embeddings, edge=edge
    )
    return features, labels


def open_set_pack(
    query,
    gallery,
    qids,
    gids,
    k=10,
    with_embeddings=True,
    edge=0.5,
    fraction=OPEN_SET_FRACTION,
    seed=0,
):
    query, gallery, qids, gids = _pair_inputs(query, gallery, qids, gids)
    _, gallery_keep, held = open_set_split(qids, gids, fraction=fraction, seed=seed)
    packed = scored_pack(
        query,
        gallery[gallery_keep],
        qids,
        gids[gallery_keep],
        k=k,
        with_embeddings=with_embeddings,
        edge=edge,
    )
    return (*packed, np.asarray(held))
