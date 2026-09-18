import numpy as np

STAT_NAMES = (
    "top1",
    "top2",
    "top3",
    "gap12",
    "gap13",
    "mean_k",
    "std_k",
    "median_k",
    "min_k",
    "mean_top3",
    "mean_top5",
    "entropy",
    "pairwise_mean",
    "pairwise_std",
    "query_to_mean",
    "n_ge_05",
    "n_ge_07",
    "n_ge_09",
    "reciprocal",
    "n_edges",
    "n_components",
    "mean_clustering",
    "margin_rest",
)


def l2_normalize(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        raise ValueError("Cannot normalize empty embeddings")
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def similarities(query, gallery):
    q = np.asarray(query, dtype=np.float32)
    g = np.asarray(gallery, dtype=np.float32)
    if q.ndim == 1:
        q = q[None]
    if g.ndim != 2 or g.shape[1] != q.shape[1]:
        raise ValueError("gallery embeddings must be (G, D) matching query")
    if g.shape[0] < 1:
        raise ValueError("gallery is empty")
    return l2_normalize(q) @ l2_normalize(g).T


def _softmax_entropy(values):
    z = values - float(values.max())
    p = np.exp(z)
    p = p / np.maximum(p.sum(), 1e-12)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def _components(adj):
    n = len(adj)
    seen = np.zeros(n, dtype=bool)
    count = 0
    for start in range(n):
        if seen[start]:
            continue
        count += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            for nxt in np.flatnonzero(adj[node]):
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(int(nxt))
    return count


def _clustering(adj):
    n = len(adj)
    scores = []
    for i in range(n):
        nbr = np.flatnonzero(adj[i])
        deg = len(nbr)
        if deg < 2:
            scores.append(0.0)
            continue
        sub = adj[np.ix_(nbr, nbr)]
        triangles = float(np.tril(sub, k=-1).sum())
        scores.append(2.0 * triangles / (deg * (deg - 1)))
    return float(np.mean(scores)) if scores else 0.0


def stat_vector(query, gallery, k=10, edge=0.5):
    if k < 1:
        raise ValueError("k must be >= 1")
    if not 0 <= edge <= 1:
        raise ValueError("edge threshold must be in [0, 1]")
    q = l2_normalize(np.asarray(query, dtype=np.float32).reshape(-1))
    g = l2_normalize(np.asarray(gallery, dtype=np.float32))
    sim = similarities(q, g)[0]
    k_eff = min(int(k), len(g))
    order = np.argsort(-sim, kind="stable")[:k_eff]
    top = sim[order]
    padded = np.pad(top, (0, max(0, 3 - len(top))), constant_values=top[-1])
    rest = top[1:] if len(top) > 1 else top
    local = g[order]
    gram = local @ local.T
    iu = np.triu_indices(k_eff, k=1)
    pair = gram[iu] if k_eff > 1 else np.array([1.0], dtype=np.float32)
    mean_local = l2_normalize(local.mean(0))
    adj = gram >= float(edge)
    np.fill_diagonal(adj, False)
    retrieved = np.concatenate([q[None], local], 0)
    reciprocal = 0.0
    for i, idx in enumerate(order):
        to_set = g[idx] @ retrieved.T
        mask = np.ones(len(retrieved), dtype=bool)
        mask[i + 1] = False
        reciprocal += float(np.argmax(np.where(mask, to_set, -np.inf)) == 0)
    return np.array(
        [
            float(padded[0]),
            float(padded[1]),
            float(padded[2]),
            float(padded[0] - padded[1]),
            float(padded[0] - padded[2]),
            float(top.mean()),
            float(top.std(ddof=0)),
            float(np.median(top)),
            float(top.min()),
            float(top[: min(3, k_eff)].mean()),
            float(top[: min(5, k_eff)].mean()),
            _softmax_entropy(top),
            float(pair.mean()),
            float(pair.std(ddof=0)),
            float(q @ mean_local),
            float((top >= 0.5).sum()),
            float((top >= 0.7).sum()),
            float((top >= 0.9).sum()),
            reciprocal,
            float(np.tril(adj, k=-1).sum()),
            float(_components(adj)),
            _clustering(adj),
            float(padded[0] - rest.mean()),
        ],
        dtype=np.float32,
    )


def example_vector(query, gallery, k=10, with_embeddings=True, edge=0.5):
    stats = stat_vector(query, gallery, k=k, edge=edge)
    if not with_embeddings:
        return stats
    q = l2_normalize(np.asarray(query, dtype=np.float32).reshape(-1))
    g = l2_normalize(np.asarray(gallery, dtype=np.float32))
    sim = similarities(q, g)[0]
    top1 = g[int(np.argmax(sim))]
    return np.concatenate([stats, q, top1]).astype(np.float32)


def batch_examples(query, gallery, k=10, with_embeddings=True, edge=0.5):
    q = l2_normalize(np.asarray(query, dtype=np.float32))
    if q.ndim == 1:
        q = q[None]
    rows = [example_vector(item, gallery, k=k, with_embeddings=with_embeddings, edge=edge) for item in q]
    return np.stack(rows)


def feature_names(embedding_dim=0):
    names = list(STAT_NAMES)
    if embedding_dim < 0:
        raise ValueError("embedding_dim must be >= 0")
    if embedding_dim:
        names.extend([f"q_{i}" for i in range(int(embedding_dim))])
        names.extend([f"g_{i}" for i in range(int(embedding_dim))])
    return names
