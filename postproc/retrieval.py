import numpy as np
import torch
from torch.nn import functional as F

from third_party.reranking.re_ranking_original import re_ranking


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def dense_guard(n, max_gb, matrices=12):
    estimate = matrices * n * n * 4 / 1024**3
    if estimate > max_gb:
        raise MemoryError(
            f"Dense reranking estimate {estimate:.2f} GiB > postproc.max_dense_gb={max_gb}; "
            "reduce gallery, disable reranking, or explicitly raise the budget"
        )


def aqe(query, gallery, k=5, alpha=3.0, iterations=1, gallery_only=True, chunk_size=256):

    q, g = normalize(query), normalize(gallery)
    if k < 1 or iterations < 1 or alpha < 0:
        raise ValueError("AQE requires k>=1, iterations>=1, alpha>=0")
    for _ in range(iterations):
        ref = g if gallery_only else np.concatenate([q, g])
        all_x = np.concatenate([q, g])
        out = []
        for start in range(0, len(all_x), chunk_size):
            x = all_x[start : start + chunk_size]
            sim = x @ ref.T
            idx = np.argsort(-sim, axis=1, kind="stable")[:, : min(k, len(ref))]
            weight = np.take_along_axis(sim, idx, axis=1).clip(0) ** alpha
            mixed = (ref[idx] * weight[..., None]).sum(1)
            mixed = np.where(np.linalg.norm(mixed, axis=1, keepdims=True) > 1e-12, mixed, x)
            out.append(normalize(mixed))
        joined = np.concatenate(out)
        q, g = joined[: len(q)], joined[len(q) :]
    return q, g


def aggregate_gallery(gallery, k=3, alpha=0.7, min_similarity=0.9, chunk_size=256):

    g = normalize(gallery)
    if not 0 <= alpha <= 1 or k < 1:
        raise ValueError("Invalid gallery aggregation parameters")
    neighbors, scores = [], []
    for start in range(0, len(g), chunk_size):
        sim = g[start : start + chunk_size] @ g.T
        idx = np.argsort(-sim, axis=1, kind="stable")[:, : min(k + 1, len(g))]
        neighbors.extend(idx)
        scores.extend(np.take_along_axis(sim, idx, 1))
    out = g.copy()
    for i, (idx, sims) in enumerate(zip(neighbors, scores, strict=True)):
        selected = [
            j for j, s in zip(idx, sims, strict=True) if j != i and s >= min_similarity and i in neighbors[j]
        ]
        if selected:
            prototype = normalize(g[[i] + selected].mean(0, keepdims=True))[0]
            out[i] = alpha * g[i] + (1 - alpha) * prototype
    return normalize(out)


def k_reciprocal(query, gallery, k1=20, k2=6, lambda_value=0.3, max_dense_gb=4):
    q, g = normalize(query), normalize(gallery)
    dense_guard(len(q) + len(g), max_dense_gb)
    k1, k2 = min(k1, len(q) + len(g) - 1), min(k2, len(q) + len(g))
    if k1 < 1 or k2 < 1 or not 0 <= lambda_value <= 1:
        raise ValueError("Invalid k-reciprocal parameters")

    joined = np.concatenate([q, g])
    if np.max(np.abs(joined - joined[0])) < 1e-7:
        return np.zeros((len(q), len(g)), dtype=np.float32)
    result = re_ranking(q @ g.T, q @ q.T, g @ g.T, k1=k1, k2=k2, lambda_value=lambda_value)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite k-reciprocal distances")
    return result


@torch.no_grad()
def gnn_rerank(query, gallery, k1=20, k2=6, lambda_value=0.3, device="cpu", max_dense_gb=4):

    q = torch.as_tensor(normalize(query), device=device)
    g = torch.as_tensor(normalize(gallery), device=device)
    x = torch.cat([q, g])
    dense_guard(len(x), max_dense_gb)
    k1 = min(int(k1), len(x))
    k2 = min(int(k2), k1)
    if k1 < 1 or k2 < 1 or not 0 <= lambda_value <= 1:
        raise ValueError("Invalid GNN parameters")
    s, idx = (x @ x.T).topk(k1, dim=1, sorted=True)
    a = torch.zeros(len(x), len(x), device=device).scatter_(1, idx, 1.0)
    s = s.square()
    if k2 != 1:
        for _ in range(2):
            sym = a + a.T

            a = torch.zeros_like(sym)
            for j in range(k2):
                a.add_(sym[idx[:, j]] * s[:, j : j + 1])
            a = F.normalize(a, dim=1)
    similarity = (1 - lambda_value) * (a[: len(q)] @ a[len(q) :].T) + lambda_value * (q @ g.T)
    return (1 - similarity).cpu().numpy()


def postprocess(query, gallery, cfg):
    q, g = normalize(query), normalize(gallery)
    if not cfg.enabled:
        return 1 - q @ g.T, q, g
    if cfg.gallery_aggregation.enabled:
        p = {k: v for k, v in cfg.gallery_aggregation.items() if k != "enabled"}
        g = aggregate_gallery(g, **p)
    if cfg.aqe.enabled:
        p = {k: v for k, v in cfg.aqe.items() if k != "enabled"}
        q, g = aqe(q, g, **p)
    r = cfg.rerank
    common = dict(k1=r.k1, k2=r.k2, lambda_value=r.lambda_value, max_dense_gb=cfg.max_dense_gb)
    if r.kind == "none":
        distance = 1 - q @ g.T
    elif r.kind == "k_reciprocal":
        distance = k_reciprocal(q, g, **common)
    elif r.kind == "gnn":
        distance = gnn_rerank(q, g, device=r.device, **common)
    else:
        raise ValueError(f"Unknown reranker {r.kind}")
    return distance, q, g
