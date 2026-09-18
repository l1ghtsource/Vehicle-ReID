import numpy as np
import torch
from torch.nn import functional as F

from .embeddings import embedding_stability, l2_normalize, pixel_shift
from .retrieval import retrieval_stability


def _mean_fields(rows):
    if not rows:
        raise ValueError("no rows to aggregate")
    out = {}
    for key in rows[0]:
        if key == "samples":
            continue
        values = [row[key] for row in rows if key in row]
        if all(isinstance(item, (bool, np.bool_)) for item in values):
            out[key] = float(np.mean(values))
        elif all(isinstance(item, (int, float, np.integer, np.floating)) for item in values):
            out[key] = float(np.nanmean(np.asarray(values, dtype=np.float64)))
    return out


def _keep_for_query(keep, gallery_size, query_index, n_queries):
    if keep is None:
        return None
    mask = np.asarray(keep, dtype=bool)
    if mask.ndim == 1:
        if mask.shape[0] != gallery_size:
            raise ValueError("keep must match gallery")
        return mask
    if mask.ndim != 2 or mask.shape != (n_queries, gallery_size):
        raise ValueError("keep must be (Q, G)")
    return mask[query_index]


def embed(model, images, device=None, batch_size=16):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    x = torch.as_tensor(images)
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("Expected NCHW RGB tensors")
    if device is not None:
        dev = torch.device(device)
    else:
        params = list(model.parameters())
        dev = params[0].device if params else torch.device("cpu")
    x = x.to(dev)
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), int(batch_size)):
            out = model(x[start : start + int(batch_size)])
            emb = out["embedding"] if isinstance(out, dict) else out
            chunks.append(F.normalize(emb.float(), dim=1))
    if not chunks:
        raise ValueError("Empty embedding batch")
    result = torch.cat(chunks)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite embeddings")
    return result.cpu().numpy()


def compare_query(
    orig_emb,
    corr_emb,
    gallery_emb,
    *,
    qid=None,
    gids=None,
    keep=None,
    k_neighbors=10,
    orig_image=None,
    corr_images=None,
    include_samples=True,
):
    if (orig_image is None) != (corr_images is None):
        raise ValueError("orig_image and corr_images must be passed together")
    emb = embedding_stability(orig_emb, corr_emb)
    orig = l2_normalize(np.asarray(orig_emb, dtype=np.float32).reshape(-1))
    corr = l2_normalize(np.asarray(corr_emb, dtype=np.float32))
    if corr.ndim == 1:
        corr = corr[None]
    gallery = l2_normalize(gallery_emb)
    orig_sim = gallery @ orig
    if corr_images is not None and len(corr_images) != len(corr):
        raise ValueError("corr_images must match corrupted embeddings")
    samples = []
    for i, item in enumerate(corr):
        row = retrieval_stability(orig_sim, gallery @ item, k=k_neighbors, keep=keep, qid=qid, gids=gids)
        if orig_image is not None and corr_images is not None:
            row.update(pixel_shift(orig_image, corr_images[i]))
        samples.append(row)
    summary = {**emb, **_mean_fields(samples)}
    if include_samples:
        summary["samples"] = samples
    return summary


def score_queries(
    orig_emb,
    corr_emb,
    gallery_emb,
    *,
    qids=None,
    gids=None,
    keep=None,
    k_neighbors=10,
    include_samples=False,
):
    orig = np.asarray(orig_emb, dtype=np.float32)
    corr = np.asarray(corr_emb, dtype=np.float32)
    gallery = np.asarray(gallery_emb, dtype=np.float32)
    if orig.ndim != 2:
        raise ValueError("orig_emb must be (Q, D)")
    if corr.ndim != 3 or corr.shape[0] != len(orig) or corr.shape[2] != orig.shape[1]:
        raise ValueError("corr_emb must be (Q, K, D)")
    if gallery.ndim != 2 or gallery.shape[1] != orig.shape[1]:
        raise ValueError("gallery_emb must be (G, D)")
    if qids is not None and len(np.asarray(qids)) != len(orig):
        raise ValueError("qids must match queries")
    if gids is not None and len(np.asarray(gids)) != len(gallery):
        raise ValueError("gids must match gallery")
    rows = []
    qids_arr = None if qids is None else np.asarray(qids)
    for i in range(len(orig)):
        mask = _keep_for_query(keep, len(gallery), i, len(orig))
        qid = None if qids_arr is None else qids_arr[i]
        row = compare_query(
            orig[i],
            corr[i],
            gallery,
            qid=qid,
            gids=gids,
            keep=mask,
            k_neighbors=k_neighbors,
            include_samples=include_samples,
        )
        row["query_index"] = i
        rows.append(row)
    return rows


def summarize(rows, by=("corruption", "severity")):
    if not rows:
        raise ValueError("no rows to summarize")
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in by)
        groups.setdefault(key, []).append(row)
    out = []
    for key, items in groups.items():
        rec = dict(zip(by, key, strict=True))
        rec["n"] = len(items)
        rec.update(_mean_fields(items))
        out.append(rec)
    return out
