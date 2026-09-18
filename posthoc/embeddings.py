import numpy as np


def l2_normalize(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        raise ValueError("Cannot normalize empty embeddings")
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def pixel_shift(original, corrupted):
    a = np.asarray(original, dtype=np.float32)
    b = np.asarray(corrupted, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("pixel_shift shape mismatch")
    mse = float(np.mean((a - b) ** 2))
    psnr = float(10.0 * np.log10(255.0**2 / max(mse, 1e-12)))
    l1 = float(np.mean(np.abs(a - b)) / 255.0)
    return {"mse": mse, "psnr": psnr, "l1": l1}


def embedding_stability(original, corrupted):
    orig = l2_normalize(np.asarray(original, dtype=np.float32).reshape(-1))
    corr = np.asarray(corrupted, dtype=np.float32)
    if corr.ndim == 1:
        corr = corr[None]
    if corr.ndim != 2 or corr.shape[1] != orig.shape[0] or corr.shape[0] < 1:
        raise ValueError("corrupted embeddings must be (K, D) matching original")
    corr = l2_normalize(corr)
    cosine = np.clip(corr @ orig, -1.0, 1.0)
    if len(corr) == 1:
        pairwise = 1.0
    else:
        gram = corr @ corr.T
        pairwise = float(gram[np.triu_indices(len(corr), k=1)].mean())
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_min": float(cosine.min()),
        "cosine_std": float(cosine.std(ddof=0)),
        "angular_mean_deg": float(np.degrees(np.arccos(cosine)).mean()),
        "pairwise_cosine": pairwise,
        "n": int(len(corr)),
    }
