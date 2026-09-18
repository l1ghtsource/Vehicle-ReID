import time

import numpy as np

from .device import allocated_bytes, peak_bytes, reserved_bytes, reset_peak, synchronize

CONTEST_WARMUP = 50
CONTEST_REPEATS = 300
CONTEST_BATCH_SIZES = (1, 8, 16, 32)
CONTEST_MIN_SECONDS = 10.0


def summarize_ms(samples) -> dict:
    values = np.asarray(list(samples), dtype=np.float64)
    if values.size == 0:
        raise ValueError("empty latency samples")
    return {
        "n": int(values.size),
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std(ddof=0)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "p50_ms": float(np.quantile(values, 0.50)),
        "p90_ms": float(np.quantile(values, 0.90)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "samples_ms": values.astype(np.float64).tolist(),
    }


def measure_latency(run_one, device, *, warmup: int = CONTEST_WARMUP, repeats: int = CONTEST_REPEATS) -> dict:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")
    for _ in range(warmup):
        run_one()
        synchronize(device)
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        run_one()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    stats = summarize_ms(samples)
    stats["warmup"] = int(warmup)
    stats["latency_b1_ms"] = stats["p50_ms"]
    return stats


def measure_throughput(
    run_batch,
    device,
    *,
    batch_sizes=CONTEST_BATCH_SIZES,
    min_seconds: float = CONTEST_MIN_SECONDS,
    warmup_batches: int = 2,
) -> dict:
    sizes = [int(size) for size in batch_sizes]
    if not sizes or min(sizes) < 1:
        raise ValueError("batch_sizes must be nonempty positive integers")
    if min_seconds < 0:
        raise ValueError("min_seconds must be >= 0")
    if warmup_batches < 0:
        raise ValueError("warmup_batches must be >= 0")
    rows = []
    for size in sizes:
        for _ in range(warmup_batches):
            run_batch(size)
        synchronize(device)
        start = time.perf_counter()
        images = 0
        while True:
            run_batch(size)
            images += size
            elapsed = time.perf_counter() - start
            if elapsed >= min_seconds:
                break
        synchronize(device)
        elapsed = time.perf_counter() - start
        fps = images / elapsed if elapsed > 0 else float("inf")
        rows.append(
            {
                "batch_size": size,
                "images": images,
                "seconds": float(elapsed),
                "fps": float(fps),
                "ms_per_image": float(1000.0 / fps) if fps else float("inf"),
            }
        )
    best = max(rows, key=lambda row: row["fps"])
    return {
        "by_batch": rows,
        "best_batch_size": int(best["batch_size"]),
        "best_fps": float(best["fps"]),
        "min_seconds": float(min_seconds),
    }


def measure_vram(run_batch, device, *, batch_sizes=CONTEST_BATCH_SIZES, repeats: int = 3) -> list[dict]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    rows = []
    for size in batch_sizes:
        reset_peak(device)
        for _ in range(repeats):
            run_batch(int(size))
        synchronize(device)
        rows.append(
            {
                "batch_size": int(size),
                "peak_bytes": peak_bytes(device),
                "allocated_bytes": allocated_bytes(device),
                "reserved_bytes": reserved_bytes(device),
            }
        )
    return rows


def compare_embeddings(first, second) -> dict:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("embedding shapes must match")
    if left.size == 0:
        raise ValueError("empty embeddings")
    delta = np.abs(left - right)
    norms = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    dots = np.sum(left * right, axis=-1)
    cosine = np.divide(dots, np.maximum(norms, 1e-12))
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "min_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
        "bit_identical": bool(np.array_equal(np.asarray(first), np.asarray(second))),
        "allclose_fp32": bool(np.allclose(left, right, rtol=1e-5, atol=1e-6)),
    }
