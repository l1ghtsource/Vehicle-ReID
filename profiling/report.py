from pathlib import Path

import numpy as np
import torch

from .weights import LIMIT_BYTES

LATENCY_FULL_MS = 40.0
LATENCY_ZERO_MS = 80.0
THROUGHPUT_FULL_FPS = 100.0
THROUGHPUT_ZERO_FPS = 50.0
LATENCY_WEIGHT = 0.10
THROUGHPUT_WEIGHT = 0.10
TIME_BUDGET_MULTIPLIER = 3.0


def bytes_text(n: int | float | None) -> str:
    if n is None:
        return "n/a"
    value = float(n)
    sign = "-" if value < 0 else ""
    remaining = abs(value)
    units = ("B", "KiB", "MiB", "GiB")
    for unit in units:
        if remaining < 1024.0:
            if unit == "B":
                return f"{sign}{int(remaining)} {unit}"
            return f"{sign}{remaining:.2f} {unit}"
        remaining /= 1024.0
    return f"{sign}{remaining:.2f} TiB"


def nested_tensor_bytes(obj) -> int:
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        return int(sum(nested_tensor_bytes(value) for value in obj.values()))
    if isinstance(obj, (list, tuple)):
        return int(sum(nested_tensor_bytes(value) for value in obj))
    return 0


def checkpoint_breakdown(blob) -> dict:
    state = blob.get("state_dict") if isinstance(blob, dict) else None
    ema = blob.get("ema") if isinstance(blob, dict) else None
    shadow = ema.get("shadow") if isinstance(ema, dict) else None
    keys = sorted(blob.keys()) if isinstance(blob, dict) else []
    by_key = {key: nested_tensor_bytes(blob[key]) for key in keys} if isinstance(blob, dict) else {}
    return {
        "keys": keys,
        "bytes_by_key": by_key,
        "state_dict_bytes": nested_tensor_bytes(state),
        "ema_shadow_bytes": nested_tensor_bytes(shadow),
        "total_tensor_bytes": nested_tensor_bytes(blob),
    }


def serving_bytes(breakdown: dict, weights: str) -> int:
    if weights == "ema":
        return int(breakdown["ema_shadow_bytes"] or breakdown["state_dict_bytes"])
    return int(breakdown["state_dict_bytes"])


def latency_score(ms: float) -> float:
    value = float(ms)
    if value <= LATENCY_FULL_MS:
        return 1.0
    if value >= LATENCY_ZERO_MS:
        return 0.0
    return (LATENCY_ZERO_MS - value) / (LATENCY_ZERO_MS - LATENCY_FULL_MS)


def throughput_score(fps: float) -> float:
    value = float(fps)
    if value >= THROUGHPUT_FULL_FPS:
        return 1.0
    if value <= THROUGHPUT_ZERO_FPS:
        return 0.0
    return (value - THROUGHPUT_ZERO_FPS) / (THROUGHPUT_FULL_FPS - THROUGHPUT_ZERO_FPS)


def time_budget_seconds(latency_ms: float, n_test) -> float | None:
    if n_test is None:
        return None
    return float(latency_ms) / 1000.0 * float(n_test) * TIME_BUDGET_MULTIPLIER


def performance_scores(latency_ms, fps, *, within_limit: bool, n_test=None) -> dict:
    budget = time_budget_seconds(latency_ms, n_test)
    if not within_limit:
        return {
            "performance_eligible": False,
            "latency_score": None,
            "throughput_score": None,
            "performance_score": None,
            "time_budget_s": budget,
        }
    lat = latency_score(latency_ms)
    thr = throughput_score(fps)
    return {
        "performance_eligible": True,
        "latency_score": float(lat),
        "throughput_score": float(thr),
        "performance_score": float(LATENCY_WEIGHT * lat + THROUGHPUT_WEIGHT * thr),
        "time_budget_s": budget,
    }


def contest_summary(
    *,
    inventory: dict,
    latency: dict,
    throughput: dict,
    vram: list[dict],
    load_ms: float,
    load_snapshot: dict,
    determinism: dict,
    serving_weight_bytes: int,
    n_test=None,
) -> dict:
    peak = max((row["peak_bytes"] for row in vram), default=load_snapshot.get("peak_bytes", 0) or 0)
    scores = performance_scores(
        latency["latency_b1_ms"],
        throughput["best_fps"],
        within_limit=bool(inventory["within_limit"]),
        n_test=n_test,
    )
    return {
        "latency_b1_ms": float(latency["latency_b1_ms"]),
        "throughput_best_fps": float(throughput["best_fps"]),
        "throughput_best_batch_size": int(throughput["best_batch_size"]),
        "peak_vram_bytes": int(peak),
        "load_ms": float(load_ms),
        "weight_bytes": int(inventory["total_bytes"]),
        "weight_limit_bytes": int(LIMIT_BYTES),
        "within_2gib": bool(inventory["within_limit"]),
        "serving_tensor_bytes": int(serving_weight_bytes),
        "deterministic": bool(determinism["bit_identical"] or determinism["allclose_fp32"]),
        "load_snapshot": load_snapshot,
        **scores,
    }


def as_jsonable(obj):
    if isinstance(obj, dict):
        return {str(key): as_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    return obj
