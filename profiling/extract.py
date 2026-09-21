from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from dataset.images import crop_record, decode_images, decode_rgb, preprocess_record, read_file
from modules.inference import embed_tensor, tta_context_pcts

from .device import as_device, is_cuda, synchronize

STAGES = ("read", "decode", "crop", "preprocess", "h2d", "forward", "l2")

__all__ = [
    "STAGES",
    "StageClock",
    "crop_record",
    "decode_images",
    "decode_rgb",
    "extract",
    "preprocess_record",
    "read_file",
]


class StageClock:
    def __init__(self, device):
        self.device = as_device(device)
        self.ms = {name: 0.0 for name in STAGES}
        self._mark = 0.0

    def start(self) -> None:
        synchronize(self.device)
        self._mark = perf_counter()

    def add(self, name: str) -> None:
        synchronize(self.device)
        now = perf_counter()
        self.ms[name] += (now - self._mark) * 1000.0
        self._mark = now


def extract(
    paths,
    bboxes,
    transform,
    model,
    device,
    *,
    context_pct: float = 0.0,
    full_images=None,
    precision: str = "fp32",
    tta=None,
    model_cfg=None,
    timed: bool = False,
    decode_backend: str = "pil",
    decode_workers: int = 0,
):
    records = list(zip(paths, bboxes, strict=True))
    if not records:
        raise ValueError("Empty extract batch")
    if decode_workers < 0:
        raise ValueError("decode_workers must be nonnegative")
    flags = list(full_images) if full_images is not None else [False] * len(records)
    if len(flags) != len(records):
        raise ValueError("full_images must match the batch")
    target = as_device(device)
    clock = StageClock(target) if timed else None
    decode_device = target if decode_backend == "jpeg_cuda" and is_cuda(target) else "cpu"
    decoded = []
    for (path, bbox), full_image in zip(records, flags, strict=True):
        if clock is not None:
            clock.start()
        payload = read_file(path)
        if clock is not None:
            clock.add("read")
        image = decode_rgb(payload, decode_backend, device=decode_device)
        if clock is not None:
            clock.add("decode")
        decoded.append((image, bbox, full_image))
    views = []
    for context in tta_context_pcts(tta, context_pct):
        tensors = []
        for image, bbox, full_image in decoded:
            if clock is not None:
                clock.start()
            cropped = crop_record(image, bbox, context, full_image=full_image)
            if clock is not None:
                clock.add("crop")
            tensors.append(preprocess_record(cropped, transform))
            if clock is not None:
                clock.add("preprocess")
        if clock is not None:
            clock.start()
        batch = torch.stack(tensors).to(target, non_blocking=is_cuda(target))
        if clock is not None:
            clock.add("h2d")
        views.append(embed_tensor(model, batch, target, precision=precision, tta=tta, model_cfg=model_cfg))
        if clock is not None:
            clock.add("forward")
    if clock is not None:
        clock.start()
    raw = torch.stack(views).mean(0)
    out = F.normalize(raw.float(), dim=1)
    if clock is not None:
        clock.add("l2")
    embeddings = out.detach().cpu().numpy().astype(np.float32, copy=False)
    if clock is None:
        return embeddings
    n = len(records)
    per_image = {name: clock.ms[name] / n for name in STAGES}
    total_ms = float(sum(clock.ms.values()))
    return embeddings, {"batch_ms": dict(clock.ms), "per_image_ms": per_image, "total_ms": total_ms, "n": n}
