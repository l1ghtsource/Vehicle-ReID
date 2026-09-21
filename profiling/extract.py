from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from dataset.images import crop_bbox
from modules.inference import embed_tensor, tta_context_pcts

from .device import as_device, is_cuda, synchronize

STAGES = ("read", "decode", "crop", "preprocess", "h2d", "forward", "l2")


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


def read_file(path) -> bytes:
    return Path(path).read_bytes()


DECODE_BACKENDS = frozenset({"pil", "cv2"})


def decode_rgb(payload: bytes, backend: str = "pil") -> Image.Image:
    if backend not in DECODE_BACKENDS:
        raise ValueError("decode backend must be pil/cv2")
    if backend == "cv2":
        array = np.frombuffer(payload, dtype=np.uint8)
        bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    with Image.open(BytesIO(payload)) as image:
        image.load()
        return image.convert("RGB")


def decode_images(payloads, backend: str = "pil", workers: int = 8) -> list[Image.Image]:
    items = list(payloads)
    if workers <= 1 or len(items) <= 1:
        return [decode_rgb(payload, backend) for payload in items]
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        return list(pool.map(partial(decode_rgb, backend=backend), items))


def crop_record(image: Image.Image, bbox, context_pct: float, full_image: bool = False) -> Image.Image:
    if full_image:
        return image
    return crop_bbox(image, bbox, context_pct)


def preprocess_record(image: Image.Image, transform) -> torch.Tensor:
    return transform(np.asarray(image))


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
    decode_workers: int = 8,
):
    records = list(zip(paths, bboxes, strict=True))
    if not records:
        raise ValueError("Empty extract batch")
    flags = list(full_images) if full_images is not None else [False] * len(records)
    if len(flags) != len(records):
        raise ValueError("full_images must match the batch")
    target = as_device(device)
    clock = StageClock(target) if timed else None
    decoded = []
    if clock is None and int(decode_workers) > 1 and len(records) > 1:
        payloads = [read_file(path) for path, _bbox in records]
        images = decode_images(payloads, decode_backend, workers=int(decode_workers))
        decoded = [
            (image, bbox, full)
            for image, (_path, bbox), full in zip(images, records, flags, strict=True)
        ]
    else:
        for (path, bbox), full_image in zip(records, flags, strict=True):
            if clock is not None:
                clock.start()
            payload = read_file(path)
            if clock is not None:
                clock.add("read")
            image = decode_rgb(payload, decode_backend)
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
