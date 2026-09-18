from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import InterpolationMode, rotate

from dataset.images import crop_bbox
from models.input_size import scaled_hw, spatial_multiple

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


def decode_rgb(payload: bytes) -> Image.Image:
    with Image.open(BytesIO(payload)) as image:
        image.load()
        return image.convert("RGB")


def crop_record(image: Image.Image, bbox, context_pct: float, full_image: bool = False) -> Image.Image:
    if full_image:
        return image
    return crop_bbox(image, bbox, context_pct)


def preprocess_record(image: Image.Image, transform) -> torch.Tensor:
    return transform(np.asarray(image))


def _tta_views(tta) -> tuple[list[float], list[float], list[bool]]:
    if tta is None or not bool(getattr(tta, "enabled", False)):
        return [1.0], [0.0], [False]
    scales = [float(value) for value in tta.scales]
    angles = [float(value) for value in tta.rotations]
    flips = [False, True] if bool(tta.hflip) else [False]
    if not scales or not angles or min(scales) <= 0:
        raise ValueError("TTA scales/rotations must be nonempty and scales positive")
    return scales, angles, flips


def autocast_context(device, precision: str):
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32/bf16/fp16")
    if precision == "fp32" or not is_cuda(device):
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _embedding(output) -> torch.Tensor:
    return output["embedding"] if isinstance(output, dict) else output


def embed_tensor(model, batch: torch.Tensor, device, precision: str = "fp32", tta=None, model_cfg=None):
    target = as_device(device)
    scales, angles, flips = _tta_views(tta)
    backend = str(getattr(model_cfg, "backend", "")) if model_cfg is not None else ""
    multiple = spatial_multiple(model_cfg) if model_cfg is not None else 1
    embeddings = []
    with torch.inference_mode(), autocast_context(target, precision):
        for scale in scales:
            native = (int(batch.shape[-2]), int(batch.shape[-1]))
            size = native if backend == "llm2clip" else scaled_hw(native[0], native[1], scale, multiple)
            resized = (
                batch
                if size == native
                else F.interpolate(batch, size=size, mode="bilinear", align_corners=False)
            )
            for angle in angles:
                rotated = (
                    rotate(resized, angle, interpolation=InterpolationMode.BILINEAR) if angle else resized
                )
                for flip in flips:
                    inp = rotated.flip(-1) if flip else rotated
                    embeddings.append(F.normalize(_embedding(model(inp)).float(), dim=1))
    stacked = torch.stack(embeddings).mean(0)
    result = F.normalize(stacked.float(), dim=1)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite extract embeddings")
    return result


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
):
    records = list(zip(paths, bboxes, strict=True))
    if not records:
        raise ValueError("Empty extract batch")
    flags = list(full_images) if full_images is not None else [False] * len(records)
    if len(flags) != len(records):
        raise ValueError("full_images must match the batch")
    target = as_device(device)
    clock = StageClock(target) if timed else None
    tensors = []
    for (path, bbox), full_image in zip(records, flags, strict=True):
        if clock is not None:
            clock.start()
        payload = read_file(path)
        if clock is not None:
            clock.add("read")
        image = decode_rgb(payload)
        if clock is not None:
            clock.add("decode")
        cropped = crop_record(image, bbox, context_pct, full_image=full_image)
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
    raw = embed_tensor(model, batch, target, precision=precision, tta=tta, model_cfg=model_cfg)
    if clock is not None:
        clock.add("forward")
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
