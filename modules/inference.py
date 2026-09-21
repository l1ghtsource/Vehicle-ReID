from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode, rotate

from augmentations import build_transforms
from dataset.images import VehicleDataset
from models.input_size import scaled_hw, spatial_multiple


def tta_views(tta) -> tuple[list[float], list[float], list[bool]]:
    if tta is None or not bool(getattr(tta, "enabled", False)):
        return [1.0], [0.0], [False]
    scales = [float(value) for value in tta.scales]
    angles = [float(value) for value in tta.rotations]
    flips = [False, True] if bool(tta.hflip) else [False]
    if not scales or not angles or min(scales) <= 0:
        raise ValueError("TTA scales/rotations must be nonempty and scales positive")
    return scales, angles, flips


def tta_context_pcts(tta, default_context: float) -> list[float]:
    if tta is None or not bool(getattr(tta, "enabled", False)):
        return [float(default_context)]
    contexts = getattr(tta, "context_pcts", None)
    if not contexts:
        return [float(default_context)]
    return [float(value) for value in contexts]


def autocast_context(device, precision: str):
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32/bf16/fp16")
    if precision == "fp32" or not str(device).startswith("cuda"):
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _embedding(output) -> torch.Tensor:
    return output["embedding"] if isinstance(output, dict) else output


def embed_tensor(model, batch: torch.Tensor, device, precision: str = "fp32", tta=None, model_cfg=None):
    scales, angles, flips = tta_views(tta)
    backend = str(getattr(model_cfg, "backend", "")) if model_cfg is not None else ""
    multiple = spatial_multiple(model_cfg) if model_cfg is not None else 1
    embeddings = []
    with torch.inference_mode(), autocast_context(device, precision):
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
        raise FloatingPointError("Nonfinite inference embeddings")
    return result


@torch.inference_mode()
def embed_loader(model, loader, cfg, device):
    model.eval()
    outputs = []
    precision = str(cfg.eval.precision)
    tta = cfg.eval.tta
    model_cfg = cfg.model
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        outputs.append(
            embed_tensor(model, image, device, precision=precision, tta=tta, model_cfg=model_cfg).cpu()
        )
    if not outputs:
        raise ValueError("Empty inference dataset")
    return torch.cat(outputs).numpy()


def embed_frame(model, cfg, device, frame):
    device = torch.device(device)
    contexts = tta_context_pcts(cfg.eval.tta, cfg.data.context_pct)
    transform = build_transforms(cfg)
    table = frame.reset_index(drop=True)
    views = []
    for context in contexts:
        loader = DataLoader(
            VehicleDataset(table, cfg, transform, context_pct=float(context)),
            batch_size=int(cfg.data.batch_size_eval),
            shuffle=False,
            num_workers=int(cfg.data.num_workers),
            pin_memory=bool(cfg.data.pin_memory),
        )
        views.append(embed_loader(model, loader, cfg, device))
    emb = np.stack(views).mean(0)
    return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
