from contextlib import nullcontext

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import InterpolationMode, rotate


@torch.inference_mode()
def embed_loader(model, loader, cfg, device):
    model.eval()
    outputs = []
    use_amp = cfg.eval.precision in {"bf16", "fp16"} and str(device).startswith("cuda")
    dtype = torch.bfloat16 if cfg.eval.precision == "bf16" else torch.float16
    tta = cfg.eval.tta
    scales = list(tta.scales) if tta.enabled else [1.0]
    angles = list(tta.rotations) if tta.enabled else [0]
    flips = [False, True] if tta.enabled and tta.hflip else [False]
    if not scales or not angles or min(scales) <= 0:
        raise ValueError("TTA scales/rotations must be nonempty and scales positive")
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        embeddings = []
        for scale in scales:
            size = [max(1, round(s * scale)) for s in x.shape[-2:]]
            z = F.interpolate(x, size=size, mode="bilinear", align_corners=False) if scale != 1 else x
            for angle in angles:
                zr = rotate(z, angle, interpolation=InterpolationMode.BILINEAR) if angle else z
                for flip in flips:
                    inp = zr.flip(-1) if flip else zr
                    with torch.autocast(device_type="cuda", dtype=dtype) if use_amp else nullcontext():
                        out = model(inp)
                        emb = out["embedding"] if isinstance(out, dict) else out
                    embeddings.append(F.normalize(emb.float(), dim=1))
        outputs.append(F.normalize(torch.stack(embeddings).mean(0), dim=1).cpu())
    if not outputs:
        raise ValueError("Empty inference dataset")
    result = torch.cat(outputs)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite inference embeddings")
    return result.numpy()
