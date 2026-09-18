import numpy as np
import torch
from torch.nn import functional as F


def as_nchw(feat, prefix=0):
    if feat.ndim == 4:
        return feat
    if feat.ndim == 2:
        feat = feat.unsqueeze(1)
    if feat.ndim != 3:
        raise ValueError(f"Expected NCHW/NLC/NC features, got {tuple(feat.shape)}")
    spatial = feat[:, int(prefix) :]
    if spatial.shape[1] == 0:
        raise ValueError("No spatial tokens after stripping prefix tokens")
    batch, tokens, channels = spatial.shape
    side = int(tokens**0.5)
    height, width = (side, side) if side * side == tokens else (1, tokens)
    return spatial.transpose(1, 2).reshape(batch, channels, height, width)


def upsample(maps, size):
    maps = torch.as_tensor(maps, dtype=torch.float32)
    if maps.ndim == 2:
        maps = maps.unsqueeze(0)
    if maps.ndim == 3:
        maps = maps.unsqueeze(1)
    if maps.ndim != 4:
        raise ValueError(f"Expected maps NHW/NCHW, got {tuple(maps.shape)}")
    height, width = int(size[0]), int(size[1])
    if height < 1 or width < 1:
        raise ValueError("Output size must be positive")
    return F.interpolate(maps, size=(height, width), mode="bilinear", align_corners=False)[:, 0]


def normalize_maps(maps):
    maps = np.asarray(maps, dtype=np.float32)
    if maps.ndim == 2:
        maps = maps[None]
    if maps.ndim != 3:
        raise ValueError(f"Expected NHW maps, got {maps.shape}")
    flat = maps.reshape(len(maps), -1)
    lo = flat.min(1)
    hi = flat.max(1)
    scale = np.maximum(hi - lo, 1e-12)
    return ((flat - lo[:, None]) / scale[:, None]).reshape(maps.shape)


def overlay(image, heatmap, alpha=0.45):
    if not 0 <= alpha <= 1:
        raise ValueError("overlay alpha must be in [0, 1]")
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("overlay image must be HWC or CHW RGB")
    if rgb.max() <= 1.5:
        rgb = rgb * 255.0
    heat = normalize_maps(np.asarray(heatmap, dtype=np.float32))[0]
    if heat.shape != rgb.shape[:2]:
        raise ValueError("heatmap spatial size must match the image")
    color = np.stack([heat, np.clip(1.0 - 2.0 * np.abs(heat - 0.5), 0, 1), 1.0 - heat], axis=-1) * 255.0
    return np.clip((1.0 - alpha) * rgb + alpha * color, 0, 255).astype(np.uint8)


def embedding_score(out, reference=None):
    if isinstance(out, dict):
        emb = out["embedding"]
        feat = out["neck"] if "neck" in out else out.get("raw", emb)
    else:
        emb = feat = out
    if reference is None:
        return feat.float().pow(2).sum()
    ref = F.normalize(torch.as_tensor(reference, device=emb.device, dtype=torch.float32), dim=-1)
    if ref.ndim == 1:
        ref = ref.unsqueeze(0)
    if ref.shape[0] not in {1, len(emb)} or ref.shape[-1] != emb.shape[-1]:
        raise ValueError("reference embedding must match the batch embedding width")
    return (emb.float() * ref).sum()


class Unfrozen:
    def __init__(self, model):
        self.model = model
        self.was_frozen = False

    def __enter__(self):
        self.was_frozen = bool(getattr(self.model, "frozen", False))
        if self.was_frozen:
            self.model.freeze_backbone(False)
        self.model.eval()
        return self

    def __exit__(self, *exc):
        if self.was_frozen:
            self.model.freeze_backbone(True)


class FeatureHook:
    def __init__(self, model):
        self.model = model
        self.feature = None
        self.handle = None

    def _keep(self, _module, _inputs, output):
        self.feature = output[-1] if isinstance(output, (list, tuple)) else output

    def __enter__(self):
        self.handle = self.model.backbone.register_forward_hook(self._keep)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()


class AttentionCapture:
    def __init__(self):
        self.maps = []
        self._tensor = torch.Tensor.softmax

    def __enter__(self):
        capture = self

        def softmax(self, dim=None, dtype=None):
            out = capture._tensor(self, dim) if dtype is None else capture._tensor(self, dim, dtype=dtype)
            if self.ndim == 4 and dim in {-1, 3}:
                capture.maps.append(out)
            return out

        torch.Tensor.softmax = softmax
        return self

    def __exit__(self, *exc):
        torch.Tensor.softmax = self._tensor


def prefix_tokens(model):
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return 0
    return int(getattr(backbone, "prefix", 0) or 0)
