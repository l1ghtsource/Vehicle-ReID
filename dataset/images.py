import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def image_path(root, image_id):
    p = Path(root) / str(image_id)
    if p.suffix:
        return p
    for suffix in (".jpg", ".png", ".jpeg"):
        if p.with_suffix(suffix).is_file():
            return p.with_suffix(suffix)
    raise FileNotFoundError(p.with_suffix(".jpg"))


def crop_bbox(image, bbox, context_pct=0):
    x, y, w, h = map(float, bbox)
    if w <= 0 or h <= 0 or context_pct < 0:
        raise ValueError("BBox dimensions must be positive; context_pct must be nonnegative")
    W, H = image.size
    dx, dy = w * context_pct / 100, h * context_pct / 100
    left, top = max(0, math.floor(x - dx)), max(0, math.floor(y - dy))
    right, bottom = min(W, math.ceil(x + w + dx)), min(H, math.ceil(y + h + dy))
    if right <= left or bottom <= top:
        raise ValueError(f"BBox outside image: {bbox}, image_size={image.size}")
    return image.crop((left, top, right, bottom))


class VehicleDataset(Dataset):
    def __init__(self, frame, cfg, transform, label_map=None, train=False, context_pct=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map or {}
        self.context = cfg.data.context_pct if context_pct is None else context_pct
        self.jitter = cfg.data.context_jitter_pct if train else 0
        if "image_path" in self.frame:
            self.paths = [Path(value) for value in self.frame.image_path]
        else:
            self.paths = [image_path(cfg.data.image_dir, value) for value in self.frame.image_id]
        self.full_images = (
            self.frame.full_image.astype(bool).tolist()
            if "full_image" in self.frame
            else [False] * len(self.frame)
        )
        if cfg.data.verify_files:
            missing = [str(p) for p in self.paths if not p.is_file()]
            if missing:
                raise FileNotFoundError(missing[:10])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        context = max(0.0, self.context + (2 * torch.rand(()).item() - 1) * self.jitter)
        with Image.open(self.paths[index]) as im:
            im = im.convert("RGB")
            if not self.full_images[index]:
                im = crop_bbox(im, [row.x, row.y, row.w, row.h], context)
            x = self.transform(np.asarray(im))
        pid = int(row.get("vehicle_id", -1))
        return {
            "image": x,
            "label": self.label_map.get(pid, -1),
            "pid": pid,
            "camera": int(row.camera_id),
            "index": index,
            "image_id": str(row.image_id),
        }
