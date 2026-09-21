import math
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg

DECODE_BACKENDS = frozenset({"pil", "cv2", "jpeg_cuda"})


def image_path(root, image_id):
    p = Path(root) / str(image_id)
    if p.suffix:
        return p
    for suffix in (".jpg", ".png", ".jpeg"):
        if p.with_suffix(suffix).is_file():
            return p.with_suffix(suffix)
    raise FileNotFoundError(p.with_suffix(".jpg"))


def read_file(path) -> bytes:
    return Path(path).read_bytes()


def crop_bbox(image, bbox, context_pct=0):
    x, y, w, h = map(float, bbox)
    if w <= 0 or h <= 0 or context_pct < 0:
        raise ValueError("BBox dimensions must be positive; context_pct must be nonnegative")
    width, height = image.size
    dx, dy = w * context_pct / 100, h * context_pct / 100
    left, top = max(0, math.floor(x - dx)), max(0, math.floor(y - dy))
    right, bottom = min(width, math.ceil(x + w + dx)), min(height, math.ceil(y + h + dy))
    if right <= left or bottom <= top:
        raise ValueError(f"BBox outside image: {bbox}, image_size={image.size}")
    return image.crop((left, top, right, bottom))


def crop_record(image: Image.Image, bbox, context_pct: float, full_image: bool = False) -> Image.Image:
    if full_image:
        return image
    return crop_bbox(image, bbox, context_pct)


def preprocess_record(image: Image.Image, transform) -> torch.Tensor:
    return transform(np.asarray(image))


def decode_jpeg_tensor(payload: bytes, device="cpu") -> torch.Tensor:
    encoded = torch.frombuffer(memoryview(payload), dtype=torch.uint8).clone()
    target = torch.device(device)
    if target.type == "cuda":
        return cast(torch.Tensor, decode_jpeg(encoded, mode=ImageReadMode.RGB, device=target))
    return cast(torch.Tensor, decode_jpeg(encoded, mode=ImageReadMode.RGB))


def decode_rgb(payload: bytes, backend: str = "pil", device="cpu") -> Image.Image:
    if backend not in DECODE_BACKENDS:
        raise ValueError("decode backend must be pil/cv2/jpeg_cuda")
    if backend == "cv2":
        array = np.frombuffer(payload, dtype=np.uint8)
        bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Failed to decode image")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if backend == "jpeg_cuda":
        rgb = decode_jpeg_tensor(payload, device=device)
        array = rgb.detach().to("cpu").permute(1, 2, 0).contiguous().numpy()
        return Image.fromarray(array)
    with Image.open(BytesIO(payload)) as image:
        image.load()
        return image.convert("RGB")


def decode_images(payloads, backend: str = "pil", workers: int = 0, device="cpu") -> list[Image.Image]:
    items = list(payloads)
    decode = partial(decode_rgb, backend=backend, device=device)
    if workers <= 1 or len(items) <= 1:
        return [decode(payload) for payload in items]
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        return list(pool.map(decode, items))


def load_record(path, bbox, transform, context_pct: float, full_image: bool = False, backend: str = "pil"):
    image = decode_rgb(read_file(path), backend)
    cropped = crop_record(image, bbox, context_pct, full_image=full_image)
    return preprocess_record(cropped, transform)


class VehicleDataset(Dataset):
    def __init__(self, frame, cfg, transform, label_map=None, train=False, context_pct=None, views=1):
        if views not in {1, 2}:
            raise ValueError("views must be 1 or 2")
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map or {}
        self.views = int(views)
        self.context = cfg.data.context_pct if context_pct is None else context_pct
        self.jitter = cfg.data.context_jitter_pct if train else 0
        self.decode_backend = str(cfg.data.get("decode_backend", "pil"))
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

    def _augmented_crop(self, index):
        row = self.frame.iloc[index]
        context = max(0.0, self.context + (2 * torch.rand(()).item() - 1) * self.jitter)
        bbox = [0.0, 0.0, 1.0, 1.0] if self.full_images[index] else [row.x, row.y, row.w, row.h]
        return load_record(
            self.paths[index],
            bbox,
            self.transform,
            context,
            full_image=self.full_images[index],
            backend=self.decode_backend,
        )

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        pid = int(row.get("vehicle_id", -1))
        item = {
            "image": self._augmented_crop(index),
            "label": self.label_map.get(pid, -1),
            "pid": pid,
            "camera": int(row.camera_id),
            "index": index,
            "image_id": str(row.image_id),
        }
        if self.views == 2:
            item["view"] = self._augmented_crop(index)
        return item
