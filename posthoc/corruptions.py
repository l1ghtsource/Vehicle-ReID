from io import BytesIO

import cv2
import numpy as np
from PIL import Image

SEVERITIES = (1, 2, 3, 4, 5)


def as_hwc(image):
    x = np.asarray(image)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("Expected HWC RGB image")
    if x.dtype != np.uint8:
        raise ValueError("Expected uint8 HWC RGB")
    return np.ascontiguousarray(x)


def _severity(severity):
    level = int(severity)
    if level not in SEVERITIES:
        raise ValueError("severity must be in 1..5")
    return level


def _clip(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def identity(image, severity, rng):
    return np.array(image, copy=True)


def gaussian_noise(image, severity, rng):
    sigma = (0.04, 0.08, 0.12, 0.18, 0.26)[severity - 1] * 255.0
    noise = rng.normal(0.0, sigma, image.shape)
    return _clip(image.astype(np.float32) + noise)


def impulse_noise(image, severity, rng):
    amount = (0.02, 0.04, 0.07, 0.12, 0.18)[severity - 1]
    out = image.copy()
    mask = rng.random(image.shape[:2]) < amount
    salt = rng.random(image.shape[:2]) < 0.5
    out[mask & salt] = 255
    out[mask & ~salt] = 0
    return out


def gaussian_blur(image, severity, rng):
    k = (3, 5, 7, 9, 13)[severity - 1]
    return cv2.GaussianBlur(image, (k, k), 0)


def motion_blur(image, severity, rng):
    k = (5, 9, 13, 17, 21)[severity - 1]
    kernel = np.zeros((k, k), np.float32)
    direction = int(rng.integers(3))
    if direction == 0:
        kernel[k // 2, :] = 1.0 / k
    elif direction == 1:
        kernel[:, k // 2] = 1.0 / k
    else:
        np.fill_diagonal(kernel, 1.0 / k)
    return cv2.filter2D(image, -1, kernel)


def jpeg(image, severity, rng):
    quality = int(np.clip((70, 50, 35, 20, 10)[severity - 1] + int(rng.integers(-2, 3)), 5, 95))
    buf = BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def brightness(image, severity, rng):
    delta = (0.1, 0.2, 0.35, 0.5, 0.7)[severity - 1]
    sign = 1.0 if int(rng.integers(2)) == 0 else -1.0
    return _clip(image.astype(np.float32) * (1.0 + sign * delta))


def contrast(image, severity, rng):
    factor = (0.8, 0.6, 0.45, 0.3, 0.15)[severity - 1]
    if int(rng.integers(2)) == 1:
        factor = min(2.5, 1.0 / max(factor, 0.15))
    x = image.astype(np.float32)
    return _clip((x - x.mean()) * factor + x.mean())


def saturate(image, severity, rng):
    factor = (0.7, 0.5, 0.3, 0.15, 0.0)[severity - 1]
    if int(rng.integers(2)) == 1 and factor > 0:
        factor = min(1.8, 1.0 / max(factor, 0.2))
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def downsample(image, severity, rng):
    scale = (2, 3, 4, 6, 8)[severity - 1]
    height, width = image.shape[:2]
    small_w = max(1, width // scale)
    small_h = max(1, height // scale)
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def occlude(image, severity, rng):
    n_boxes = (1, 1, 2, 2, 3)[severity - 1]
    frac = (0.08, 0.12, 0.18, 0.25, 0.35)[severity - 1]
    out = image.copy()
    height, width = out.shape[:2]
    area = height * width * frac / n_boxes
    for _ in range(n_boxes):
        box_h = max(1, min(height, int(np.sqrt(area) * float(rng.uniform(0.6, 1.4)))))
        box_w = max(1, min(width, int(area / box_h)))
        y = int(rng.integers(0, height - box_h + 1))
        x = int(rng.integers(0, width - box_w + 1))
        out[y : y + box_h, x : x + box_w] = rng.integers(0, 256, size=3)
    return out


def rotate(image, severity, rng):
    angle = (4, 8, 12, 18, 25)[severity - 1]
    sign = 1.0 if int(rng.integers(2)) == 0 else -1.0
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), sign * angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def crop(image, severity, rng):
    keep = (0.9, 0.8, 0.7, 0.58, 0.45)[severity - 1]
    height, width = image.shape[:2]
    new_h = max(1, int(height * keep))
    new_w = max(1, int(width * keep))
    y = int(rng.integers(0, height - new_h + 1))
    x = int(rng.integers(0, width - new_w + 1))
    return cv2.resize(image[y : y + new_h, x : x + new_w], (width, height), interpolation=cv2.INTER_LINEAR)


def fog(image, severity, rng):
    amount = (0.15, 0.3, 0.45, 0.6, 0.8)[severity - 1]
    color = rng.integers(200, 256, size=3).astype(np.float32)
    return _clip(image.astype(np.float32) * (1.0 - amount) + color * amount)


CORRUPTIONS = {
    "identity": identity,
    "gaussian_noise": gaussian_noise,
    "impulse_noise": impulse_noise,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "jpeg": jpeg,
    "brightness": brightness,
    "contrast": contrast,
    "saturate": saturate,
    "downsample": downsample,
    "occlude": occlude,
    "rotate": rotate,
    "crop": crop,
    "fog": fog,
}


def apply(image, name, severity=3, seed=0):
    if name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption {name}")
    rgb = as_hwc(image)
    return CORRUPTIONS[name](rgb, _severity(severity), np.random.default_rng(seed))


def apply_k(image, name, severity=3, k=4, seed=0):
    if k < 1:
        raise ValueError("k must be >= 1")
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=int(k), dtype=np.int64)
    return np.stack([apply(image, name, severity, int(item)) for item in seeds])
