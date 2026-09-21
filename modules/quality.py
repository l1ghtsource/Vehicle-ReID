import numpy as np


def bbox_area(width, height):
    width = np.asarray(width, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    if np.any(width <= 0) or np.any(height <= 0):
        raise ValueError("BBox dimensions must be positive")
    return width * height


def as_gray(rgb):
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3 or arr.shape[2] < 1:
        raise ValueError("rgb must be HxW or HxWxC")
    return arr.mean(axis=2)


def brightness(rgb):
    return float(as_gray(rgb).mean())


def sharpness(rgb):
    gray = as_gray(rgb)
    if min(gray.shape) < 3:
        return 0.0
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(lap.var())
