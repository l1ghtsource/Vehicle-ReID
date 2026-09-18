import numpy as np
from PIL import Image, ImageDraw

from .pairs import as_hwc


def _color(score):
    value = float(np.clip(score, 0.0, 1.0))
    return (int(255 * (1.0 - value)), int(255 * value), 0)


def draw_matches(image0, image1, keypoints0, keypoints1, scores):
    left = as_hwc(image0)
    right = as_hwc(image1)
    pts0 = np.asarray(keypoints0, dtype=np.float32).reshape(-1, 2)
    pts1 = np.asarray(keypoints1, dtype=np.float32).reshape(-1, 2)
    sc = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not (len(pts0) == len(pts1) == len(sc)):
        raise ValueError("keypoints and scores must align")
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + right.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right
    picture = Image.fromarray(canvas)
    draw = ImageDraw.Draw(picture)
    shift = left.shape[1]
    for (x0, y0), (x1, y1), score in zip(pts0, pts1, sc, strict=True):
        color = _color(score)
        draw.line((float(x0), float(y0), float(x1) + shift, float(y1)), fill=color, width=2)
        draw.ellipse((float(x0) - 2, float(y0) - 2, float(x0) + 2, float(y0) + 2), fill=(0, 0, 0))
        draw.ellipse(
            (float(x1) + shift - 2, float(y1) - 2, float(x1) + shift + 2, float(y1) + 2),
            fill=(0, 0, 0),
        )
    return np.asarray(picture)
