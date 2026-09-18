import numpy as np
import torch
from PIL import Image


def as_hwc(image):
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("image must be HWC/CHW RGB or PIL")
    if arr.dtype != np.uint8:
        peak = float(np.max(arr)) if arr.size else 0.0
        arr = np.clip(arr * 255.0 if peak <= 1.5 else arr, 0, 255).astype(np.uint8)
    return arr


def image_hw(image):
    rgb = as_hwc(image)
    return int(rgb.shape[0]), int(rgb.shape[1])


def _as_pairs(images):
    if not isinstance(images, (list, tuple)) or len(images) < 1:
        raise ValueError("images must be a pair or a list of pairs")
    first = images[0]
    if isinstance(first, (list, tuple)):
        pairs = [tuple(pair) for pair in images]
    elif len(images) == 2:
        pairs = [tuple(images)]
    else:
        raise ValueError("images must be a pair or a list of pairs")
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("each pair must contain two images")
        as_hwc(pair[0])
        as_hwc(pair[1])
    return pairs


def _move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in dict(batch).items()}


def _device(model):
    params = list(model.parameters()) if hasattr(model, "parameters") else []
    if params:
        return params[0].device
    return torch.device("cpu")


def _tensor_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
    return np.asarray(value)


def _numpy_match(item):
    keypoints0 = np.asarray(_tensor_numpy(item["keypoints0"]), dtype=np.float32).reshape(-1, 2)
    keypoints1 = np.asarray(_tensor_numpy(item["keypoints1"]), dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(_tensor_numpy(item["matching_scores"]), dtype=np.float32).reshape(-1)
    if not (len(keypoints0) == len(keypoints1) == len(scores)):
        raise ValueError("keypoints and matching_scores must align")
    return {"keypoints0": keypoints0, "keypoints1": keypoints1, "scores": scores}


def match_pairs(processor, model, images, threshold=0.2, batch_size=4):
    if not 0 <= float(threshold) <= 1:
        raise ValueError("threshold must be in [0, 1]")
    step = int(batch_size)
    if step < 1:
        raise ValueError("batch_size must be >= 1")
    pairs = _as_pairs(images)
    device = _device(model)
    out = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), step):
            chunk = [list(pair) for pair in pairs[start : start + step]]
            sizes = [[image_hw(left), image_hw(right)] for left, right in chunk]
            batch = _move(processor(chunk, return_tensors="pt"), device)
            decoded = processor.post_process_keypoint_matching(
                model(**batch), sizes, threshold=float(threshold)
            )
            if len(decoded) != len(chunk):
                raise ValueError("processor returned a different number of pairs than the batch")
            out.extend(_numpy_match(item) for item in decoded)
    return out


def match_pair(processor, model, image0, image1, threshold=0.2):
    return match_pairs(processor, model, [(image0, image1)], threshold=threshold, batch_size=1)[0]
