import warnings

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision import transforms as T


class PatchShuffle(A.ImageOnlyTransform):
    def __init__(self, grid=2, p=0.1):
        super().__init__(p=p)
        self.grid = int(grid)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        gh, gw = h // self.grid, w // self.grid
        patches = [
            img[y * gh : (y + 1) * gh, x * gw : (x + 1) * gw].copy()
            for y in range(self.grid)
            for x in range(self.grid)
        ]
        self.random_generator.shuffle(patches)
        result = img.copy()
        for i, patch in enumerate(patches):
            y, x = divmod(i, self.grid)
            result[y * gh : (y + 1) * gh, x * gw : (x + 1) * gw] = patch
        return result


class TorchPolicy(A.ImageOnlyTransform):
    def __init__(self, policy, p, **kwargs):
        super().__init__(p=p)
        self.policy = getattr(T, policy)(**kwargs)

    def apply(self, img, **params):
        return np.array(self.policy(Image.fromarray(img)))


class Pipeline:
    def __init__(self, compose, erasers):
        self.compose = compose
        self.erasers = erasers
        self.worker_seed = None

    def __call__(self, image):
        worker = torch.utils.data.get_worker_info()
        seed = worker.seed if worker else torch.initial_seed()
        if seed != self.worker_seed:
            self.compose.set_random_seed(seed % (2**32))
            self.worker_seed = seed
        x = self.compose(image=image)["image"]
        for op in self.erasers:
            x = op(x)
        return x


def build_transforms(cfg, train=False):
    h, w = map(int, cfg.data.image_size)
    ops: list[A.BasicTransform | A.BaseCompose]
    if cfg.data.resize_mode == "pad":
        ops = [A.LongestMaxSize(max_size_hw=(h, w)), A.PadIfNeeded(h, w, border_mode=cv2.BORDER_CONSTANT)]
    elif cfg.data.resize_mode == "stretch":
        ops = [A.Resize(h, w)]
    else:
        raise ValueError("data.resize_mode must be pad or stretch")
    erasers: list[T.RandomErasing] = []
    if train:
        for spec in cfg.augmentation.transforms:
            if not spec.enabled:
                continue
            p, name, kw = float(spec.p), spec.name, dict(spec.params)
            if not 0 <= p <= 1:
                raise ValueError(f"Invalid probability for {name}: {p}")
            if name == "RandomErasing":
                erasers.append(T.RandomErasing(p=p, **kw))
                continue
            if name == "RandomResizedCrop":
                kw["size"] = (h, w)
            if name in ("RandAugment", "AugMix"):
                op = TorchPolicy(name, p, **kw)
            elif name == "PatchShuffle":
                op = PatchShuffle(p=p, **kw)
            else:
                cls = getattr(A, name)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", UserWarning)
                    op = cls(p=p, **kw)
            ops.append(op)
    ops += [A.Normalize(mean=cfg.model.normalization.mean, std=cfg.model.normalization.std), ToTensorV2()]
    return Pipeline(A.Compose(ops, seed=int(cfg.seed)), erasers)
