import math

import numpy as np
from torch.utils.data import Sampler


class PKBatchSampler(Sampler):
    def __init__(
        self,
        labels,
        cameras=None,
        identities=16,
        instances=4,
        steps=None,
        seed=42,
        rank=0,
        world_size=1,
        camera_diverse=True,
    ):
        self.labels = np.asarray(labels)
        self.cameras = np.asarray(cameras) if cameras is not None else None
        self.p, self.k = int(identities), int(instances)
        self.seed, self.rank, self.world_size, self.epoch = seed, rank, world_size, 0
        self.camera_diverse = camera_diverse
        self.groups = {pid: np.flatnonzero(self.labels == pid) for pid in np.unique(self.labels)}
        self.pids = np.asarray(list(self.groups))
        if self.p < 2 or self.k < 2 or len(self.pids) < self.p:
            raise ValueError("PK requires P>=2, K>=2, at least P distinct training identities")
        self.steps = (
            int(steps) if steps is not None else math.ceil(len(labels) / (self.p * self.k * world_size))
        )
        if self.steps < 1:
            raise ValueError("steps_per_epoch must be positive")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 100003 + self.rank * 1009)
        for _ in range(self.steps):
            batch = []
            for pid in rng.choice(self.pids, self.p, replace=False):
                pool = self.groups[pid]
                selected = []
                if self.camera_diverse and self.cameras is not None:
                    cams = rng.permutation(np.unique(self.cameras[pool]))
                    for cam in cams[: self.k]:
                        selected.append(int(rng.choice(pool[self.cameras[pool] == cam])))
                remaining = np.setdiff1d(pool, selected)
                count = self.k - len(selected)
                if count:
                    choices = remaining if len(remaining) else pool
                    selected.extend(rng.choice(choices, count, replace=len(choices) < count).tolist())
                batch.extend(selected)
            yield batch
