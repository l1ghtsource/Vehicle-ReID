from pathlib import Path

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from augmentations import build_transforms

from .folds import ensure_folds, query_gallery_split
from .images import VehicleDataset
from .samplers import PKBatchSampler


class ReIDDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_sampler = None

    def prepare_data(self):
        ensure_folds(self.cfg)

    def setup(self, stage=None):
        cfg = self.cfg
        full = bool(cfg.data.full_retrain)
        if not full and not 0 <= cfg.data.fold < cfg.data.n_folds:
            raise ValueError("Invalid data.fold")
        self.folds = ensure_folds(cfg)
        if full:
            tr = self.folds.copy()
            va = self.folds.copy()
        else:
            tr = self.folds[self.folds.fold != cfg.data.fold].copy()
            va = self.folds[self.folds.fold == cfg.data.fold].copy()
            assert set(tr.vehicle_id).isdisjoint(va.vehicle_id)
        self.label_map = {int(pid): i for i, pid in enumerate(sorted(tr.vehicle_id.unique()))}
        self.num_classes = len(self.label_map)
        q, g = query_gallery_split(
            va,
            cfg.seed,
            **{
                "query_per_identity": cfg.data.validation.query_per_identity,
                "cross_camera": cfg.data.validation.cross_camera,
            },
        )
        self.query_frame, self.gallery_frame = q, g
        self.train_frame = tr.reset_index(drop=True)
        self.train_set = VehicleDataset(tr, cfg, build_transforms(cfg, True), self.label_map, True)
        self.val_set = VehicleDataset(pd.concat([q, g]), cfg, build_transforms(cfg))
        self.num_query = len(q)

    def loader_kwargs(self):
        d = self.cfg.data
        out = dict(
            num_workers=d.num_workers,
            pin_memory=d.pin_memory,
            persistent_workers=d.persistent_workers and d.num_workers > 0,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        if d.num_workers > 0:
            out["prefetch_factor"] = d.prefetch_factor
        return out

    def train_dataloader(self):
        c = self.cfg.data.sampler
        trainer = self.trainer
        if trainer is None:
            raise RuntimeError("train_dataloader requires an attached Trainer")
        rank, world = trainer.global_rank, trainer.world_size
        if c.kind == "pk":
            self.train_sampler = PKBatchSampler(
                self.train_frame.vehicle_id,
                self.train_frame.camera_id,
                identities=c.identities,
                instances=c.instances,
                steps=c.steps_per_epoch,
                seed=self.cfg.seed,
                rank=rank,
                world_size=world,
                camera_diverse=c.camera_diverse,
            )
            return DataLoader(self.train_set, batch_sampler=self.train_sampler, **self.loader_kwargs())
        if c.kind != "random":
            raise ValueError(f"Unknown sampler {c.kind}")
        s = (
            DistributedSampler(self.train_set, world, rank, seed=self.cfg.seed)
            if world > 1
            else RandomSampler(self.train_set)
        )
        self.train_sampler = s
        return DataLoader(
            self.train_set,
            sampler=s,
            batch_size=c.identities * c.instances,
            drop_last=True,
            **self.loader_kwargs(),
        )

    def val_dataloader(self):
        trainer = self.trainer
        if trainer is None:
            raise RuntimeError("val_dataloader requires an attached Trainer")
        s = (
            DistributedSampler(self.val_set, trainer.world_size, trainer.global_rank, shuffle=False)
            if trainer.world_size > 1
            else None
        )
        return DataLoader(
            self.val_set,
            sampler=s,
            batch_size=self.cfg.data.batch_size_eval,
            shuffle=False,
            **self.loader_kwargs(),
        )

    def save_split(self, directory):
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        self.train_frame.to_csv(p / "train.csv", index=False)
        self.query_frame.to_csv(p / "query.csv", index=False)
        self.gallery_frame.to_csv(p / "gallery.csv", index=False)
