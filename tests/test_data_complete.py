import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DistributedSampler, RandomSampler

from augmentations.pipeline import Pipeline, build_transforms
from dataset.datamodule import ReIDDataModule
from dataset.folds import ensure_folds, make_folds, query_gallery_split, read_annotations
from dataset.images import VehicleDataset, crop_bbox, image_path
from dataset.samplers import PKBatchSampler


def test_read_annotations_validation(tmp_path):
    path = tmp_path / "data.csv"
    valid = pd.DataFrame([{"image_id": "1", "x": 0, "y": 0, "w": 2, "h": 2}])
    valid.to_csv(path, index=False)
    frame = read_annotations(path)
    assert frame.camera_id.tolist() == [-1]
    assert frame.row_id.tolist() == [0]

    for bad in (
        valid.drop(columns="x"),
        valid.assign(x=np.nan),
        valid.assign(w=0),
        valid.assign(x=np.inf),
        pd.concat([valid, valid]),
    ):
        bad.to_csv(path, index=False)
        with pytest.raises(ValueError):
            read_annotations(path)
    valid.to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_annotations(path, labeled=True)


def test_make_folds_and_query_edge_cases():
    frame = pd.DataFrame(
        [
            {"image_id": str(i), "vehicle_id": i, "camera_id": 0, "x": 0, "y": 0, "w": 1, "h": 1}
            for i in range(3)
        ]
    )
    with pytest.raises(ValueError):
        make_folds(frame, group_column="camera_id")
    with pytest.raises(ValueError):
        make_folds(frame, n_folds=4)
    with pytest.raises(ValueError):
        query_gallery_split(frame)

    mixed = pd.DataFrame(
        [
            {"image_id": "single", "vehicle_id": 0, "camera_id": 0},
            {"image_id": "unknown1", "vehicle_id": 1, "camera_id": -1},
            {"image_id": "unknown2", "vehicle_id": 1, "camera_id": -1},
            {"image_id": "a", "vehicle_id": 2, "camera_id": 0},
            {"image_id": "b", "vehicle_id": 2, "camera_id": 1},
        ]
    )
    query, gallery = query_gallery_split(mixed)
    assert query.vehicle_id.tolist() == [2]
    assert set(gallery.vehicle_id) == {0, 1, 2}
    query, gallery = query_gallery_split(mixed, cross_camera=False, query_per_identity=5)
    assert len(query) == 2
    assert len(gallery) == 3


def test_ensure_folds_manifest_validation(data_cfg):
    created = ensure_folds(data_cfg)
    saved = ensure_folds(data_cfg)
    pd.testing.assert_frame_equal(created, saved)
    path = data_cfg.data.folds_file
    meta_path = Path(path).with_suffix(".json")
    original_csv = pd.read_csv(path)
    original_meta = json.loads(meta_path.read_text())

    meta_path.write_text("{}")
    with pytest.raises(ValueError, match="Stale"):
        ensure_folds(data_cfg)
    meta_path.write_text(json.dumps(original_meta))

    changed = original_csv.copy()
    changed.loc[0, "x"] += 1
    changed.to_csv(path, index=False)
    with pytest.raises(ValueError, match="rows"):
        ensure_folds(data_cfg)

    original_csv.to_csv(path, index=False)
    leaked = original_csv.copy()
    leaked.loc[leaked.vehicle_id == leaked.vehicle_id.iloc[0], "fold"] = [0, 1, 0, 1]
    leaked.to_csv(path, index=False)
    with pytest.raises(ValueError, match="leakage"):
        ensure_folds(data_cfg)

    invalid = original_csv.copy()
    invalid["fold"] = 0
    invalid.to_csv(path, index=False)
    with pytest.raises(ValueError, match="indices"):
        ensure_folds(data_cfg)


def test_images_and_vehicle_dataset(data_cfg, tmp_path, monkeypatch):
    root = tmp_path / "paths"
    root.mkdir()
    explicit = root / "a.png"
    Image.new("RGB", (4, 4)).save(explicit)
    assert image_path(root, "a.png") == explicit
    assert image_path(root, "a") == explicit
    with pytest.raises(FileNotFoundError):
        image_path(root, "missing")
    with pytest.raises(ValueError):
        crop_bbox(Image.new("RGB", (4, 4)), (0, 0, 0, 1))

    frame = read_annotations(data_cfg.data.train_csv, labeled=True).iloc[:1]

    def transform(image):
        return torch.from_numpy(image.copy()).permute(2, 0, 1)

    ds = VehicleDataset(frame, data_cfg, transform, {0: 7}, train=True)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(1.0))
    item = ds[0]
    assert len(ds) == 1
    assert item["label"] == 7
    assert item["pid"] == 0
    assert item["camera"] == 0
    assert item["index"] == 0

    data_cfg.data.verify_files = False
    frame = frame.copy()
    frame.loc[:, "image_id"] = "missing.jpg"
    VehicleDataset(frame, data_cfg, transform)
    data_cfg.data.verify_files = True
    with pytest.raises(FileNotFoundError):
        VehicleDataset(frame, data_cfg, transform)


def test_datamodule_all_loaders(data_cfg, tmp_path):
    dm = ReIDDataModule(data_cfg)
    dm.prepare_data()
    dm.setup("fit")
    dm.save_split(tmp_path / "split")
    assert (tmp_path / "split/train.csv").is_file()
    assert dm.num_classes == 8

    dm.trainer = cast(L.Trainer, SimpleNamespace(global_rank=0, world_size=1))
    assert isinstance(dm.train_dataloader().batch_sampler, PKBatchSampler)
    assert dm.val_dataloader().sampler is not None

    data_cfg.data.sampler.kind = "random"
    assert isinstance(dm.train_dataloader().sampler, RandomSampler)
    dm.trainer = cast(L.Trainer, SimpleNamespace(global_rank=1, world_size=2))
    assert isinstance(dm.train_dataloader().sampler, DistributedSampler)
    assert isinstance(dm.val_dataloader().sampler, DistributedSampler)

    data_cfg.data.sampler.kind = "invalid"
    with pytest.raises(ValueError, match="Unknown sampler"):
        dm.train_dataloader()
    dm.trainer = None
    with pytest.raises(RuntimeError):
        dm.train_dataloader()
    with pytest.raises(RuntimeError):
        dm.val_dataloader()

    data_cfg.data.fold = -1
    with pytest.raises(ValueError, match="Invalid data.fold"):
        ReIDDataModule(data_cfg).setup()


def test_loader_kwargs_workers(data_cfg):
    dm = ReIDDataModule(data_cfg)
    data_cfg.data.num_workers = 1
    kwargs = dm.loader_kwargs()
    assert kwargs["prefetch_factor"] == data_cfg.data.prefetch_factor
    assert kwargs["persistent_workers"] is False


def test_sampler_validation_and_replacement():
    with pytest.raises(ValueError):
        PKBatchSampler([0, 0, 1, 1], identities=1)
    with pytest.raises(ValueError):
        PKBatchSampler([0, 0, 1, 1], identities=2, instances=2, steps=0)
    sampler = PKBatchSampler([0, 1], identities=2, instances=3, cameras=None, steps=1)
    batch = next(iter(sampler))
    assert len(batch) == 6


def test_transform_edges(cfg, monkeypatch):
    cfg.data.resize_mode = "stretch"
    pipeline = build_transforms(cfg)
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    assert pipeline(image).shape == (3, 64, 64)

    cfg.data.resize_mode = "invalid"
    with pytest.raises(ValueError):
        build_transforms(cfg)
    cfg.data.resize_mode = "pad"

    cfg.augmentation.transforms = [
        OmegaConf.create({"name": "HorizontalFlip", "enabled": False, "p": 1.0, "params": {}}),
        OmegaConf.create({"name": "VerticalFlip", "enabled": True, "p": 2.0, "params": {}}),
    ]
    with pytest.raises(ValueError, match="probability"):
        build_transforms(cfg, train=True)

    class CallableCompose:
        def __init__(self):
            self.seed = None

        def set_random_seed(self, seed):
            self.seed = seed

        def __call__(self, **kwargs):
            return kwargs

    callable_compose = CallableCompose()
    worker = SimpleNamespace(seed=123)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: worker)
    wrapped = Pipeline(callable_compose, [lambda tensor: tensor + 1])
    np.testing.assert_array_equal(wrapped(np.zeros(1)), np.ones(1))
    assert callable_compose.seed == 123
