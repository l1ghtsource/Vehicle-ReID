import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from augmentations import build_transforms
from dataset.folds import make_folds, query_gallery_split
from dataset.images import crop_bbox
from dataset.samplers import PKBatchSampler


def example():
    return pd.DataFrame(
        [
            dict(image_id=f"{p}_{i}", vehicle_id=p, camera_id=i % 2, x=0, y=0, w=10, h=10)
            for p in range(20)
            for i in range(4)
        ]
    )


def test_folds_and_protocol():
    df = example()
    a = make_folds(df)
    pd.testing.assert_frame_equal(a, make_folds(df))
    assert a.groupby("vehicle_id").fold.nunique().max() == 1
    assert set(a.fold) == set(range(5))
    for f in range(5):
        tr, va = a[a.fold != f], a[a.fold == f]
        assert set(tr.vehicle_id).isdisjoint(va.vehicle_id)
        q, g = query_gallery_split(va)
        assert set(q.image_id).isdisjoint(g.image_id)
        for row in q.itertuples():
            assert ((g.vehicle_id == row.vehicle_id) & (g.camera_id != row.camera_id)).any()


def test_crop_context():
    im = Image.new("RGB", (100, 80))
    assert crop_bbox(im, (20, 20, 40, 20), 10).size == (48, 24)
    assert crop_bbox(im, (0, 0, 40, 20), 10).size == (44, 22)
    with pytest.raises(ValueError):
        crop_bbox(im, (200, 0, 10, 10))


def test_pk_epoch_rank_balance():
    df = example()
    a = PKBatchSampler(df.vehicle_id, df.camera_id, 4, 4, steps=3)
    first = list(a)
    assert list(a) == first
    for indices in first:
        labels = df.iloc[indices].vehicle_id.to_numpy().reshape(4, 4)
        assert np.all(labels == labels[:, 0:1])
        assert len(np.unique(labels)) == 4
        for group in np.array(indices).reshape(4, 4):
            assert df.iloc[group].camera_id.nunique() == 2
    a.set_epoch(1)
    assert list(a) != first
    b = PKBatchSampler(df.vehicle_id, df.camera_id, 4, 4, steps=3, rank=1, world_size=2)
    assert list(b) != first


def test_all_augmentations(cfg):

    original = cfg.augmentation.transforms
    image = np.random.default_rng(1).integers(0, 256, (73, 111, 3), dtype=np.uint8)
    for spec in original:
        s = OmegaConf.to_container(spec)
        assert isinstance(s, dict)
        s.update(enabled=True, p=1.0)
        cfg.augmentation.transforms = [s]
        pipeline = build_transforms(cfg, True)
        x = pipeline(image)
        assert x.shape == (3, 64, 64), spec.name
        assert torch.isfinite(x).all(), spec.name
