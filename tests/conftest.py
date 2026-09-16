import os
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from PIL import Image


def pytest_configure():
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


@pytest.fixture
def cfg():
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
    ):
        return compose(config_name="config", overrides=["experiment=smoke"])


@pytest.fixture
def data_cfg(cfg, tmp_path):
    root = tmp_path / "data"
    images = root / "images"
    images.mkdir(parents=True)
    rows = []
    for vehicle_id in range(10):
        for camera_id in range(2):
            for shot in range(2):
                image_id = f"train_{vehicle_id}_{camera_id}_{shot}"
                Image.new("RGB", (20, 16), (vehicle_id * 20, camera_id * 80, shot * 100)).save(
                    images / f"{image_id}.jpg"
                )
                rows.append(
                    {
                        "image_id": image_id,
                        "vehicle_id": vehicle_id,
                        "camera_id": camera_id,
                        "x": 1,
                        "y": 1,
                        "w": 16,
                        "h": 12,
                    }
                )
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    for split in ("test_query", "test_gallery"):
        image_id = split
        Image.new("RGB", (20, 16)).save(images / f"{image_id}.jpg")
        pd.DataFrame([{"image_id": image_id, "camera_id": 0, "x": 0, "y": 0, "w": 20, "h": 16}]).to_csv(
            root / f"{split}.csv", index=False
        )
    cfg.data.root = str(root)
    cfg.data.train_csv = str(root / "train.csv")
    cfg.data.query_csv = str(root / "test_query.csv")
    cfg.data.gallery_csv = str(root / "test_gallery.csv")
    cfg.data.image_dir = str(images)
    cfg.data.folds_file = str(tmp_path / "folds.csv")
    cfg.data.verify_files = True
    cfg.data.pin_memory = False
    cfg.data.num_workers = 0
    return cfg
