import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import eval as eval_module
import train
from dataset.folds import ensure_folds, fingerprint


class FakeDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.folds = pd.DataFrame({"image_id": ["a", "b"], "vehicle_id": [1, 2], "fold": [0, 1]})
        self.label_map = {1: 0, 2: 1}
        self.num_classes = 2
        self.saved = None

    def prepare_data(self):
        self.prepared = True

    def setup(self, stage=None):
        self.stage = stage

    def save_split(self, path):
        self.saved = path
        Path(path).mkdir(parents=True, exist_ok=True)


class FakeCheckpoint:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.best_model_path = "best.ckpt"
        self.last_model_path = "last.ckpt"


class FakeTrainer:
    instances = []
    global_zero = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_global_zero = self.global_zero
        self.fit_args = None
        self.instances.append(self)

    def fit(self, model, datamodule=None, ckpt_path=None):
        self.fit_args = (model, datamodule, ckpt_path)


def test_train_main_all_paths(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ReIDDataModule", FakeDataModule)
    monkeypatch.setattr(train, "ReIDModule", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(train, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(train, "CSVLogger", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(train.L, "Trainer", FakeTrainer)
    monkeypatch.setattr(train.L, "seed_everything", lambda *args, **kwargs: None)
    cfg.output_dir = str(tmp_path / "run")
    cfg.trainer.accelerator = "cpu"
    cfg.trainer.devices = 2
    cfg.trainer.strategy = "auto"
    cfg.resume = None
    train.main.__wrapped__(cfg)
    assert FakeTrainer.instances[-1].kwargs["strategy"] == "ddp_find_unused_parameters_true"
    assert FakeTrainer.instances[-1].fit_args[2] is None

    dm = FakeDataModule(cfg)
    resume = tmp_path / "resume.ckpt"
    torch.save(
        {"data_fingerprint": fingerprint(dm.folds), "label_map": dm.label_map},
        resume,
    )
    cfg.resume = str(resume)
    cfg.trainer.devices = 1
    cfg.trainer.limit_val_batches = 1.0
    FakeTrainer.global_zero = False
    train.main.__wrapped__(cfg)
    assert FakeTrainer.instances[-1].fit_args[2] == str(resume)

    torch.save({"data_fingerprint": "wrong", "label_map": dm.label_map}, resume)
    with pytest.raises(ValueError, match="unsafe resume"):
        train.main.__wrapped__(cfg)
    with pytest.raises(TypeError, match="mapping"):
        train.container_dict(OmegaConf.create([1]))


class LoadedModel(nn.Module):
    def __init__(self, cfg, initialize_pretrained=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.loaded = None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.loaded = (state_dict, strict)
        return SimpleNamespace()


def checkpoint_for(cfg):
    return {
        "hyper_parameters": {"cfg": OmegaConf.to_container(cfg, resolve=True)},
        "state_dict": {"model.weight": torch.ones(1), "other": torch.zeros(1)},
        "validation_weights": "raw",
    }


def test_eval_load_model_choices(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(eval_module, "ReIDModel", LoadedModel)
    path = tmp_path / "model.ckpt"
    cfg.checkpoint = None
    with pytest.raises(ValueError, match="checkpoint"):
        eval_module.load_model(cfg)

    checkpoint = checkpoint_for(cfg)
    torch.save(checkpoint, path)
    cfg.checkpoint = str(path)
    cfg.eval.weights = "auto"
    model, _, _, choice = eval_module.load_model(cfg)
    assert choice == "raw"
    assert list(model.loaded[0]) == ["weight"]

    cfg.eval.weights = "ema"
    with pytest.raises(ValueError, match="absent"):
        eval_module.load_model(cfg)
    checkpoint["ema"] = {"shadow": {"weight": torch.ones(1)}}
    torch.save(checkpoint, path)
    model, _, _, choice = eval_module.load_model(cfg)
    assert choice == "ema"
    assert list(model.loaded[0]) == ["weight"]

    cfg.eval.weights = "invalid"
    with pytest.raises(ValueError, match="auto/raw/ema"):
        eval_module.load_model(cfg)


class EvalModel(nn.Module):
    def forward(self, image):
        return image


def fake_embeddings(model, loader, cfg, device):
    size = len(loader.dataset)
    values = np.arange(size * 8, dtype=np.float32).reshape(size, 8) + 1
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_eval_main_val_test_and_guards(data_cfg, tmp_path, monkeypatch):
    folds = ensure_folds(data_cfg)
    checkpoint = {"data_fingerprint": fingerprint(folds)}
    monkeypatch.setattr(
        eval_module,
        "load_model",
        lambda cfg: (EvalModel(), cfg, checkpoint, "raw"),
    )
    monkeypatch.setattr(eval_module, "embed_loader", fake_embeddings)
    monkeypatch.setattr(eval_module.L, "seed_everything", lambda *args, **kwargs: None)
    data_cfg.eval.device = "cpu"
    data_cfg.eval.output_dir = str(tmp_path / "val")
    data_cfg.eval.top_k = 2
    data_cfg.eval.tta.enabled = True
    data_cfg.eval.tta.context_pcts = [0, 10]
    data_cfg.eval.split = "val"
    eval_module.main.__wrapped__(data_cfg)
    metadata = json.loads((tmp_path / "val/metrics.json").read_text())
    assert metadata["n_query"] == 2
    assert "metrics" in metadata

    data_cfg.eval.split = "test"
    data_cfg.eval.output_dir = str(tmp_path / "test")
    data_cfg.eval.save_distances = False
    eval_module.main.__wrapped__(data_cfg)
    assert (tmp_path / "test/submission.csv").is_file()
    assert not (tmp_path / "test/distances.npy").exists()

    gallery = pd.read_csv(data_cfg.data.query_csv)
    gallery.to_csv(data_cfg.data.gallery_csv, index=False)
    with pytest.raises(ValueError, match="overlap"):
        eval_module.main.__wrapped__(data_cfg)

    data_cfg.eval.split = "invalid"
    with pytest.raises(ValueError, match="val/test"):
        eval_module.main.__wrapped__(data_cfg)

    data_cfg.eval.split = "val"
    checkpoint["data_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="differs"):
        eval_module.main.__wrapped__(data_cfg)
    torch.use_deterministic_algorithms(False)


def test_entrypoint_main_guards(monkeypatch):
    called = []

    def decorator(**kwargs):
        def wrap(function):
            return lambda: called.append(function.__module__)

        return wrap

    monkeypatch.setattr("hydra.main", decorator)
    runpy.run_module("train", run_name="__main__")
    runpy.run_module("eval", run_name="__main__")
    assert called == ["__main__", "__main__"]
