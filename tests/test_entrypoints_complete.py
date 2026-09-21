import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

import eval as eval_module
import train
from dataset.folds import ensure_folds, fingerprint, split_fingerprint


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
    cfg.init_checkpoint = None
    train.main.__wrapped__(cfg)
    assert FakeTrainer.instances[-1].kwargs["strategy"] == "ddp_find_unused_parameters_true"
    assert FakeTrainer.instances[-1].fit_args[2] is None
    callback = FakeTrainer.instances[-1].kwargs["callbacks"][0]
    assert callback.kwargs["dirpath"] == tmp_path / "run" / "checkpoints"
    assert json.loads((tmp_path / "run/run_summary.json").read_text())["best_checkpoint"] == "best.ckpt"

    cfg.trainer.devices = 1
    cfg.trainer.limit_val_batches = 1
    FakeTrainer.global_zero = False
    train.main.__wrapped__(cfg)
    integer_limit = FakeTrainer.instances[-1].kwargs["callbacks"][0]
    assert integer_limit.kwargs["monitor"] is None
    assert integer_limit.kwargs["save_top_k"] == 0
    cfg.trainer.limit_val_batches = 1.0
    train.main.__wrapped__(cfg)
    assert FakeTrainer.instances[-1].kwargs["callbacks"][0].kwargs["monitor"] == cfg.checkpointing.monitor

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
    callback = FakeTrainer.instances[-1].kwargs["callbacks"][0]
    assert callback.kwargs["dirpath"] == resume.resolve().parent

    torch.save(
        {
            "data_fingerprint": fingerprint(dm.folds),
            "split_fingerprint": "wrong",
            "label_map": dm.label_map,
        },
        resume,
    )
    with pytest.raises(ValueError, match="unsafe resume"):
        train.main.__wrapped__(cfg)

    torch.save({"data_fingerprint": "wrong", "label_map": dm.label_map}, resume)
    with pytest.raises(ValueError, match="unsafe resume"):
        train.main.__wrapped__(cfg)
    with pytest.raises(TypeError, match="mapping"):
        train.container_dict(OmegaConf.create([1]))
    cfg.resume = str(resume)
    cfg.init_checkpoint = str(resume)
    with pytest.raises(ValueError, match="mutually exclusive"):
        train.main.__wrapped__(cfg)


def test_train_full_retrain_skips_val_monitor(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ReIDDataModule", FakeDataModule)
    monkeypatch.setattr(train, "ReIDModule", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(train, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(train, "CSVLogger", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(train.L, "Trainer", FakeTrainer)
    monkeypatch.setattr(train.L, "seed_everything", lambda *args, **kwargs: None)
    cfg.output_dir = str(tmp_path / "full")
    cfg.trainer.accelerator = "cpu"
    cfg.trainer.devices = 1
    cfg.trainer.strategy = "auto"
    cfg.resume = None
    cfg.init_checkpoint = None
    cfg.data.full_retrain = True
    cfg.data.cv_dir = None
    cfg.train.epochs = 27
    train.main.__wrapped__(cfg)
    callback = FakeTrainer.instances[-1].kwargs["callbacks"][0]
    assert FakeTrainer.instances[-1].kwargs["limit_val_batches"] == 0
    assert FakeTrainer.instances[-1].kwargs["num_sanity_val_steps"] == 0
    assert callback.kwargs["monitor"] is None
    assert callback.kwargs["save_top_k"] == 0
    assert cfg.train.epochs == 27

    cv = tmp_path / "cv"
    for fold, epoch in enumerate((25, 25, 24, 21, 12)):
        directory = cv / f"fold{fold}" / "val"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(
            json.dumps({"checkpoint": str(tmp_path / f"epoch{epoch:03d}.ckpt")})
        )
    cfg.data.cv_dir = str(cv)
    train.main.__wrapped__(cfg)
    assert cfg.train.epochs == 22
    assert FakeTrainer.instances[-1].kwargs["max_epochs"] == 22


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

    cfg.eval.weights = "raw"
    torch.save({"state_dict": {"other": torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match="Unrecognized"):
        eval_module.load_model(cfg)
    empty_raw = checkpoint_for(cfg)
    empty_raw["state_dict"] = {"other": torch.zeros(1)}
    torch.save(empty_raw, path)
    with pytest.raises(ValueError, match="model"):
        eval_module.load_model(cfg)

    lightning = checkpoint_for(cfg)
    lightning["ema"] = {"shadow": {"weight": torch.ones(1)}}
    payload = eval_module.build_serving_payload(lightning, "ema")
    assert payload["format"] == eval_module.SERVING_FORMAT
    torch.save(payload, path)
    cfg.eval.weights = "auto"
    _, _, blob, choice = eval_module.load_model(cfg)
    assert choice == "ema" and blob["format"] == eval_module.SERVING_FORMAT
    cfg.eval.weights = "raw"
    with pytest.raises(ValueError, match="requested"):
        eval_module.load_model(cfg)
    with pytest.raises(ValueError, match="already a serving"):
        eval_module.build_serving_payload(payload, "ema")
    with pytest.raises(ValueError, match="Unrecognized"):
        eval_module.build_serving_payload({"state_dict": {"weight": torch.ones(1)}}, "ema")
    unlabeled = dict(payload)
    unlabeled.pop("weights")
    torch.save(unlabeled, path)
    cfg.eval.weights = "auto"
    _, _, _, choice = eval_module.load_model(cfg)
    assert choice == "raw"
    empty_serve = dict(payload)
    empty_serve["state_dict"] = {}
    torch.save(empty_serve, path)
    cfg.eval.weights = "ema"
    with pytest.raises(ValueError, match="missing state_dict"):
        eval_module.load_model(cfg)
    torch.save(checkpoint, path)
    cfg.eval.weights = "ema"

    saved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(saved, dict):
        raise TypeError("Expected mapping checkpoint config")
    eval_cfg = saved["eval"]
    postproc_cfg = saved["postproc"]
    data_cfg = saved["data"]
    if not isinstance(eval_cfg, dict) or not isinstance(postproc_cfg, dict) or not isinstance(data_cfg, dict):
        raise TypeError("Expected mapping eval/postproc/data config")
    tta_cfg = eval_cfg["tta"]
    rerank_cfg = postproc_cfg["rerank"]
    if not isinstance(tta_cfg, dict) or not isinstance(rerank_cfg, dict):
        raise TypeError("Expected mapping tta/rerank config")
    tta_cfg["enabled"] = True
    tta_cfg["scales"] = [1.0, 1.1]
    postproc_cfg["enabled"] = True
    rerank_cfg["kind"] = "k_reciprocal"
    data_cfg["fold"] = 2
    data_cfg["folds_file"] = "/ckpt/folds.csv"
    data_cfg["root"] = "/ckpt/root"
    data_cfg["train_csv"] = "/ckpt/root/train.csv"
    data_cfg["query_csv"] = "/ckpt/root/test_query.csv"
    data_cfg["gallery_csv"] = "/elsewhere/gallery.csv"
    data_cfg["image_dir"] = "/ckpt/root/images"
    checkpoint["hyper_parameters"]["cfg"] = saved
    torch.save(checkpoint, path)
    cfg.eval.tta.enabled = False
    cfg.eval.tta.scales = [1.0]
    cfg.postproc.enabled = False
    cfg.postproc.rerank.kind = "none"
    cfg.data.root = "/custom/root"
    cfg.data.fold = 0
    cfg.data.folds_file = "/cli/folds.csv"
    cfg.eval.weights = "raw"
    cfg.refusal.kind = "threshold"
    cfg.refusal.cosine_threshold = 0.64
    _, effective, _, _ = eval_module.load_model(cfg, [])
    assert bool(effective.eval.tta.enabled) is True
    assert [float(scale) for scale in effective.eval.tta.scales] == [1.0, 1.1]
    assert bool(effective.postproc.enabled) is True
    assert str(effective.postproc.rerank.kind) == "k_reciprocal"
    assert int(effective.data.fold) == 2
    assert str(effective.data.folds_file) == "/ckpt/folds.csv"
    assert str(effective.data.root) == "/ckpt/root"
    assert str(effective.data.query_csv) == "/ckpt/root/test_query.csv"
    assert str(effective.data.gallery_csv) == "/elsewhere/gallery.csv"
    assert str(effective.refusal.kind) == "threshold"
    assert float(effective.refusal.cosine_threshold) == 0.64

    cfg.eval.tta.enabled = True
    cfg.postproc.enabled = True
    cfg.postproc.rerank.kind = "gnn"
    cfg.data.fold = 1
    cfg.data.root = "/custom/root"
    cfg.data.query_csv = "/explicit/query.csv"
    cfg.refusal.kind = "ensemble"
    cfg.refusal.rank_threshold = 0.4
    cfg.refusal.model_path = "/tmp/refuse.cbm"
    _, effective, _, _ = eval_module.load_model(
        cfg,
        [
            "eval.tta.enabled=true",
            "postproc.enabled=true",
            "postproc.rerank.kind=gnn",
            "data.fold=1",
            "data.root=/custom/root",
            "data.query_csv=/explicit/query.csv",
            "refusal.kind=ensemble",
            "refusal.rank_threshold=0.4",
            "refusal.model_path=/tmp/refuse.cbm",
            "experiment=smoke",
            "~trainer.devices",
            "+data.foo=1",
            "missing.path=1",
        ],
    )
    assert bool(effective.eval.tta.enabled) is True
    assert bool(effective.postproc.enabled) is True
    assert str(effective.postproc.rerank.kind) == "gnn"
    assert int(effective.data.fold) == 1
    assert str(effective.data.root) == "/custom/root"
    assert str(effective.data.folds_file) == "/ckpt/folds.csv"
    assert str(effective.data.train_csv) == "/custom/root/train.csv"
    assert str(effective.data.query_csv) == "/explicit/query.csv"
    assert str(effective.data.gallery_csv) == "/elsewhere/gallery.csv"
    assert str(effective.data.image_dir) == "/custom/root/images"
    assert str(effective.refusal.kind) == "ensemble"
    assert float(effective.refusal.rank_threshold) == 0.4
    assert str(effective.refusal.model_path) == "/tmp/refuse.cbm"


def test_overlay_eval_migrates_h7_junk_protocol(cfg):
    saved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    saved.data.validation.exclude_all_same_camera = True
    migrated = eval_module.overlay_eval_config(saved, cfg, [])
    assert bool(migrated.data.validation.exclude_all_same_camera) is False
    sibling = eval_module.overlay_eval_config(saved, cfg, ["data.fold=1"])
    assert bool(sibling.data.validation.exclude_all_same_camera) is False
    cfg.data.validation.exclude_all_same_camera = True
    kept = eval_module.overlay_eval_config(
        saved,
        cfg,
        ["data.validation.exclude_all_same_camera=true"],
    )
    assert bool(kept.data.validation.exclude_all_same_camera) is True
    grouped = eval_module.overlay_eval_config(
        saved,
        cfg,
        ["data.validation={exclude_all_same_camera: true}"],
    )
    assert bool(grouped.data.validation.exclude_all_same_camera) is True
    assert eval_module.protocol_overridden(
        "data.validation.exclude_all_same_camera",
        {"data.validation"},
    )
    assert not eval_module.protocol_overridden(
        "data.validation.exclude_all_same_camera",
        {"data.fold"},
    )
    stub = OmegaConf.create(
        {
            "checkpoint": "weights.pt",
            "eval": {
                "split": "val",
                "device": "cpu",
                "output_dir": "out",
                "weights": "raw",
                "top_k": 10,
                "save_distances": False,
                "precision": "fp32",
            },
        }
    )
    defaulted = eval_module.overlay_eval_config(saved, stub, [])
    assert bool(defaulted.data.validation.exclude_all_same_camera) is False


def test_overlay_eval_config_partial_runtime_cfg(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(eval_module, "ReIDModel", LoadedModel)
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint_for(cfg), path)
    stub = OmegaConf.create(
        {
            "checkpoint": str(path),
            "eval": {
                "split": "val",
                "device": "cpu",
                "output_dir": str(tmp_path),
                "weights": "raw",
                "top_k": 10,
                "save_distances": False,
                "precision": "fp32",
            },
        },
    )
    model, effective, _, choice = eval_module.load_model(stub, [])
    assert choice == "raw"
    assert list(model.loaded[0]) == ["weight"]
    assert str(effective.refusal.kind) == str(cfg.refusal.kind)
    assert bool(effective.postproc.streaming) == bool(cfg.postproc.streaming)
    overlay = eval_module.overlay_eval_config(
        OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)),
        stub,
        [],
    )
    assert overlay.eval.split == "val"
    assert overlay.eval.weights == "raw"
    assert str(overlay.refusal.kind) == str(cfg.refusal.kind)
    kept = eval_module.overlay_eval_config(
        OmegaConf.create(
            {"model": {"backend": "llm2clip", "compile": True}, "data": {"root": str(tmp_path)}}
        ),
        stub,
        [],
    )
    assert str(kept.model.backend) == "llm2clip"
    assert bool(kept.model.compile) is True
    matched = eval_module.overlay_eval_config(
        OmegaConf.create({"model": {"backend": "llm2clip", "compile": False}}),
        OmegaConf.merge(stub, {"model": {"backend": "llm2clip", "compile": True, "attn_kernel": "sdpa"}}),
        [],
    )
    assert bool(matched.model.compile) is True
    assert str(matched.model.attn_kernel) == "sdpa"
    dynamic = eval_module.overlay_eval_config(
        OmegaConf.create({"model": {"backend": "llm2clip", "compile_dynamic": True}}),
        OmegaConf.merge(
            stub,
            {"model": {"backend": "llm2clip", "compile_dynamic": False, "compile_embedding": False}},
        ),
        [],
    )
    assert bool(dynamic.model.compile_dynamic) is False
    assert bool(dynamic.model.compile_embedding) is False
    old_saved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    del old_saved.data["decode_backend"]
    with pytest.raises(AttributeError):
        _ = old_saved.data.decode_backend
    filled = eval_module.overlay_eval_config(old_saved, cfg, [])
    assert str(filled.data.decode_backend) == "pil"
    kept_backend = OmegaConf.create(OmegaConf.to_container(old_saved, resolve=True))
    kept_backend.data.decode_backend = "cv2"
    runtime = OmegaConf.merge(cfg, {"data": {"decode_backend": "jpeg"}})
    unchanged = eval_module.overlay_eval_config(kept_backend, runtime, [])
    assert str(unchanged.data.decode_backend) == "cv2"
    overridden = eval_module.overlay_eval_config(old_saved, runtime, ["data.decode_backend=jpeg"])
    assert str(overridden.data.decode_backend) == "jpeg"
    with_refusal = OmegaConf.merge(stub, {"refusal": {"kind": "none"}})
    _, refusal_cfg, _, _ = eval_module.load_model(with_refusal, [])
    assert str(refusal_cfg.refusal.kind) == "none"
    with_stream = OmegaConf.merge(stub, {"postproc": {"streaming": False}})
    _, stream_cfg, _, _ = eval_module.load_model(with_stream, [])
    assert bool(stream_cfg.postproc.streaming) is False


def test_load_compatible_state_ignores_mask_token():
    class Masked(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.mask_token = nn.Parameter(torch.zeros(1))

    model = Masked()
    eval_module.load_compatible_state(model, {"weight": torch.ones(1)})
    assert torch.equal(model.weight, torch.ones(1))
    with pytest.raises(ValueError, match="mismatch"):
        eval_module.load_compatible_state(model, {"weight": torch.ones(1), "extra": torch.ones(1)})
    assert eval_module.load_result_keys(SimpleNamespace()) == ([], [])
    assert eval_module.load_result_keys((["a"], ["b"])) == (["a"], ["b"])


def test_refusal_serving_presets():
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        none = compose(config_name="config", overrides=["experiment=smoke"])
        threshold = compose(config_name="config", overrides=["experiment=smoke", "refusal=eva02_threshold"])
        model = compose(config_name="config", overrides=["experiment=smoke", "refusal=eva02_model"])
        ensemble = compose(config_name="config", overrides=["experiment=smoke", "refusal=eva02_ensemble"])
    assert none.refusal.kind == "none"
    assert none.refusal.cosine_threshold is None
    assert threshold.refusal.kind == "threshold"
    assert float(threshold.refusal.cosine_threshold) == pytest.approx(0.6638)
    assert model.refusal.kind == "model"
    assert float(model.refusal.model_threshold) == pytest.approx(0.5540)
    assert str(model.refusal.model_path) == "weights/finetuned/eva02_catboost.cbm"
    assert ensemble.refusal.kind == "ensemble"
    assert float(ensemble.refusal.cosine_threshold) == pytest.approx(0.6638)
    assert float(ensemble.refusal.model_threshold) == pytest.approx(0.5540)
    assert ensemble.refusal.rank_threshold is None
    assert "refusal" not in none.eval


class EvalModel(nn.Module):
    def forward(self, image):
        return image


def fake_embeddings(model, cfg, device, frame):
    size = len(frame)
    values = np.arange(size * 8, dtype=np.float32).reshape(size, 8) + 1
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_eval_main_val_test_and_guards(data_cfg, tmp_path, monkeypatch):
    folds = ensure_folds(data_cfg)
    val_ids = set(folds[folds.fold == data_cfg.data.fold].vehicle_id.astype(int))
    train_ids = sorted(set(folds.vehicle_id.astype(int)) - val_ids)
    checkpoint = {
        "data_fingerprint": fingerprint(folds),
        "split_fingerprint": split_fingerprint(folds),
        "label_map": {int(pid): index for index, pid in enumerate(train_ids)},
    }
    monkeypatch.setattr(
        eval_module,
        "load_model",
        lambda cfg: (EvalModel(), cfg, checkpoint, "raw"),
    )
    monkeypatch.setattr(eval_module, "embed_frame", fake_embeddings)
    monkeypatch.setattr(eval_module.L, "seed_everything", lambda *args, **kwargs: None)
    data_cfg.eval.device = "cpu"
    data_cfg.eval.output_dir = str(tmp_path / "val")
    data_cfg.eval.top_k = 2
    data_cfg.eval.tta.enabled = True
    data_cfg.eval.tta.context_pcts = [0, 10]
    data_cfg.eval.split = "val"
    data_cfg.refusal.kind = None
    data_cfg.refusal.k = None
    data_cfg.refusal.with_embeddings = None
    eval_module.main.__wrapped__(data_cfg)
    metadata = json.loads((tmp_path / "val/metrics.json").read_text())
    assert metadata["n_query"] == 2
    assert "metrics" in metadata
    assert metadata["refusal"]["kind"] == "none"
    assert metadata["refusal"]["n_accept"] == metadata["n_query"]
    assert metadata["refusal"]["n_refuse"] == 0
    oof = pd.read_csv(tmp_path / "val/oof.csv")
    assert len(oof) == len(folds[folds.fold == data_cfg.data.fold])
    assert np.load(tmp_path / "val/embeddings.npy").shape[0] == len(oof)

    data_cfg.refusal.kind = "threshold"
    data_cfg.refusal.cosine_threshold = 2.0
    data_cfg.eval.output_dir = str(tmp_path / "val_refuse")
    eval_module.main.__wrapped__(data_cfg)
    refused = pd.read_csv(tmp_path / "val_refuse/candidates.csv")
    kept = pd.read_csv(tmp_path / "val_refuse/submission.csv")
    refuse_meta = json.loads((tmp_path / "val_refuse/metrics.json").read_text())
    assert refused.empty
    assert len(kept) == refuse_meta["n_query"]
    assert "confidence" not in kept.columns
    assert list(kept.columns)[:1] == ["query_id"]
    assert refuse_meta["refusal"]["kind"] == "threshold"
    assert refuse_meta["refusal"]["n_refuse"] == refuse_meta["n_query"]
    data_cfg.refusal.kind = "model"
    data_cfg.refusal.model_threshold = 0.5
    with pytest.raises(ValueError, match="model_path"):
        eval_module.main.__wrapped__(data_cfg)
    data_cfg.refusal.kind = "off"
    data_cfg.refusal.cosine_threshold = None
    data_cfg.eval.output_dir = str(tmp_path / "val_off")
    eval_module.main.__wrapped__(data_cfg)
    off_meta = json.loads((tmp_path / "val_off/metrics.json").read_text())
    assert off_meta["refusal"]["kind"] == "none"
    assert off_meta["refusal"]["n_refuse"] == 0
    data_cfg.refusal.kind = "none"

    data_cfg.eval.split = "test"
    data_cfg.eval.output_dir = str(tmp_path / "test")
    data_cfg.eval.save_distances = False
    eval_module.main.__wrapped__(data_cfg)
    assert (tmp_path / "test/submission.csv").is_file()
    cand = pd.read_csv(tmp_path / "test/candidates.csv")
    assert list(cand.columns) == ["query_id", "gallery_id", "confidence"]
    assert not cand.empty
    assert not (tmp_path / "test/distances.npy").exists()
    query_src = pd.read_csv(data_cfg.data.query_csv, dtype={"image_id": str})
    gallery_src = pd.read_csv(data_cfg.data.gallery_csv, dtype={"image_id": str})
    order = pd.read_csv(tmp_path / "test/embedding_order.csv", dtype={"image_id": str})
    emb = np.load(tmp_path / "test/embeddings.npy")
    assert order.image_id.tolist() == query_src.image_id.tolist() + gallery_src.image_id.tolist()
    assert emb.shape == (len(order), 8) and emb.dtype == np.float32

    gallery = pd.read_csv(data_cfg.data.query_csv)
    gallery.to_csv(data_cfg.data.gallery_csv, index=False)
    with pytest.raises(ValueError, match="overlap"):
        eval_module.main.__wrapped__(data_cfg)

    data_cfg.eval.split = "invalid"
    with pytest.raises(ValueError, match="val/test"):
        eval_module.main.__wrapped__(data_cfg)

    data_cfg.eval.split = "val"
    checkpoint["label_map"] = {}
    with pytest.raises(ValueError, match="label_map"):
        eval_module.main.__wrapped__(data_cfg)
    checkpoint["label_map"] = {int(pid): 0 for pid in val_ids}
    with pytest.raises(ValueError, match="overlap"):
        eval_module.main.__wrapped__(data_cfg)
    checkpoint["label_map"] = {int(pid): index for index, pid in enumerate(train_ids)}
    permuted = folds.copy()
    permuted["fold"] = (permuted.fold + 1) % (int(permuted.fold.max()) + 1)
    permuted.to_csv(data_cfg.data.folds_file, index=False)
    with pytest.raises(ValueError, match="fold assignment"):
        eval_module.main.__wrapped__(data_cfg)
    folds.to_csv(data_cfg.data.folds_file, index=False)
    checkpoint["data_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="differs"):
        eval_module.main.__wrapped__(data_cfg)
    torch.use_deterministic_algorithms(False)


def test_eval_override_helpers(monkeypatch):
    assert eval_module.task_overrides() == []
    assert eval_module.override_key("") is None
    assert eval_module.override_key("??") is None
    assert eval_module.override_key("~data.fold") is None
    assert eval_module.override_key("experiment=smoke") is None
    assert eval_module.override_key("@eval.tta.enabled=true") is None
    assert eval_module.override_key("+data.fold=2") == "data.fold"
    assert eval_module.override_key("++eval.tta.enabled=true") == "eval.tta.enabled"

    class FakeHydra:
        @staticmethod
        def initialized():
            return True

        @staticmethod
        def get():
            return SimpleNamespace(overrides=SimpleNamespace(task=["eval.tta.enabled=true"]))

    monkeypatch.setattr(eval_module, "HydraConfig", FakeHydra)
    assert eval_module.task_overrides() == ["eval.tta.enabled=true"]

    class EmptyHydra:
        @staticmethod
        def initialized():
            return True

        @staticmethod
        def get():
            return SimpleNamespace(overrides=SimpleNamespace(task=None))

    monkeypatch.setattr(eval_module, "HydraConfig", EmptyHydra)
    assert eval_module.task_overrides() == []


def test_eval_cli_force_add_and_data_root(data_cfg, tmp_path):
    old_root = "/old-training-host/data"
    saved = OmegaConf.to_container(data_cfg, resolve=True)
    if not isinstance(saved, dict):
        raise TypeError("Expected mapping checkpoint config")
    data = saved["data"]
    eval_cfg = saved["eval"]
    if not isinstance(data, dict) or not isinstance(eval_cfg, dict):
        raise TypeError("Expected mapping data/eval config")
    tta_cfg = eval_cfg["tta"]
    if not isinstance(tta_cfg, dict):
        raise TypeError("Expected mapping tta config")
    data["root"] = old_root
    data["train_csv"] = f"{old_root}/train.csv"
    data["val_source_csv"] = f"{old_root}/train.csv"
    data["query_csv"] = f"{old_root}/test_query.csv"
    data["gallery_csv"] = f"{old_root}/test_gallery.csv"
    data["image_dir"] = f"{old_root}/images"
    tta_cfg["enabled"] = False
    checkpoint = tmp_path / "model.ckpt"
    model = eval_module.ReIDModel(data_cfg, initialize_pretrained=False)
    torch.save(
        {
            "hyper_parameters": {"cfg": saved},
            "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
            "validation_weights": "raw",
        },
        checkpoint,
    )
    out = tmp_path / "eval_out"
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(project / "eval.py"),
            f"checkpoint={checkpoint}",
            f"data.root={data_cfg.data.root}",
            "++eval.tta.enabled=true",
            "eval.split=test",
            "eval.device=cpu",
            "eval.weights=raw",
            "eval.save_distances=false",
            f"eval.output_dir={out}",
            f"output_dir={tmp_path / 'run'}",
            "experiment=smoke",
            "trainer.deterministic=false",
            "data.num_workers=0",
            "data.pin_memory=false",
        ],
        cwd=project,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    written = OmegaConf.load(out / "config.yaml")
    assert bool(written.eval.tta.enabled) is True
    assert str(written.data.root) == str(data_cfg.data.root)
    assert str(written.data.query_csv) == str(Path(data_cfg.data.root) / "test_query.csv")
    assert str(written.data.gallery_csv) == str(Path(data_cfg.data.root) / "test_gallery.csv")
    assert str(written.data.image_dir) == str(Path(data_cfg.data.root) / "images")
    assert str(written.data.train_csv) == str(Path(data_cfg.data.root) / "train.csv")
    assert str(written.data.val_source_csv) == str(Path(data_cfg.data.root) / "train.csv")
    assert not str(written.data.query_csv).startswith(old_root)
    assert (out / "submission.csv").is_file()
    cand = pd.read_csv(out / "candidates.csv")
    assert list(cand.columns) == ["query_id", "gallery_id", "confidence"]


def test_entrypoint_main_guards(monkeypatch):
    called = []

    def decorator(**kwargs):
        def wrap(function):
            return lambda: called.append(function.__module__)

        return wrap

    monkeypatch.setattr("hydra.main", decorator)
    runpy.run_module("train", run_name="__main__")
    runpy.run_module("eval", run_name="__main__")
    runpy.run_module("pretrain", run_name="__main__")
    assert called == ["__main__", "__main__", "__main__"]
