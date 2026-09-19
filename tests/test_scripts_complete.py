import json
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import models
import scripts.aggregate_cv as aggregate_cv
import scripts.audit_data as audit_data
import scripts.check_backbone as check_backbone
import scripts.download_weights as download_weights
import scripts.export_refusal as export_refusal
import scripts.export_serving as export_serving
import scripts.prepare_folds as prepare_folds
import scripts.zero_shot as zero_shot


def test_train_folds_forwards_overrides_to_eval():
    script = (Path(__file__).resolve().parents[1] / "scripts/train_folds.sh").read_text()
    assert "Exactly five GPU IDs are required" in script
    eval_block = script.split('"$PYTHON" eval.py', 1)[1].split("fold_dir/eval.log", 1)[0]
    assert '"$@"' in eval_block


def test_zero_shot_weights_script_runs_all_local_backbones():
    script = (Path(__file__).resolve().parents[1] / "scripts/zero_shot_weights.sh").read_text()
    assert "eval.device=$DEVICE" in script
    assert "dinov3_convnext_base" in script
    assert "dinov3_convnext_large" in script
    assert "radio" in script
    assert "llm2clip" in script
    assert "summary.json" in script


def test_pretrain_script_selects_model_data_and_recipe():
    script = (Path(__file__).resolve().parents[1] / "scripts/pretrain.sh").read_text()
    assert "experiment defaults to current_best_tuned" in script
    assert 'EXPERIMENT="${EXPERIMENT:-current_best_tuned}"' in script
    assert '"pretrain.datasets=[$DATASETS]"' in script
    assert '"model=$MODEL"' in script
    assert "CUDA_VISIBLE_DEVICES" in script
    assert "data.image_size=[336,336]" in script
    assert "model.head.local_parts=0" in script
    assert "test_train_ssl" in script
    assert "test_train_ssl_crop" in script
    assert "test_train_ssl_full" in script
    assert '"loss=dino"' in script
    assert '"$@"' in script.split('"$PYTHON" pretrain.py', 1)[1]


def test_aggregate_cv_main_and_guard(tmp_path, monkeypatch):
    first = tmp_path / "fold0.json"
    second = tmp_path / "fold1.json"
    for path, fold, score, queries in ((first, 0, 0.5, 1), (second, 1, 1.0, 3)):
        path.write_text(
            json.dumps(
                {
                    "fold": fold,
                    "metrics": {
                        "mAP": score,
                        "evaluated_queries": queries,
                        "queries_without_positive": 0,
                    },
                }
            )
        )
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["aggregate_cv", str(first), str(second), "--output", str(output)],
    )
    aggregate_cv.main()
    report = json.loads(output.read_text())
    assert report["metrics"]["mAP"]["mean"] == 0.75
    assert report["metrics"]["mAP"]["query_weighted_mean"] == 0.875

    monkeypatch.setattr(sys, "argv", ["aggregate_cv", str(first), str(first)])
    with pytest.raises(ValueError, match="Duplicate folds"):
        aggregate_cv.main()

    with pytest.raises(ValueError, match="Cannot read stop epoch"):
        aggregate_cv.checkpoint_epoch(tmp_path / "last.ckpt")
    with pytest.raises(ValueError, match="n_folds"):
        aggregate_cv.mean_stop_epochs(tmp_path, n_folds=0)
    cv = tmp_path / "cv"
    for fold, epoch in enumerate((25, 25, 24, 21, 12)):
        directory = cv / f"fold{fold}" / "val"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(
            json.dumps({"checkpoint": f"/ckpt/fold{fold}/epoch{epoch:03d}.ckpt"})
        )
    assert aggregate_cv.mean_stop_epochs(cv) == 21
    assert aggregate_cv.checkpoint_epoch(Path("epoch000.ckpt")) == 0

    monkeypatch.setattr(
        sys,
        "argv",
        ["aggregate_cv", str(first), str(second), "--output", str(output)],
    )
    runpy.run_module("scripts.aggregate_cv", run_name="__main__")


def test_audit_data_main(data_cfg, tmp_path, monkeypatch):
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_data",
            "--root",
            str(data_cfg.data.root),
            "--output",
            str(output),
            "--check-images",
        ],
    )
    audit_data.main()
    report = json.loads(output.read_text())
    assert report["train"]["rows"] == 40
    assert report["train"]["identities"] == 10
    runpy.run_module("scripts.audit_data", run_name="__main__")


def test_download_weights_all_models(tmp_path, monkeypatch):
    snapshots = []
    files = []
    monkeypatch.setattr(
        download_weights,
        "snapshot_download",
        lambda *args, **kwargs: snapshots.append((args, kwargs)) or "snapshot",
    )
    monkeypatch.setattr(
        download_weights,
        "hf_hub_download",
        lambda *args, **kwargs: files.append((args, kwargs)) or "file",
    )
    monkeypatch.setattr(download_weights, "verify_digests", lambda *args, **kwargs: None)
    for name in ("dinov3_base", "dinov3_large", "radio", "llm2clip", "efficientloftr"):
        monkeypatch.setattr(
            sys,
            "argv",
            ["download_weights", name, "--directory", str(tmp_path)],
        )
        download_weights.main()
    assert len(snapshots) == 3
    assert len(files) == 2
    assert "revision" not in snapshots[0][1]
    radio = download_weights.MODELS["radio"]
    assert radio["kind"] == "file"
    assert files[0][1]["revision"] == radio["revision"]

    monkeypatch.setattr(
        sys,
        "argv",
        ["download_weights", "radio", "--directory", str(tmp_path)],
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: "file")
    monkeypatch.setattr("scripts.verify_weights.verify_digests", lambda *args, **kwargs: None)
    runpy.run_module("scripts.download_weights", run_name="__main__")


class SmokeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def cuda(self, device=None):
        return self

    def forward(self, x):
        raw = self.weight.expand(len(x), 4)
        return {"raw": raw, "embedding": torch.nn.functional.normalize(raw, dim=1)}


class BadSmokeModel(SmokeModel):
    def forward(self, x):
        raw = self.weight.expand(len(x), 4) * torch.nan
        return {"raw": raw, "embedding": raw}


def test_check_backbone_main(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(check_backbone, "ReIDModel", SmokeModel)
    original_randn = torch.randn
    monkeypatch.setattr(
        check_backbone.torch,
        "randn",
        lambda *shape, **kwargs: original_randn(*shape),
    )
    monkeypatch.setattr(check_backbone.torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(check_backbone.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(sys, "argv", ["check_backbone", "llm2clip", "--backward"])
    check_backbone.main()
    result = json.loads(Path("artifacts/backbone_checks.json").read_text())
    assert result[0]["backward"] is True
    assert result[0]["shape"] == [2, 4]

    monkeypatch.setattr(sys, "argv", ["check_backbone", "convnext_tiny"])
    check_backbone.main()

    monkeypatch.setattr(check_backbone, "ReIDModel", BadSmokeModel)
    monkeypatch.setattr(sys, "argv", ["check_backbone", "convnext_tiny", "--backward"])
    with pytest.raises(FloatingPointError):
        check_backbone.main()

    monkeypatch.setattr(models, "ReIDModel", SmokeModel)
    monkeypatch.setattr(sys, "argv", ["check_backbone", "convnext_tiny"])
    runpy.run_module("scripts.check_backbone", run_name="__main__")


class ProbeModel(nn.Module):
    def __init__(self, cfg, initialize_pretrained=True):
        super().__init__()
        self.cfg = cfg
        self.initialized = initialize_pretrained
        self.weight = nn.Parameter(torch.ones(1))


def fake_zero_shot_embeddings(model, loader, cfg, device):
    size = len(loader.dataset)
    values = np.arange(size * 8, dtype=np.float32).reshape(size, 8) + 1
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_zero_shot_main(data_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(zero_shot, "ReIDModel", ProbeModel)
    monkeypatch.setattr(zero_shot, "embed_loader", fake_zero_shot_embeddings)
    monkeypatch.setattr(zero_shot.L, "seed_everything", lambda *args, **kwargs: None)
    out = tmp_path / "probe"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "zero_shot",
            "convnext_tiny",
            f"data.root={data_cfg.data.root}",
            f"data.train_csv={data_cfg.data.train_csv}",
            f"data.image_dir={data_cfg.data.image_dir}",
            "eval.device=cpu",
            f"eval.output_dir={out}",
            "eval.tta.enabled=true",
            "eval.tta.context_pcts=[0,10]",
            "eval.save_distances=true",
            "data.num_workers=0",
            "data.pin_memory=false",
            "trainer.deterministic=false",
            "model.pretrained=false",
        ],
    )
    zero_shot.main()
    metadata = json.loads((out / "metrics.json").read_text())
    assert metadata["model"] == "convnext_tiny"
    assert metadata["pretrained"] is False
    assert "metrics" in metadata
    assert metadata["n_train"] == 40
    assert (out / "distances.npy").is_file()
    saved = OmegaConf.load(out / "config.yaml")
    assert str(saved.model.pooling.kind) == "gap"
    assert bool(saved.model.head.bnneck) is False

    second = tmp_path / "probe_default"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "zero_shot",
            "convnext_tiny",
            f"data.root={data_cfg.data.root}",
            f"data.train_csv={data_cfg.data.train_csv}",
            f"data.image_dir={data_cfg.data.image_dir}",
            "eval.device=cpu",
            f"eval.output_dir={second}",
            "data.num_workers=0",
            "data.pin_memory=false",
            "trainer.deterministic=false",
            "model.pretrained=false",
        ],
    )
    zero_shot.main()
    assert not (second / "distances.npy").exists()
    torch.use_deterministic_algorithms(False)

    monkeypatch.setattr(models, "ReIDModel", ProbeModel)
    monkeypatch.setattr("modules.inference.embed_loader", fake_zero_shot_embeddings)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "zero_shot",
            "convnext_tiny",
            f"data.root={data_cfg.data.root}",
            f"data.train_csv={data_cfg.data.train_csv}",
            f"data.image_dir={data_cfg.data.image_dir}",
            "eval.device=cpu",
            f"eval.output_dir={tmp_path / 'probe_main'}",
            "data.num_workers=0",
            "data.pin_memory=false",
            "trainer.deterministic=false",
            "model.pretrained=false",
        ],
    )
    runpy.run_module("scripts.zero_shot", run_name="__main__")


def test_prepare_folds_body_and_main_guard(data_cfg, monkeypatch):
    prepare_folds.main.__wrapped__(data_cfg)
    called = []

    def decorator(**kwargs):
        def wrap(function):
            return lambda: called.append(function.__name__)

        return wrap

    monkeypatch.setattr("hydra.main", decorator)
    runpy.run_module("scripts.prepare_folds", run_name="__main__")
    assert called == ["main"]


def test_export_serving_payload(tmp_path, monkeypatch):
    source = tmp_path / "train.ckpt"
    target = tmp_path / "out" / "eva02.pt"
    blob = {
        "hyper_parameters": {"cfg": {"seed": 1}},
        "state_dict": {"model.weight": torch.ones(2), "skip": torch.zeros(1)},
        "ema": {"shadow": {"weight": torch.ones(2) * 2}},
        "validation_weights": "ema",
        "label_map": {1: 0},
    }
    torch.save(blob, source)
    written = export_serving.export_serving(source, target, "ema")
    loaded = torch.load(written, map_location="cpu", weights_only=False)
    assert loaded["format"] == "reid-serving"
    assert torch.equal(loaded["state_dict"]["weight"], torch.ones(2) * 2)
    raw = export_serving.export_serving(source, tmp_path / "raw.pt", "raw")
    raw_blob = torch.load(raw, map_location="cpu", weights_only=False)
    assert torch.equal(raw_blob["state_dict"]["weight"], torch.ones(2))
    with pytest.raises(ValueError, match="already a serving"):
        export_serving.export_serving(written, tmp_path / "again.pt", "ema")
    with pytest.raises(FileNotFoundError):
        export_serving.source_checkpoint(tmp_path / "missing.ckpt")
    monkeypatch.setattr(export_serving, "DEFAULT_SUMMARY", tmp_path / "missing-summary.json")
    monkeypatch.setattr(export_serving, "DEFAULT_METRICS", tmp_path / "missing.json")
    with pytest.raises(ValueError, match="Pass --checkpoint"):
        export_serving.source_checkpoint(None)
    with pytest.raises(ValueError, match="No checkpoint path"):
        export_serving.listed_checkpoint({})
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"checkpoint": str(tmp_path / "absent.ckpt")}))
    monkeypatch.setattr(export_serving, "DEFAULT_METRICS", metrics)
    with pytest.raises(FileNotFoundError):
        export_serving.source_checkpoint(None)
    summary = tmp_path / "run_summary.json"
    summary.write_text(json.dumps({"last_checkpoint": str(source), "best_checkpoint": None}))
    monkeypatch.setattr(export_serving, "DEFAULT_SUMMARY", summary)
    assert export_serving.source_checkpoint(None) == source
    monkeypatch.setattr(export_serving, "DEFAULT_SUMMARY", tmp_path / "missing-summary.json")
    metrics.write_text(json.dumps({"checkpoint": str(source)}))
    assert export_serving.source_checkpoint(None) == source
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_serving",
            "--checkpoint",
            str(source),
            "--output",
            str(tmp_path / "cli.pt"),
            "--weights",
            "auto",
            "--sha256",
        ],
    )
    export_serving.main()
    assert (tmp_path / "cli.pt").is_file()
    assert "cli.pt" in (tmp_path / "SHA256SUMS").read_text()
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_serving", "--checkpoint", str(source), "--output", str(tmp_path / "main.pt")],
    )
    runpy.run_module("scripts.export_serving", run_name="__main__")
    assert (tmp_path / "main.pt").is_file()


def _write_refusal_fold(cv: Path, fold: int) -> None:
    val = cv / f"fold{fold}" / "val"
    val.mkdir(parents=True)
    ids = [f"{fold}a", f"{fold}b", f"{fold}c", f"{fold}d"]
    (val / "oof.csv").write_text("image_id\n" + "\n".join(ids) + "\n")
    (val / "query.csv").write_text(f"image_id,vehicle_id\n{ids[0]},1\n{ids[2]},2\n")
    (val / "gallery.csv").write_text(f"image_id,vehicle_id\n{ids[1]},1\n{ids[3]},2\n")
    emb = np.eye(4, 8, dtype=np.float32)
    emb[1] = emb[0] + 0.05
    emb[3] = emb[2] + 0.05
    np.save(val / "embeddings.npy", emb)


def test_export_refusal_head(tmp_path, monkeypatch):
    cv = tmp_path / "cv"
    _write_refusal_fold(cv, 0)
    _write_refusal_fold(cv, 1)
    target = tmp_path / "heads" / "eva02_catboost.cbm"
    written = export_refusal.export_refusal(cv, target, seed=0, n_folds=2, k=2)
    assert written.is_file()
    features, labels, hits = export_refusal.collect_features(cv, n_folds=2, k=2)
    assert features.shape[0] == len(labels) == len(hits) == 8
    with pytest.raises(ValueError, match="n_folds"):
        export_refusal.collect_features(cv, n_folds=0)
    with pytest.raises(FileNotFoundError):
        export_refusal.load_pack(tmp_path / "missing")
    _write_refusal_fold(tmp_path / "orphan", 0)
    packed = tmp_path / "orphan" / "fold0" / "val"
    (packed / "query.csv").write_text("image_id,vehicle_id\nmissing,1\n0c,2\n")
    with pytest.raises(ValueError, match="missing image_id"):
        export_refusal.load_pack(packed)
    query, gallery = export_refusal.read_split(cv / "fold0" / "val")

    def fake_embed(fold, frame):
        n = len(frame)
        emb = np.eye(n, 8, dtype=np.float32)
        if n >= 4:
            emb[1] = emb[0] + 0.05
            emb[3] = emb[2] + 0.05
        return emb

    with pytest.raises(ValueError, match="query/gallery rows"):
        export_refusal.pack_from_embeddings(query, gallery, np.zeros((1, 8)))
    retrained = tmp_path / "heads" / "full.cbm"
    yaml_path = tmp_path / "refuse.yaml"
    yaml_path.write_text("kind: model\nmodel_threshold: 0.1234\n")
    written = export_refusal.export_refusal(
        cv,
        retrained,
        seed=0,
        n_folds=2,
        k=2,
        embed_fold=fake_embed,
    )
    assert written.is_file()
    assert not retrained.with_name("full.threshold.json").is_file()
    tuned = tmp_path / "heads" / "tuned.cbm"
    written = export_refusal.export_refusal(
        cv,
        tuned,
        seed=0,
        n_folds=2,
        k=2,
        embed_fold=fake_embed,
        update_config=[yaml_path],
    )
    meta = json.loads(tuned.with_name("tuned.threshold.json").read_text())
    assert "threshold" in meta
    assert "model_threshold:" in yaml_path.read_text()
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("kind: model\n")
    with pytest.raises(ValueError, match="model_threshold"):
        export_refusal.update_refusal_config(bad_yaml, 0.5)
    converted = export_refusal.jsonable(
        {"a": np.float64(1.5), "b": np.int64(2), "c": np.bool_(True), "d": "x"}
    )
    assert converted == {"a": 1.5, "b": 2, "c": 1, "d": "x"}

    class _Model:
        def to(self, device):
            return self

        def eval(self):
            return self

    class _Cfg:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(export_refusal, "initialize_config_dir", lambda **kwargs: _Cfg())
    monkeypatch.setattr(export_refusal, "compose", lambda **kwargs: object())
    monkeypatch.setattr(export_refusal, "load_model", lambda cfg: (_Model(), cfg, {}, "ema"))
    model, cfg, choice = export_refusal.load_embedder(tmp_path / "x.pt", "cpu")
    assert choice == "ema"
    monkeypatch.setattr(export_refusal, "VehicleDataset", lambda frame, cfg, tf: list(range(len(frame))))
    monkeypatch.setattr(export_refusal, "build_transforms", lambda cfg: None)
    monkeypatch.setattr(export_refusal, "DataLoader", lambda ds, **kwargs: ds)
    monkeypatch.setattr(
        export_refusal,
        "embed_loader",
        lambda model, loader, cfg, device: np.ones((len(loader), 4), dtype=np.float32),
    )
    data = type("D", (), {"batch_size_eval": 2, "num_workers": 0, "pin_memory": False})()
    cfg = type("C", (), {"data": data})()
    stacked = export_refusal.embed_split(model, cfg, "cpu", pd.DataFrame({"image_id": ["a", "b"]}))
    assert stacked.shape == (2, 4)

    def fake_split(model, cfg, device, frame):
        return np.ones((len(frame), 4))

    monkeypatch.setattr(export_refusal, "embed_split", fake_split)
    bound = export_refusal.bind_checkpoint_embedder(tmp_path / "x.pt", "cpu")
    assert bound(0, pd.DataFrame({"image_id": ["a"]})).shape == (1, 4)
    monkeypatch.setattr(export_refusal, "source_checkpoint", lambda path: tmp_path / "served.pt")
    (tmp_path / "served.pt").write_bytes(b"x")
    monkeypatch.setattr(export_refusal, "bind_checkpoint_embedder", lambda *args, **kwargs: fake_embed)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_refusal",
            "--full-retrain",
            "--cv",
            str(cv),
            "--checkpoint",
            str(tmp_path / "served.pt"),
            "--output",
            str(tmp_path / "cli-full.cbm"),
            "--n-folds",
            "1",
            "--k",
            "2",
            "--device",
            "cpu",
            "--update-config",
        ],
    )
    monkeypatch.setattr(export_refusal, "DEFAULT_THRESHOLD_CONFIGS", (yaml_path,))
    yaml_path.write_text("kind: model\nmodel_threshold: 0.1234\n")
    export_refusal.main()
    assert (tmp_path / "cli-full.cbm").is_file()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_refusal",
            "--cv",
            str(cv),
            "--output",
            str(tmp_path / "nope.cbm"),
            "--update-config",
            str(yaml_path),
        ],
    )
    with pytest.raises(ValueError, match="--full-retrain"):
        export_refusal.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_refusal",
            "--cv",
            str(cv),
            "--output",
            str(tmp_path / "cli.cbm"),
            "--n-folds",
            "1",
            "--k",
            "2",
            "--sha256",
        ],
    )
    export_refusal.main()
    assert (tmp_path / "cli.cbm").is_file()
    assert "cli.cbm" in (tmp_path / "SHA256SUMS").read_text()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_refusal",
            "--cv",
            str(cv),
            "--output",
            str(tmp_path / "main.cbm"),
            "--n-folds",
            "1",
            "--k",
            "2",
        ],
    )
    runpy.run_module("scripts.export_refusal", run_name="__main__")
    assert (tmp_path / "main.cbm").is_file()
