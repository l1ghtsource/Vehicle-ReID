import json
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import models
import scripts.aggregate_cv as aggregate_cv
import scripts.audit_data as audit_data
import scripts.check_backbone as check_backbone
import scripts.download_weights as download_weights
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
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_cv.main()

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
    for name in ("dinov3_base", "dinov3_large", "radio", "llm2clip"):
        monkeypatch.setattr(
            sys,
            "argv",
            ["download_weights", name, "--directory", str(tmp_path)],
        )
        download_weights.main()
    assert len(snapshots) == 2
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
