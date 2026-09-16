import json
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from torch import nn

import models
import scripts.aggregate_cv as aggregate_cv
import scripts.audit_data as audit_data
import scripts.check_backbone as check_backbone
import scripts.download_weights as download_weights
import scripts.prepare_folds as prepare_folds


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
    for name in ("dinov3_base", "dinov3_large", "radio", "llm2clip"):
        monkeypatch.setattr(
            sys,
            "argv",
            ["download_weights", name, "--directory", str(tmp_path)],
        )
        download_weights.main()
    assert len(snapshots) == 2
    assert len(files) == 2

    monkeypatch.setattr(
        sys,
        "argv",
        ["download_weights", "radio", "--directory", str(tmp_path)],
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: "file")
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
    Path("artifacts").mkdir()
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
