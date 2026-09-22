import hashlib
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.download_weights as download_weights
import scripts.verify_weights as verify_weights

ROOT = Path(__file__).resolve().parents[1]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_docker_offline_image_pins_and_bakes_weights():
    dockerfile = (ROOT / "Dockerfile").read_text()
    compose = (ROOT / "docker-compose.yml").read_text()
    dockerignore = (ROOT / ".dockerignore").read_text()
    gitattributes = (ROOT / ".gitattributes").read_text()
    gitignore = (ROOT / ".gitignore").read_text()
    runtime = (ROOT / "requirements/runtime.txt").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()
    makefile = (ROOT / "Makefile").read_text()
    assert "COPY refusal /app/refusal" in dockerfile
    assert "COPY weights/finetuned /app/weights/finetuned" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile
    assert "scripts/verify_weights.py --root /app/weights/finetuned" in dockerfile
    assert "checkpoint=/app/weights/finetuned/eva02.pt" in dockerfile
    assert "eval.top_k=10" in dockerfile
    assert "refusal=eva02_model" in dockerfile
    assert "consul-tech" not in dockerfile
    assert "network_mode: none" in compose
    assert "gpus: all" in compose
    assert "./weights:/app/weights" not in compose
    assert "CHECKPOINT:-/app/weights/finetuned/eva02.pt" in compose
    assert "refusal=eva02_model" in compose
    assert "!weights/finetuned/" in dockerignore
    assert "extra_data" in dockerignore
    assert "filter=lfs" in gitattributes
    assert "weights/finetuned/SHA256SUMS" in gitattributes
    assert "!/weights/finetuned/" in gitignore
    assert "consul-tech" not in runtime
    assert "--hash=sha256:" in runtime
    assert "catboost==1.2.10" in runtime
    assert "torch==2.8.0" in runtime
    assert "opencv-python-headless==4.14.0.94" in runtime
    assert 'catboost==1.2.10"' in pyproject
    assert "setuptools==84.0.0" in pyproject
    assert "uv export --frozen --no-dev --no-emit-project" in makefile
    assert (ROOT / "weights/finetuned/SHA256SUMS").is_file()
    assert (ROOT / "weights/finetuned/.gitkeep").is_file()


def test_pretrained_download_specs_are_pinned():
    for name, spec in download_weights.MODELS.items():
        assert spec["sha256"]
        if spec["kind"] == "file":
            assert len(spec["revision"]) == 40
        else:
            assert "revision" not in spec, name


def test_verify_weights_tree_write_and_guards(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert verify_weights.verify_tree(empty) == {}
    (empty / ".gitkeep").write_text("")
    (empty / "SHA256SUMS").write_text("# comment\n\n")
    assert verify_weights.verify_tree(empty) == {}

    payload = tmp_path / "payload"
    payload.mkdir()
    blob = payload / "model.ckpt"
    blob.write_bytes(b"ckpt")
    with pytest.raises(ValueError, match="missing from SHA256SUMS"):
        verify_weights.verify_tree(payload)
    written = verify_weights.write_sha256sums(payload)
    listed = verify_weights.parse_sha256sums(written.read_text())
    assert listed["model.ckpt"] == _digest(b"ckpt")
    assert verify_weights.verify_tree(payload) == listed

    extra = payload / "orphan.bin"
    extra.write_bytes(b"x")
    with pytest.raises(ValueError, match="unlisted payload files"):
        verify_weights.verify_tree(payload)
    extra.unlink()

    nested = payload / "heads"
    nested.mkdir()
    head = nested / "eva02_catboost.cbm"
    head.write_bytes(b"cbm")
    verify_weights.write_sha256sums(payload)
    listed = verify_weights.verify_tree(payload)
    assert listed["heads/eva02_catboost.cbm"] == _digest(b"cbm")

    binary_sums = f"{_digest(b'ckpt')} *model.ckpt\n"
    assert verify_weights.parse_sha256sums(binary_sums)["model.ckpt"] == _digest(b"ckpt")
    with pytest.raises(ValueError, match="invalid SHA256SUMS line"):
        verify_weights.parse_sha256sums("not-a-hash  model.ckpt")
    with pytest.raises(ValueError, match="invalid SHA256SUMS path"):
        verify_weights.parse_sha256sums(f"{_digest(b'x')}  ../escape.ckpt")
    with pytest.raises(ValueError, match="invalid SHA256SUMS path"):
        verify_weights.parse_sha256sums(f"{_digest(b'x')}  /abs.ckpt")
    with pytest.raises(FileNotFoundError):
        verify_weights.verify_tree(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        verify_weights.verify_digests(payload, {"missing.ckpt": _digest(b"x")})
    with pytest.raises(ValueError, match="expected"):
        verify_weights.verify_digests(payload, {"model.ckpt": _digest(b"wrong")})

    monkeypatch.chdir(payload)
    monkeypatch.setattr(sys, "argv", ["verify_weights", "--root", str(payload), "--write"])
    verify_weights.main()
    monkeypatch.setattr(sys, "argv", ["verify_weights", "--root", str(payload)])
    verify_weights.main()
    monkeypatch.setattr(sys, "argv", ["verify_weights", "--root", str(payload)])
    runpy.run_module("scripts.verify_weights", run_name="__main__")


def test_download_weights_verify_digests(tmp_path, monkeypatch):
    root = tmp_path / "radio"
    root.mkdir()
    data = b"radio-bytes"
    (root / "model.safetensors").write_bytes(data)
    monkeypatch.setitem(
        download_weights.MODELS,
        "radio",
        {
            "kind": "file",
            "repo_id": "nvidia/C-RADIOv4-SO400M",
            "filename": "model.safetensors",
            "revision": "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8",
            "sha256": {"model.safetensors": _digest(data)},
        },
    )
    monkeypatch.setattr(
        download_weights,
        "hf_hub_download",
        lambda *args, **kwargs: str(root / "model.safetensors"),
    )
    monkeypatch.setattr(sys, "argv", ["download_weights", "radio", "--directory", str(tmp_path)])
    download_weights.main()
    radio = download_weights.MODELS["radio"]
    assert radio["kind"] == "file"
    radio["sha256"] = {"model.safetensors": _digest(b"other")}
    with pytest.raises(ValueError, match="expected"):
        download_weights.main()


def test_ruff_and_ty_pass_on_notebooks():
    notebooks = ROOT / "notebooks"
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", str(notebooks)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ruff.returncode == 0, ruff.stdout + ruff.stderr
    typed = subprocess.run(
        [sys.executable, "-m", "ty", "check", str(notebooks)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert typed.returncode == 0, typed.stdout + typed.stderr
