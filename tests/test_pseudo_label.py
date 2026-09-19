import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.pseudo_label as pseudo_label


def _ann_csv(path: Path, n: int, *, labeled: bool, prefix: str) -> Path:
    rows = []
    for index in range(n):
        row = {"image_id": f"{prefix}{index}", "x": 0, "y": 0, "w": 4, "h": 4}
        if labeled:
            row["vehicle_id"] = index // 2
            row["camera_id"] = index % 2
        rows.append(row)
    path.write_text(pd.DataFrame(rows).to_csv(index=False))
    return path


def test_cluster_embeddings_and_remap():
    rng = np.random.default_rng(0)
    first = rng.normal(0, 0.01, size=(12, 8)).astype(np.float32)
    second = rng.normal(6, 0.01, size=(12, 8)).astype(np.float32)
    labels = pseudo_label.cluster_embeddings(
        np.concatenate([first, second]),
        min_cluster_size=4,
        min_samples=2,
        allow_single_cluster=True,
    )
    assert len(set(labels[labels >= 0])) == 2
    vehicle_ids, mapping = pseudo_label.remap_cluster_ids(labels, 10)
    assert set(mapping.values()) == {10, 11}
    assert set(vehicle_ids[vehicle_ids >= 0]) == {10, 11}
    with pytest.raises(ValueError, match="min_cluster_size"):
        pseudo_label.cluster_embeddings(first, min_cluster_size=1)
    with pytest.raises(ValueError, match="min_samples"):
        pseudo_label.cluster_embeddings(first, min_samples=0)
    with pytest.raises(ValueError, match="non-empty"):
        pseudo_label.cluster_embeddings(np.zeros((0, 4)))
    with pytest.raises(ValueError, match="non-empty"):
        pseudo_label.cluster_embeddings(np.zeros(4))
    with pytest.raises(ValueError, match="Fewer test"):
        pseudo_label.cluster_embeddings(first[:3], min_cluster_size=8)


def test_test_frame_and_merge_train():
    query = pd.DataFrame({"image_id": ["a", "b"], "x": [0, 1], "y": [0, 1], "w": [2, 2], "h": [2, 2]})
    gallery = pd.DataFrame({"image_id": ["a", "c"], "x": [9, 3], "y": [9, 3], "w": [2, 2], "h": [2, 2]})
    frame, duplicated = pseudo_label.test_frame(query, gallery)
    assert duplicated == 1
    assert list(frame.image_id) == ["a", "b", "c"]
    assert int(frame.loc[0, "camera_id"]) == -1
    with_cam, _ = pseudo_label.test_frame(query.assign(camera_id=4), gallery.assign(camera_id=5))
    assert int(with_cam.loc[0, "camera_id"]) == 4
    orig = pd.DataFrame(
        {
            "image_id": ["keep"],
            "x": [0],
            "y": [0],
            "w": [2],
            "h": [2],
            "vehicle_id": [7],
        }
    )
    extra = pd.DataFrame(
        {
            "image_id": ["keep", "new", "new"],
            "x": [1, 2, 2],
            "y": [1, 2, 2],
            "w": [2, 2, 2],
            "h": [2, 2, 2],
            "vehicle_id": [8, 9, 9],
        }
    )
    with pytest.raises(ValueError, match="Repeated image_id"):
        pseudo_label.merge_train(orig, extra)
    extra = extra.drop_duplicates("image_id")
    merged = pseudo_label.merge_train(orig, extra)
    assert list(merged.image_id) == ["keep", "new"]
    assert int(merged.loc[1, "vehicle_id"]) == 9
    assert int(merged.loc[0, "camera_id"]) == -1
    orig_cam = orig.copy()
    orig_cam["camera_id"] = 5
    extra_cam = extra.copy()
    extra_cam["camera_id"] = 6
    kept = pseudo_label.merge_train(orig_cam, extra_cam)
    assert int(kept.loc[0, "camera_id"]) == 5
    assert int(kept.loc[1, "camera_id"]) == 6


def test_run_pseudo_label_writes_train(tmp_path, monkeypatch):
    train_csv = _ann_csv(tmp_path / "train.csv", 4, labeled=True, prefix="tr")
    query_csv = _ann_csv(tmp_path / "query.csv", 8, labeled=False, prefix="q")
    gallery_csv = _ann_csv(tmp_path / "gallery.csv", 8, labeled=False, prefix="g")
    empty_train = tmp_path / "empty.csv"
    empty_train.write_text("image_id,x,y,w,h,vehicle_id\n")
    output = tmp_path / "iter001"

    def two_blobs(frame):
        n = len(frame)
        emb = np.zeros((n, 4), dtype=np.float32)
        emb[n // 2 :] = 5.0
        return emb

    summary = pseudo_label.run_pseudo_label(
        output=output,
        train_csv=train_csv,
        query_csv=query_csv,
        gallery_csv=gallery_csv,
        checkpoint=tmp_path / "eva02.pt",
        device="cpu",
        min_cluster_size=4,
        embed_fn=two_blobs,
    )
    assert summary["n_clusters"] == 2
    assert summary["n_noise"] == 0
    assert summary["n_pseudo"] == 16
    assert summary["n_train_merged"] == 20
    written = pd.read_csv(output / "train.csv", dtype={"image_id": str})
    assert len(written) == 20
    assert written.vehicle_id.max() == 3
    meta = json.loads((output / "summary.json").read_text())
    assert meta["next_train"]["data.val_source_csv"] == str(train_csv)
    with pytest.raises(ValueError, match="already exists"):
        pseudo_label.run_pseudo_label(
            output=output,
            train_csv=train_csv,
            query_csv=query_csv,
            gallery_csv=gallery_csv,
            embed_fn=two_blobs,
        )
    file_out = tmp_path / "as_file"
    file_out.write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        pseudo_label.run_pseudo_label(
            output=file_out,
            train_csv=train_csv,
            query_csv=query_csv,
            gallery_csv=gallery_csv,
            embed_fn=two_blobs,
        )
    with pytest.raises(ValueError, match="empty"):
        pseudo_label.run_pseudo_label(
            output=tmp_path / "empty-run",
            train_csv=empty_train,
            query_csv=query_csv,
            gallery_csv=gallery_csv,
            embed_fn=two_blobs,
        )
    overlap_train = tmp_path / "all-overlap.csv"
    overlap_rows = pd.concat(
        [
            pd.read_csv(query_csv, dtype={"image_id": str}),
            pd.read_csv(gallery_csv, dtype={"image_id": str}),
        ],
        ignore_index=True,
    )
    overlap_rows["vehicle_id"] = range(len(overlap_rows))
    overlap_rows.to_csv(overlap_train, index=False)
    with pytest.raises(ValueError, match="unlabeled"):
        pseudo_label.run_pseudo_label(
            output=tmp_path / "all-overlap",
            train_csv=overlap_train,
            query_csv=query_csv,
            gallery_csv=gallery_csv,
            embed_fn=two_blobs,
        )
    with pytest.raises(ValueError, match="do not match"):
        pseudo_label.run_pseudo_label(
            output=tmp_path / "bad-emb",
            train_csv=train_csv,
            query_csv=query_csv,
            gallery_csv=gallery_csv,
            embed_fn=lambda frame: np.zeros((1, 4), dtype=np.float32),
        )
    monkeypatch.setattr(
        pseudo_label,
        "cluster_embeddings",
        lambda *args, **kwargs: np.full(16, -1, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="no clusters"):
        pseudo_label.run_pseudo_label(
            output=tmp_path / "noise",
            train_csv=train_csv,
            query_csv=query_csv,
            gallery_csv=gallery_csv,
            embed_fn=two_blobs,
        )


def test_run_pseudo_label_skips_train_overlap_and_loads_embedder(tmp_path, monkeypatch):
    train_csv = _ann_csv(tmp_path / "train.csv", 2, labeled=True, prefix="x")
    query = tmp_path / "query.csv"
    gallery = tmp_path / "gallery.csv"
    query.write_text("image_id,x,y,w,h\nx0,0,0,4,4\nq1,0,0,4,4\n")
    gallery.write_text("image_id,x,y,w,h\nq2,0,0,4,4\nq3,0,0,4,4\n")

    class _Cfg:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(pseudo_label, "load_embedder", lambda *args, **kwargs: (object(), object(), "ema"))
    monkeypatch.setattr(
        pseudo_label,
        "embed_split",
        lambda model, cfg, device, frame: np.ones((len(frame), 4), dtype=np.float32),
    )
    ckpt = tmp_path / "eva02.pt"
    ckpt.write_bytes(b"x")
    summary = pseudo_label.run_pseudo_label(
        output=tmp_path / "overlap",
        train_csv=train_csv,
        query_csv=query,
        gallery_csv=gallery,
        checkpoint=ckpt,
        device="cpu",
        min_cluster_size=2,
        min_samples=2,
        allow_single_cluster=True,
    )
    assert summary["n_overlap_train"] == 1
    assert summary["n_pseudo"] == 3
    assert summary["weights"] == "ema"
    with pytest.raises(FileNotFoundError):
        pseudo_label.run_pseudo_label(
            output=tmp_path / "missing-emb",
            train_csv=train_csv,
            query_csv=query,
            gallery_csv=gallery,
            embeddings_path=tmp_path / "absent.npy",
            min_cluster_size=2,
            min_samples=2,
        )
    saved = tmp_path / "saved.npy"
    np.save(saved, np.ones((3, 4), dtype=np.float32))
    reused = pseudo_label.run_pseudo_label(
        output=tmp_path / "from-npy",
        train_csv=train_csv,
        query_csv=query,
        gallery_csv=gallery,
        checkpoint=ckpt,
        embeddings_path=saved,
        min_cluster_size=2,
        min_samples=2,
        allow_single_cluster=True,
    )
    assert reused["n_pseudo"] == 3
    assert reused["weights"] == "ema"


def test_resolve_checkpoint_and_cli(tmp_path, monkeypatch):
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        pseudo_label.resolve_checkpoint(missing)
    present = tmp_path / "eva02.pt"
    present.write_bytes(b"x")
    assert pseudo_label.resolve_checkpoint(present) == present
    monkeypatch.setattr(pseudo_label, "DEFAULT_OUTPUT", present)
    assert pseudo_label.resolve_checkpoint(None) == present
    monkeypatch.setattr(pseudo_label, "DEFAULT_OUTPUT", tmp_path / "absent.pt")
    monkeypatch.setattr(pseudo_label, "source_checkpoint", lambda path: present)
    assert pseudo_label.resolve_checkpoint(None) == present

    args = pseudo_label.parse_args(["--iter", "2", "--no-allow-single-cluster", "--min-samples", "3"])
    assert args.iter == 2
    assert args.allow_single_cluster is False
    assert args.min_samples == 3
    defaults = pseudo_label.parse_args([])
    assert defaults.min_cluster_size == 4
    assert defaults.min_samples == 4
    assert defaults.allow_single_cluster is False
    with pytest.raises(ValueError, match="iter"):
        pseudo_label.main(["--iter", "0"])

    called = {}

    def capture(**kwargs):
        called.update(kwargs)
        return {"n_clusters": 1}

    monkeypatch.setattr(pseudo_label, "run_pseudo_label", capture)
    pseudo_label.main(
        ["--iter", "3", "--device", "cpu", "--embeddings", "runs/pseudo/iter001/embeddings.npy"]
    )
    assert called["output"] == Path("runs/pseudo/iter003")
    assert called["allow_single_cluster"] is False
    assert called["embeddings_path"] == Path("runs/pseudo/iter001/embeddings.npy")
    assert called["min_cluster_size"] == 4
    assert called["min_samples"] == 4
    monkeypatch.setattr(sys, "argv", ["pseudo_label", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("scripts.pseudo_label", run_name="__main__")
    assert exc.value.code == 0
