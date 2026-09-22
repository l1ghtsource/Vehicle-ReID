import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from dataset.folds import read_annotations
from scripts.export_refusal import embed_split, jsonable, load_embedder
from scripts.export_serving import DEFAULT_OUTPUT, source_checkpoint

TRAIN_COLUMNS = ["image_id", "x", "y", "w", "h", "vehicle_id", "camera_id"]


def cluster_embeddings(
    embeddings,
    min_cluster_size=2,
    min_samples=None,
    allow_single_cluster=False,
):
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be >= 2")
    if min_samples is not None and min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] == 0:
        raise ValueError("Need a non-empty embedding matrix")
    if emb.shape[0] < min_cluster_size:
        raise ValueError("Fewer test embeddings than min_cluster_size")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-12)
    kwargs = {
        "min_cluster_size": int(min_cluster_size),
        "metric": "euclidean",
        "allow_single_cluster": bool(allow_single_cluster),
    }
    if min_samples is not None:
        kwargs["min_samples"] = int(min_samples)
    labels = HDBSCAN(**kwargs).fit_predict(emb)
    return np.asarray(labels, dtype=np.int32)


def remap_cluster_ids(labels, start_id):
    unique = sorted(int(label) for label in np.unique(labels) if int(label) >= 0)
    mapping = {old: int(start_id) + index for index, old in enumerate(unique)}
    vehicle_ids = np.full(len(labels), -1, dtype=np.int64)
    for index, label in enumerate(labels):
        key = int(label)
        if key in mapping:
            vehicle_ids[index] = mapping[key]
    return vehicle_ids, mapping


def test_frame(query: pd.DataFrame, gallery: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    left = query.copy()
    right = gallery.copy()
    left["source"] = "query"
    right["source"] = "gallery"
    frame = pd.concat([left, right], ignore_index=True)
    duplicated = int(frame.image_id.duplicated().sum())
    frame = frame.drop_duplicates("image_id", keep="first").reset_index(drop=True)
    if "camera_id" not in frame:
        frame["camera_id"] = -1
    return frame, duplicated


def merge_train(orig: pd.DataFrame, pseudo: pd.DataFrame) -> pd.DataFrame:
    labeled = orig.copy()
    extra = pseudo.copy()
    if "camera_id" not in labeled:
        labeled["camera_id"] = -1
    if "camera_id" not in extra:
        extra["camera_id"] = -1
    train_ids = set(labeled.image_id.astype(str))
    extra = extra[~extra.image_id.astype(str).isin(train_ids)].copy()
    merged = pd.concat([labeled[TRAIN_COLUMNS], extra[TRAIN_COLUMNS]], ignore_index=True)
    if merged.image_id.duplicated().any():
        raise ValueError("Repeated image_id in merged train")
    return merged


def resolve_checkpoint(path: Path | None) -> Path:
    if path is not None:
        return source_checkpoint(path)
    default = Path(DEFAULT_OUTPUT)
    if default.is_file():
        return default
    return source_checkpoint(None)


def write_artifacts(
    output: Path,
    *,
    embeddings: np.ndarray,
    clusters: pd.DataFrame,
    train: pd.DataFrame,
    summary: dict,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    clusters.to_csv(output / "clusters.csv", index=False)
    train.to_csv(output / "train.csv", index=False)
    (output / "embedding_order.csv").write_text(clusters[["image_id"]].to_csv(index=False))
    (output / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    return summary


def run_pseudo_label(
    *,
    output: Path,
    train_csv: Path,
    query_csv: Path,
    gallery_csv: Path,
    checkpoint: Path | None = None,
    device: str = "cpu",
    weights: str = "ema",
    min_cluster_size: int = 4,
    min_samples: int | None = 4,
    allow_single_cluster: bool = False,
    embeddings_path: Path | None = None,
    embed_fn=None,
) -> dict:
    output = Path(output)
    if output.exists() and not output.is_dir():
        raise ValueError(f"{output} is not a directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} already exists")
    if pd.read_csv(train_csv, dtype={"image_id": str}).empty:
        raise ValueError("train.csv is empty")
    orig = read_annotations(train_csv, labeled=True)
    query = read_annotations(query_csv)
    gallery = read_annotations(gallery_csv)
    frame, n_duplicate = test_frame(query, gallery)
    n_test = int(len(frame))
    train_ids = set(orig.image_id.astype(str))
    n_overlap = int(frame.image_id.astype(str).isin(train_ids).sum())
    frame = frame[~frame.image_id.astype(str).isin(train_ids)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No unlabeled test rows")
    if embed_fn is not None:
        ckpt = Path(checkpoint) if checkpoint is not None else Path(DEFAULT_OUTPUT)
        choice = weights
        embeddings = np.asarray(embed_fn(frame), dtype=np.float32)
    elif embeddings_path is not None:
        path = Path(embeddings_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        ckpt = resolve_checkpoint(checkpoint)
        choice = weights
        embeddings = np.asarray(np.load(path), dtype=np.float32)
    else:
        ckpt = resolve_checkpoint(checkpoint)
        model, cfg, choice = load_embedder(ckpt, device, weights)
        embeddings = embed_split(model, cfg, device, frame)
    if len(embeddings) != len(frame):
        raise ValueError("Embeddings do not match test rows")
    labels = cluster_embeddings(
        embeddings,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        allow_single_cluster=allow_single_cluster,
    )
    start_id = int(orig.vehicle_id.astype(int).max()) + 1
    vehicle_ids, mapping = remap_cluster_ids(labels, start_id)
    if not mapping:
        raise ValueError("HDBSCAN produced no clusters")
    clustered = frame.copy()
    clustered["cluster_id"] = labels
    clustered["vehicle_id"] = vehicle_ids
    labeled = clustered[clustered.vehicle_id >= 0].copy()
    merged = merge_train(orig, labeled)
    n_noise = int((labels < 0).sum())
    n_pseudo = int(len(merged) - len(orig))
    summary = {
        "checkpoint": str(ckpt),
        "weights": choice,
        "output": str(output),
        "min_cluster_size": int(min_cluster_size),
        "min_samples": min_samples,
        "allow_single_cluster": bool(allow_single_cluster),
        "n_query": int(len(query)),
        "n_gallery": int(len(gallery)),
        "n_test": n_test,
        "n_unlabeled": int(len(frame)),
        "n_duplicate_image_id": n_duplicate,
        "n_noise": n_noise,
        "noise_fraction": float(n_noise / len(frame)),
        "n_clusters": int(len(mapping)),
        "n_overlap_train": n_overlap,
        "n_pseudo": n_pseudo,
        "n_train_orig": int(len(orig)),
        "n_train_merged": int(len(merged)),
        "next_train": {
            "data.train_csv": str(output / "train.csv"),
            "data.folds_file": str(output / "folds.csv"),
            "data.val_source_csv": str(train_csv),
        },
    }
    write_artifacts(
        output,
        embeddings=embeddings,
        clusters=clustered[TRAIN_COLUMNS + ["cluster_id", "source"]],
        train=merged,
        summary=summary,
    )
    return summary


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("auto", "raw", "ema"), default="ema")
    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--query-csv", type=Path, default=Path("data/test_query.csv"))
    parser.add_argument("--gallery-csv", type=Path, default=Path("data/test_gallery.csv"))
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--min-cluster-size", type=int, default=4)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument(
        "--allow-single-cluster",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.iter < 1:
        raise ValueError("iter must be >= 1")
    output = args.output or Path(f"runs/pseudo/iter{args.iter:03d}")
    summary = run_pseudo_label(
        output=output,
        train_csv=args.train_csv,
        query_csv=args.query_csv,
        gallery_csv=args.gallery_csv,
        checkpoint=args.checkpoint,
        device=args.device,
        weights=args.weights,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        allow_single_cluster=args.allow_single_cluster,
        embeddings_path=args.embeddings,
    )
    print(json.dumps(jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
