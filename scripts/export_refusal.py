import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from postproc.retrieval import normalize
from refusal import balanced_pack, fit_boosting, save_boosting
from scripts.verify_weights import write_sha256sums

DEFAULT_CV = Path("runs/cv/eva02_trial23")
DEFAULT_OUTPUT = Path("weights/finetuned/eva02_catboost.cbm")
N_FOLDS = 5


def required_file(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_pack(directory: Path) -> dict:
    oof = pd.read_csv(required_file(directory, "oof.csv"), dtype={"image_id": str})
    query = pd.read_csv(required_file(directory, "query.csv"), dtype={"image_id": str})
    gallery = pd.read_csv(required_file(directory, "gallery.csv"), dtype={"image_id": str})
    emb = normalize(np.load(required_file(directory, "embeddings.npy")))
    lookup = {image_id: i for i, image_id in enumerate(oof.image_id)}
    missing = [image_id for image_id in (*query.image_id, *gallery.image_id) if image_id not in lookup]
    if missing:
        raise ValueError(f"{directory} embeddings.npy is missing image_id {missing[0]}")
    return {
        "query": query,
        "gallery": gallery,
        "qe": emb[[lookup[image_id] for image_id in query.image_id]],
        "ge": emb[[lookup[image_id] for image_id in gallery.image_id]],
    }


def collect_features(cv: Path, n_folds: int = N_FOLDS, k: int = 10):
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    xs, ys = [], []
    for fold in range(n_folds):
        pack = load_pack(Path(cv) / f"fold{fold}" / "val")
        features, labels, _, _ = balanced_pack(
            pack["qe"],
            pack["ge"],
            pack["query"].vehicle_id.to_numpy(),
            pack["gallery"].vehicle_id.to_numpy(),
            k=k,
            with_embeddings=True,
        )
        xs.append(features)
        ys.append(labels)
    return np.concatenate(xs), np.concatenate(ys)


def export_refusal(cv: Path, destination: Path, seed: int = 0, n_folds: int = N_FOLDS, k: int = 10) -> Path:
    features, labels = collect_features(cv, n_folds=n_folds, k=k)
    model = fit_boosting(features, labels, seed=seed, iterations=200, depth=4, learning_rate=0.08)
    return save_boosting(model, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=Path, default=DEFAULT_CV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sha256", action="store_true")
    args = parser.parse_args()
    target = export_refusal(args.cv, args.output, seed=args.seed, n_folds=args.n_folds, k=args.k)
    print(f"{args.cv} -> {target} ({target.stat().st_size} bytes)")
    if args.sha256:
        written = write_sha256sums(target.parent)
        print(written)


if __name__ == "__main__":
    main()
