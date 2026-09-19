import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader

from augmentations import build_transforms
from dataset.images import VehicleDataset
from eval import load_model
from modules.inference import embed_loader
from postproc.retrieval import normalize
from refusal import balanced_pack, fit_boosting, predict_boosting, save_boosting, select_threshold
from scripts.export_serving import source_checkpoint
from scripts.verify_weights import write_sha256sums

DEFAULT_CV = Path("runs/cv/eva02_trial23")
DEFAULT_OUTPUT = Path("weights/finetuned/eva02_catboost.cbm")
DEFAULT_THRESHOLD_CONFIGS = (
    Path("configs/refusal/eva02_model.yaml"),
    Path("configs/refusal/eva02_ensemble.yaml"),
)
N_FOLDS = 5
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
THRESHOLD_LINE = re.compile(r"^model_threshold:\s*[0-9.eE+-]+\s*$", re.MULTILINE)


def required_file(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_split(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    query = pd.read_csv(required_file(directory, "query.csv"), dtype={"image_id": str})
    gallery = pd.read_csv(required_file(directory, "gallery.csv"), dtype={"image_id": str})
    return query, gallery


def load_pack(directory: Path) -> dict:
    oof = pd.read_csv(required_file(directory, "oof.csv"), dtype={"image_id": str})
    query, gallery = read_split(directory)
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


def pack_from_embeddings(query: pd.DataFrame, gallery: pd.DataFrame, embeddings: np.ndarray) -> dict:
    emb = np.asarray(embeddings, dtype=np.float32)
    if len(emb) != len(query) + len(gallery):
        raise ValueError("Embeddings do not match query/gallery rows")
    return {
        "query": query,
        "gallery": gallery,
        "qe": emb[: len(query)],
        "ge": emb[len(query) :],
    }


def load_embedder(checkpoint: Path, device: str, weights: str = "ema"):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[
                f"checkpoint={checkpoint}",
                f"eval.device={device}",
                f"eval.weights={weights}",
            ],
        )
    model, cfg, _blob, choice = load_model(cfg)
    model.to(torch.device(device)).eval()
    return model, cfg, choice


def embed_split(model, cfg, device, frame: pd.DataFrame) -> np.ndarray:
    loader = DataLoader(
        VehicleDataset(frame.reset_index(drop=True), cfg, build_transforms(cfg)),
        batch_size=int(cfg.data.batch_size_eval),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
    )
    return normalize(embed_loader(model, loader, cfg, torch.device(device)))


def bind_checkpoint_embedder(checkpoint: Path, device: str, weights: str = "ema"):
    model, cfg, _choice = load_embedder(checkpoint, device, weights)

    def embed_fold(_fold: int, frame: pd.DataFrame) -> np.ndarray:
        return embed_split(model, cfg, device, frame)

    return embed_fold


def collect_features(cv: Path, n_folds: int = N_FOLDS, k: int = 10, embed_fold=None):
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    xs, ys, hs = [], [], []
    root = Path(cv)
    for fold in range(n_folds):
        directory = root / f"fold{fold}" / "val"
        if embed_fold is None:
            pack = load_pack(directory)
        else:
            query, gallery = read_split(directory)
            frame = pd.concat([query, gallery], ignore_index=True)
            pack = pack_from_embeddings(query, gallery, embed_fold(fold, frame))
        features, labels, _, hits = balanced_pack(
            pack["qe"],
            pack["ge"],
            pack["query"].vehicle_id.to_numpy(),
            pack["gallery"].vehicle_id.to_numpy(),
            k=k,
            with_embeddings=True,
        )
        xs.append(features)
        ys.append(labels)
        hs.append(hits)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(hs)


def jsonable(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if isinstance(value, (np.floating, float)):
            out[key] = float(value)
        elif isinstance(value, (np.integer, int, np.bool_)):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def update_refusal_config(path: Path, threshold: float) -> None:
    text = Path(path).read_text()
    updated, count = THRESHOLD_LINE.subn(f"model_threshold: {threshold:.4f}", text, count=1)
    if count != 1:
        raise ValueError(f"Could not update model_threshold in {path}")
    Path(path).write_text(updated)


def export_refusal(
    cv: Path,
    destination: Path,
    seed: int = 0,
    n_folds: int = N_FOLDS,
    k: int = 10,
    checkpoint: Path | None = None,
    device: str = "cpu",
    weights: str = "ema",
    embed_fold=None,
    update_config: list[Path] | None = None,
) -> Path:
    if checkpoint is not None and embed_fold is None:
        embed_fold = bind_checkpoint_embedder(checkpoint, device, weights)
    features, labels, hits = collect_features(cv, n_folds=n_folds, k=k, embed_fold=embed_fold)
    model = fit_boosting(features, labels, seed=seed, iterations=200, depth=4, learning_rate=0.08)
    target = save_boosting(model, destination)
    if update_config:
        chosen = jsonable(select_threshold(labels.astype(bool), predict_boosting(model, features), hits))
        meta = target.with_name(f"{target.stem}.threshold.json")
        meta.write_text(json.dumps(chosen, indent=2))
        for path in update_config:
            update_refusal_config(path, float(chosen["threshold"]))
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=Path, default=DEFAULT_CV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--full-retrain", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="ema")
    parser.add_argument("--update-config", nargs="*", type=Path, default=None)
    parser.add_argument("--sha256", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint
    configs = args.update_config
    if args.full_retrain:
        checkpoint = checkpoint or source_checkpoint(None)
        if configs is not None and not configs:
            configs = list(DEFAULT_THRESHOLD_CONFIGS)
    elif configs:
        raise ValueError("--update-config requires --full-retrain")
    target = export_refusal(
        args.cv,
        args.output,
        seed=args.seed,
        n_folds=args.n_folds,
        k=args.k,
        checkpoint=checkpoint,
        device=args.device,
        weights=args.weights,
        update_config=configs,
    )
    print(f"{args.cv} -> {target} ({target.stat().st_size} bytes)")
    if args.sha256:
        written = write_sha256sums(target.parent)
        print(written)


if __name__ == "__main__":
    main()
