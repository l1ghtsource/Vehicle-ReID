import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir

from eval import load_model
from modules.inference import embed_frame
from postproc.retrieval import normalize
from refusal import (
    OPEN_SET_FRACTION,
    balanced_pack,
    decide,
    decision_metrics,
    fit_boosting,
    open_set_pack,
    predict_boosting,
    ranking_metrics,
    save_boosting,
    select_threshold,
)
from scripts.export_serving import source_checkpoint
from scripts.verify_weights import write_sha256sums

DEFAULT_CV = Path("runs/cv/eva02_trial23")
DEFAULT_OUTPUT = Path("weights/finetuned/eva02_catboost.cbm")
DEFAULT_NESTED_CONFIGS = (
    Path("configs/refusal/eva02_threshold.yaml"),
    Path("configs/refusal/eva02_model.yaml"),
    Path("configs/refusal/eva02_ensemble.yaml"),
)
N_FOLDS = 5
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


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
    return embed_frame(model, cfg, device, frame)


def bind_checkpoint_embedder(checkpoint: Path, device: str, weights: str = "ema"):
    model, cfg, _choice = load_embedder(checkpoint, device, weights)

    def embed_fold(_fold: int, frame: pd.DataFrame) -> np.ndarray:
        return embed_split(model, cfg, device, frame)

    return embed_fold


def collect_packs(cv: Path, n_folds: int = N_FOLDS, k: int = 10, embed_fold=None):
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    packs = []
    root = Path(cv)
    for fold in range(n_folds):
        directory = root / f"fold{fold}" / "val"
        if embed_fold is None:
            pack = load_pack(directory)
        else:
            query, gallery = read_split(directory)
            frame = pd.concat([query, gallery], ignore_index=True)
            pack = pack_from_embeddings(query, gallery, embed_fold(fold, frame))
        qids = pack["query"].vehicle_id.to_numpy()
        gids = pack["gallery"].vehicle_id.to_numpy()
        features, labels, cosine, hits = balanced_pack(
            pack["qe"],
            pack["ge"],
            qids,
            gids,
            k=k,
            with_embeddings=True,
        )
        eval_x, eval_y, eval_cos, eval_top, _held = open_set_pack(
            pack["qe"],
            pack["ge"],
            qids,
            gids,
            k=k,
            with_embeddings=True,
            fraction=OPEN_SET_FRACTION,
            seed=fold,
        )
        packs.append(
            {
                "X": features,
                "y": labels,
                "cos": cosine,
                "top": hits,
                "eval_X": eval_x,
                "eval_y": eval_y,
                "eval_cos": eval_cos,
                "eval_top": eval_top,
            }
        )
    return packs


def collect_features(cv: Path, n_folds: int = N_FOLDS, k: int = 10, embed_fold=None):
    packs = collect_packs(cv, n_folds=n_folds, k=k, embed_fold=embed_fold)
    return (
        np.concatenate([pack["X"] for pack in packs]),
        np.concatenate([pack["y"] for pack in packs]),
        np.concatenate([pack["top"] for pack in packs]),
    )


EVAL_FIELDS = {"X": "eval_X", "y": "eval_y", "cos": "eval_cos", "top": "eval_top"}


def concat_pack(packs, indices, key, split="train"):
    field = key if split == "train" else EVAL_FIELDS[key]
    return np.concatenate([packs[index][field] for index in indices], 0)


def public_head(report):
    return {key: value for key, value in report.items() if key != "_parts"}


def outer_metrics(y_parts, score_parts, top_parts, accept_parts):
    y = np.concatenate(y_parts)
    scores = np.concatenate(score_parts)
    top = np.concatenate(top_parts)
    accept = np.concatenate(accept_parts)
    return jsonable({**decision_metrics(y.astype(bool), accept, top), **ranking_metrics(y, scores)})


def nested_cosine(packs, kind="contest"):
    n_folds = len(packs)
    if n_folds < 3:
        raise ValueError("Nested threshold needs at least 3 folds")
    fold_thresholds = []
    y_parts, score_parts, top_parts, accept_parts = [], [], [], []
    for test_fold in range(n_folds):
        others = [fold for fold in range(n_folds) if fold != test_fold]
        picked = select_threshold(
            concat_pack(packs, others, "y", split="eval").astype(bool),
            concat_pack(packs, others, "cos", split="eval"),
            concat_pack(packs, others, "top", split="eval"),
            kind=kind,
        )
        threshold = float(picked["threshold"])
        fold_thresholds.append(threshold)
        y_parts.append(packs[test_fold]["eval_y"])
        score_parts.append(packs[test_fold]["eval_cos"])
        top_parts.append(packs[test_fold]["eval_top"])
        accept_parts.append(decide(packs[test_fold]["eval_cos"], threshold))
    return {
        "fold_thresholds": fold_thresholds,
        "threshold": float(np.mean(fold_thresholds)),
        "outer": outer_metrics(y_parts, score_parts, top_parts, accept_parts),
        "_parts": {"y": y_parts, "scores": score_parts, "top": top_parts, "accept": accept_parts},
    }


def nested_catboost(packs, seed=0, kind="contest"):
    n_folds = len(packs)
    if n_folds < 3:
        raise ValueError("Nested threshold needs at least 3 folds")
    fold_thresholds = []
    y_parts, score_parts, top_parts, accept_parts = [], [], [], []
    for test_fold in range(n_folds):
        others = [fold for fold in range(n_folds) if fold != test_fold]
        inner_y, inner_scores, inner_top = [], [], []
        for val_fold in others:
            train_inner = [fold for fold in others if fold != val_fold]
            model = fit_boosting(
                concat_pack(packs, train_inner, "X"),
                concat_pack(packs, train_inner, "y"),
                seed=10 * test_fold + val_fold,
                iterations=200,
                depth=4,
                learning_rate=0.08,
            )
            inner_y.append(packs[val_fold]["eval_y"])
            inner_top.append(packs[val_fold]["eval_top"])
            inner_scores.append(predict_boosting(model, packs[val_fold]["eval_X"]))
        picked = select_threshold(
            np.concatenate(inner_y).astype(bool),
            np.concatenate(inner_scores),
            np.concatenate(inner_top),
            kind=kind,
        )
        threshold = float(picked["threshold"])
        fold_thresholds.append(threshold)
        model = fit_boosting(
            concat_pack(packs, others, "X"),
            concat_pack(packs, others, "y"),
            seed=seed + test_fold,
            iterations=200,
            depth=4,
            learning_rate=0.08,
        )
        scores = predict_boosting(model, packs[test_fold]["eval_X"])
        y_parts.append(packs[test_fold]["eval_y"])
        score_parts.append(scores)
        top_parts.append(packs[test_fold]["eval_top"])
        accept_parts.append(decide(scores, threshold))
    return {
        "fold_thresholds": fold_thresholds,
        "threshold": float(np.mean(fold_thresholds)),
        "outer": outer_metrics(y_parts, score_parts, top_parts, accept_parts),
        "_parts": {"y": y_parts, "scores": score_parts, "top": top_parts, "accept": accept_parts},
    }


def nested_serving(cosine, catboost):
    cosine_parts = cosine["_parts"]
    model_parts = catboost["_parts"]
    accept_parts = [
        cosine_ok & model_ok
        for cosine_ok, model_ok in zip(cosine_parts["accept"], model_parts["accept"], strict=True)
    ]
    score_parts = [
        np.minimum(cosine_score, model_score)
        for cosine_score, model_score in zip(cosine_parts["scores"], model_parts["scores"], strict=True)
    ]
    return {
        "cosine_threshold": round(float(cosine["threshold"]), 4),
        "model_threshold": round(float(catboost["threshold"]), 4),
        "outer": outer_metrics(cosine_parts["y"], score_parts, cosine_parts["top"], accept_parts),
    }


def nested_operating_points(cv: Path, n_folds: int = N_FOLDS, k: int = 10, seed: int = 0, embed_fold=None):
    packs = collect_packs(cv, n_folds=n_folds, k=k, embed_fold=embed_fold)
    cosine = nested_cosine(packs)
    catboost = nested_catboost(packs, seed=seed)
    return {
        "protocol": {
            "train": "balanced_50_50",
            "eval": "open_set_identities",
            "open_set_fraction": OPEN_SET_FRACTION,
        },
        "cosine": public_head(cosine),
        "catboost": public_head(catboost),
        "serving": nested_serving(cosine, catboost),
    }


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


def update_refusal_config(path: Path, threshold: float, key: str = "model_threshold") -> None:
    pattern = re.compile(rf"^{re.escape(key)}:\s*[0-9.eE+-]+\s*$", re.MULTILINE)
    text = Path(path).read_text()
    updated, count = pattern.subn(f"{key}: {threshold:.4f}", text, count=1)
    if count != 1:
        raise ValueError(f"Could not update {key} in {path}")
    Path(path).write_text(updated)


def apply_nested_configs(paths: list[Path], cosine_threshold: float, model_threshold: float) -> None:
    cosine_line = re.compile(r"^cosine_threshold:\s*[0-9.eE+-]+\s*$", re.MULTILINE)
    model_line = re.compile(r"^model_threshold:\s*[0-9.eE+-]+\s*$", re.MULTILINE)
    for path in paths:
        text = Path(path).read_text()
        if cosine_line.search(text):
            update_refusal_config(path, cosine_threshold, key="cosine_threshold")
        if model_line.search(text):
            update_refusal_config(path, model_threshold, key="model_threshold")


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
) -> Path:
    if checkpoint is not None and embed_fold is None:
        embed_fold = bind_checkpoint_embedder(checkpoint, device, weights)
    features, labels, _ = collect_features(cv, n_folds=n_folds, k=k, embed_fold=embed_fold)
    model = fit_boosting(features, labels, seed=seed, iterations=200, depth=4, learning_rate=0.08)
    return save_boosting(model, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=Path, default=DEFAULT_CV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--full-retrain", action="store_true")
    parser.add_argument("--nested", action="store_true")
    parser.add_argument("--nested-output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="ema")
    parser.add_argument("--update-config", nargs="*", type=Path, default=None)
    parser.add_argument("--sha256", action="store_true")
    args = parser.parse_args()
    if args.update_config is not None:
        if args.full_retrain:
            raise ValueError("--update-config cannot be combined with --full-retrain")
        if not args.nested:
            raise ValueError("--update-config requires --nested")
    checkpoint = args.checkpoint
    if args.nested:
        report = nested_operating_points(args.cv, n_folds=args.n_folds, k=args.k, seed=args.seed)
        nested_path = args.nested_output or (Path(args.cv) / "nested_thresholds.json")
        nested_path.parent.mkdir(parents=True, exist_ok=True)
        nested_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report["serving"], indent=2))
        print(f"{args.cv} -> {nested_path}")
        if args.update_config is not None:
            nested_configs = list(args.update_config) if args.update_config else list(DEFAULT_NESTED_CONFIGS)
            apply_nested_configs(
                nested_configs,
                report["serving"]["cosine_threshold"],
                report["serving"]["model_threshold"],
            )
        if not args.full_retrain:
            return
    if args.full_retrain:
        checkpoint = checkpoint or source_checkpoint(None)
    target = export_refusal(
        args.cv,
        args.output,
        seed=args.seed,
        n_folds=args.n_folds,
        k=args.k,
        checkpoint=checkpoint,
        device=args.device,
        weights=args.weights,
    )
    print(f"{args.cv} -> {target} ({target.stat().st_size} bytes)")
    if args.sha256:
        written = write_sha256sums(target.parent)
        print(written)


if __name__ == "__main__":
    main()
