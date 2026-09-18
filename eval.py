import json
import os
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.errors import HydraException
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from augmentations import build_transforms
from dataset.folds import ensure_folds, fingerprint, query_gallery_split, read_annotations, split_fingerprint
from dataset.images import VehicleDataset
from models import ReIDModel
from modules.inference import embed_loader
from modules.metrics import retrieval_metrics
from postproc import postprocess
from refusal import refusal_accept, write_candidates

DATA_ROOT_PATHS = ("train_csv", "query_csv", "gallery_csv", "image_dir")


def task_overrides() -> list[str]:
    if not HydraConfig.initialized():
        return []
    task = getattr(getattr(HydraConfig.get(), "overrides", None), "task", None)
    if not task:
        return []
    return [str(item) for item in task]


def override_key(item: str) -> str | None:
    text = str(item).strip()
    if not text:
        return None
    try:
        parsed = OverridesParser.create().parse_override(text)
    except HydraException:
        return None
    if parsed.is_delete():
        return None
    key = parsed.key_or_group
    if "." not in key:
        return None
    return key


def remount_data_root(saved, effective, overridden: set[str]) -> None:
    if "data.root" not in overridden:
        return
    old_root = Path(str(saved.data.root))
    new_root = Path(str(effective.data.root))
    for name in DATA_ROOT_PATHS:
        key = f"data.{name}"
        if key in overridden:
            continue
        current = Path(str(OmegaConf.select(effective, key)))
        if current.is_relative_to(old_root):
            OmegaConf.update(effective, key, str(new_root / current.relative_to(old_root)))


def overlay_eval_config(saved, cfg, override_items: list[str] | None = None):
    effective = OmegaConf.merge(
        saved,
        {
            "checkpoint": cfg.checkpoint,
            "eval": {
                "split": cfg.eval.split,
                "device": cfg.eval.device,
                "output_dir": cfg.eval.output_dir,
                "weights": cfg.eval.weights,
                "top_k": cfg.eval.top_k,
                "save_distances": cfg.eval.save_distances,
                "precision": cfg.eval.precision,
            },
            "refusal": cfg.refusal,
            "postproc": {"streaming": cfg.postproc.streaming},
        },
    )
    missing = object()
    overridden: set[str] = set()
    for item in task_overrides() if override_items is None else override_items:
        key = override_key(item)
        if key is None:
            continue
        overridden.add(key)
        value = OmegaConf.select(cfg, key, default=missing)
        if value is missing:
            continue
        OmegaConf.update(effective, key, value, merge=True)
    remount_data_root(saved, effective, overridden)
    return effective


SERVING_FORMAT = "reid-serving"


def saved_cfg(blob):
    if blob.get("format") == SERVING_FORMAT:
        return blob["cfg"]
    if "hyper_parameters" not in blob:
        raise ValueError("Unrecognized checkpoint")
    return blob["hyper_parameters"]["cfg"]


def select_state(blob, choice):
    if blob.get("format") == SERVING_FORMAT:
        exported = str(blob.get("weights", "raw"))
        if choice == "auto":
            choice = exported
        if choice != exported:
            raise ValueError(f"Serving payload contains {exported} weights, requested {choice}")
        state = blob.get("state_dict") or {}
        if not state:
            raise ValueError("Serving payload is missing state_dict")
        return state, choice
    if "hyper_parameters" not in blob:
        raise ValueError("Unrecognized checkpoint")
    if choice == "auto":
        choice = blob.get("validation_weights", "raw")
    if choice == "ema":
        if "ema" not in blob:
            raise ValueError("EMA weights requested but absent from checkpoint")
        return blob["ema"]["shadow"], choice
    if choice == "raw":
        state = {k.removeprefix("model."): v for k, v in blob["state_dict"].items() if k.startswith("model.")}
        if not state:
            raise ValueError("Checkpoint has no model.* weights")
        return state, choice
    raise ValueError("eval.weights must be auto/raw/ema")


def build_serving_payload(blob, weights="ema"):
    if blob.get("format") == SERVING_FORMAT:
        raise ValueError("Input is already a serving payload")
    state, kind = select_state(blob, weights)
    return {
        "format": SERVING_FORMAT,
        "cfg": blob["hyper_parameters"]["cfg"],
        "state_dict": state,
        "weights": kind,
        "label_map": blob.get("label_map"),
        "data_fingerprint": blob.get("data_fingerprint"),
        "split_fingerprint": blob.get("split_fingerprint"),
        "validation_weights": blob.get("validation_weights", kind),
    }


def load_model(cfg, override_items: list[str] | None = None):
    if not cfg.checkpoint:
        raise ValueError("Pass checkpoint=/path/to/weights.pt")
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    saved = OmegaConf.create(saved_cfg(checkpoint))
    effective = overlay_eval_config(saved, cfg, override_items)
    model = ReIDModel(effective, initialize_pretrained=False)
    state, choice = select_state(checkpoint, cfg.eval.weights)
    model.load_state_dict(state, strict=True)
    return model, effective, checkpoint, choice


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    model, cfg, checkpoint, choice = load_model(cfg)
    L.seed_everything(cfg.seed, workers=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(bool(cfg.trainer.deterministic))
    device = torch.device(cfg.eval.device)
    model.to(device).eval()
    if cfg.eval.split == "val":
        folds = ensure_folds(cfg)
        if checkpoint.get("data_fingerprint") != fingerprint(folds):
            raise ValueError("Validation data differs from checkpoint dataset")
        saved_split = checkpoint.get("split_fingerprint")
        if saved_split is not None and saved_split != split_fingerprint(folds):
            raise ValueError("Validation fold assignment differs from the checkpoint split")
        va = folds[folds.fold == cfg.data.fold].reset_index(drop=True)
        label_map = checkpoint.get("label_map") or {}
        train_ids = {int(identity) for identity in label_map}
        if not train_ids:
            raise ValueError("Checkpoint is missing label_map; cannot verify identity-disjoint eval")
        if train_ids & set(va.vehicle_id.astype(int)):
            raise ValueError("Validation identities overlap the checkpoint training split")
        q, g = query_gallery_split(
            va, cfg.seed, cfg.data.validation.query_per_identity, cfg.data.validation.cross_camera
        )
        frame = va
        lookup = pd.Index(frame.image_id)
        q_indices = lookup.get_indexer(q.image_id)
        g_indices = lookup.get_indexer(g.image_id)
    elif cfg.eval.split == "test":
        q, g = read_annotations(cfg.data.query_csv), read_annotations(cfg.data.gallery_csv)
        if set(q.image_id) & set(g.image_id):
            raise ValueError("Query/gallery image overlap: define an explicit self-match policy")
        frame = pd.concat([q, g], ignore_index=True)
        q_indices = np.arange(len(q))
        g_indices = np.arange(len(q), len(frame))
    else:
        raise ValueError("eval.split must be val/test")
    contexts = (
        cfg.eval.tta.context_pcts
        if cfg.eval.tta.enabled and cfg.eval.tta.context_pcts
        else [cfg.data.context_pct]
    )
    views = []
    for context in contexts:
        ds = VehicleDataset(frame, cfg, build_transforms(cfg), context_pct=float(context))
        loader = DataLoader(
            ds,
            batch_size=cfg.data.batch_size_eval,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
        views.append(embed_loader(model, loader, cfg, device))
    emb = np.stack(views).mean(0)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    qe, ge = emb[q_indices], emb[g_indices]
    distance, expanded_q, expanded_g = postprocess(qe, ge, cfg.postproc)
    out = Path(cfg.eval.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "embeddings.npy", emb.astype(np.float32))
    np.save(out / "retrieval_embeddings.npy", np.concatenate([expanded_q, expanded_g]))
    if cfg.eval.save_distances:
        np.save(out / "distances.npy", distance)
    q.to_csv(out / "query.csv", index=False)
    g.to_csv(out / "gallery.csv", index=False)
    frame[["image_id"]].to_csv(out / "embedding_order.csv", index=False)
    if cfg.eval.split == "val":
        oof = frame.copy()
        oof.insert(0, "embedding_index", np.arange(len(oof)))
        oof.to_csv(out / "oof.csv", index=False)
    k = min(int(cfg.eval.top_k), len(g))
    order = np.argsort(distance, axis=1, kind="stable")[:, :k]
    ranked = g.image_id.to_numpy()[order]
    confidence = np.take_along_axis(1.0 - distance, order, axis=1)
    sub = pd.DataFrame(
        ranked,
        columns=pd.Index([f"gallery_id_{j + 1}" for j in range(k)]),
    )
    sub.insert(0, "query_id", q.image_id.to_numpy())
    sub.to_csv(out / "submission.csv", index=False)
    spec = cfg.refusal
    feature_k = int(cfg.eval.top_k) if spec.k is None else int(spec.k)
    embeds = True if spec.with_embeddings is None else bool(spec.with_embeddings)
    kind = "none" if spec.kind is None else str(spec.kind).strip().lower()
    if kind == "off":
        kind = "none"
    accept = refusal_accept(
        spec.kind,
        expanded_q,
        expanded_g,
        cosine_threshold=spec.cosine_threshold,
        model_path=spec.model_path,
        model_threshold=spec.model_threshold,
        rank_threshold=spec.rank_threshold,
        k=feature_k,
        with_embeddings=embeds,
    )
    write_candidates(out / "candidates.csv", q.image_id.to_numpy(), ranked, confidence, accept=accept)
    metadata = {
        "weights": choice,
        "checkpoint": str(cfg.checkpoint),
        "fold": int(cfg.data.fold),
        "n_query": len(q),
        "n_gallery": len(g),
        "streaming": True if cfg.postproc.streaming is None else bool(cfg.postproc.streaming),
        "refusal": {
            "kind": kind,
            "n_accept": int(np.asarray(accept).sum()),
            "n_refuse": int(len(accept) - np.asarray(accept).sum()),
        },
    }
    if cfg.eval.split == "val":
        args = dict(
            qids=q.vehicle_id,
            gids=g.vehicle_id,
            qcams=q.camera_id,
            gcams=g.camera_id,
            ranks=list(cfg.data.validation.ranks),
            cross_camera=cfg.data.validation.cross_camera,
            exclude_all_same_camera=cfg.data.validation.exclude_all_same_camera,
        )
        metadata["raw_metrics"] = retrieval_metrics(1 - qe @ ge.T, **args)
        metadata["metrics"] = retrieval_metrics(distance, **args)
    (out / "metrics.json").write_text(json.dumps(metadata, indent=2))
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
