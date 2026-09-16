import json
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from augmentations import build_transforms
from dataset.folds import ensure_folds, fingerprint, query_gallery_split, read_annotations
from dataset.images import VehicleDataset
from models import ReIDModel
from modules.inference import embed_loader
from modules.metrics import retrieval_metrics
from postproc import postprocess


def load_model(cfg):
    if not cfg.checkpoint:
        raise ValueError("Pass checkpoint=/path/to/checkpoint.ckpt")
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    saved = OmegaConf.create(checkpoint["hyper_parameters"]["cfg"])

    effective = OmegaConf.merge(
        saved,
        {
            "eval": OmegaConf.to_container(cfg.eval, resolve=True),
            "postproc": OmegaConf.to_container(cfg.postproc, resolve=True),
            "checkpoint": cfg.checkpoint,
            "data": {
                "root": cfg.data.root,
                "train_csv": cfg.data.train_csv,
                "query_csv": cfg.data.query_csv,
                "gallery_csv": cfg.data.gallery_csv,
                "image_dir": cfg.data.image_dir,
                "num_workers": cfg.data.num_workers,
                "batch_size_eval": cfg.data.batch_size_eval,
                "verify_files": cfg.data.verify_files,
            },
        },
    )
    model = ReIDModel(effective, initialize_pretrained=False)
    choice = cfg.eval.weights
    if choice == "auto":
        choice = checkpoint.get("validation_weights", "raw")
    if choice == "ema":
        if "ema" not in checkpoint:
            raise ValueError("EMA weights requested but absent from checkpoint")
        state = checkpoint["ema"]["shadow"]
    elif choice == "raw":
        state = {
            k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")
        }
    else:
        raise ValueError("eval.weights must be auto/raw/ema")
    model.load_state_dict(state, strict=True)
    return model, effective, checkpoint, choice


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    model, cfg, checkpoint, choice = load_model(cfg)
    L.seed_everything(cfg.seed, workers=True)
    torch.use_deterministic_algorithms(bool(cfg.trainer.deterministic))
    device = torch.device(cfg.eval.device)
    model.to(device).eval()
    if cfg.eval.split == "val":
        folds = ensure_folds(cfg)
        if checkpoint.get("data_fingerprint") != fingerprint(folds):
            raise ValueError("Validation data differs from checkpoint dataset")
        va = folds[folds.fold == cfg.data.fold]
        q, g = query_gallery_split(
            va, cfg.seed, cfg.data.validation.query_per_identity, cfg.data.validation.cross_camera
        )
    elif cfg.eval.split == "test":
        q, g = read_annotations(cfg.data.query_csv), read_annotations(cfg.data.gallery_csv)
        if set(q.image_id) & set(g.image_id):
            raise ValueError("Query/gallery image overlap: define an explicit self-match policy")
    else:
        raise ValueError("eval.split must be val/test")
    frame = pd.concat([q, g], ignore_index=True)
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
    qe, ge = emb[: len(q)], emb[len(q) :]
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
    k = min(int(cfg.eval.top_k), len(g))
    order = np.argsort(distance, axis=1, kind="stable")[:, :k]
    sub = pd.DataFrame(
        g.image_id.to_numpy()[order],
        columns=pd.Index([f"gallery_id_{j + 1}" for j in range(k)]),
    )
    sub.insert(0, "query_id", q.image_id.to_numpy())
    sub.to_csv(out / "submission.csv", index=False)
    metadata = {
        "weights": choice,
        "checkpoint": str(cfg.checkpoint),
        "fold": int(cfg.data.fold),
        "n_query": len(q),
        "n_gallery": len(g),
        "transductive": bool(cfg.postproc.enabled),
        "refusal": "not implemented (deferred by request)",
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
