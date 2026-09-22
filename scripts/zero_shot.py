import argparse
import json
import os
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dataset.folds import query_gallery_split, read_annotations
from models import ReIDModel
from modules.inference import embed_frame
from modules.metrics import retrieval_metrics
from postproc import postprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def load_config(model: str, overrides: list[str]):
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(
            config_name="config",
            overrides=[
                f"model={model}",
                "model.head.embedding_dim=null",
                "model.head.bnneck=false",
                "model.head.retrieval_feature=raw",
                "model.head.dropout=0.0",
                "model.head.local_parts=0",
                "model.pooling.kind=gap",
                f"eval.output_dir=artifacts/zero_shot/{model}",
                "eval.save_distances=false",
                *overrides,
            ],
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.model, args.overrides)
    L.seed_everything(cfg.seed, workers=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(bool(cfg.trainer.deterministic))
    frame = read_annotations(cfg.data.train_csv, labeled=True)
    query, gallery = query_gallery_split(
        frame,
        cfg.seed,
        cfg.data.validation.query_per_identity,
        cfg.data.validation.cross_camera,
    )
    combined = pd.concat([query, gallery], ignore_index=True)
    device = torch.device(cfg.eval.device)
    model = ReIDModel(cfg).to(device).eval()
    emb = embed_frame(model, cfg, device, combined)
    qe, ge = emb[: len(query)], emb[len(query) :]
    distance, expanded_q, expanded_g = postprocess(qe, ge, cfg.postproc)
    metric_args = dict(
        qids=query.vehicle_id,
        gids=gallery.vehicle_id,
        qcams=query.camera_id,
        gcams=gallery.camera_id,
        ranks=list(cfg.data.validation.ranks),
        cross_camera=cfg.data.validation.cross_camera,
        exclude_all_same_camera=cfg.data.validation.exclude_all_same_camera,
    )
    out = Path(cfg.eval.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb.astype(np.float32))
    np.save(out / "retrieval_embeddings.npy", np.concatenate([expanded_q, expanded_g]))
    if cfg.eval.save_distances:
        np.save(out / "distances.npy", distance)
    query.to_csv(out / "query.csv", index=False)
    gallery.to_csv(out / "gallery.csv", index=False)
    combined[["image_id"]].to_csv(out / "embedding_order.csv", index=False)
    metadata = {
        "model": args.model,
        "pretrained": bool(cfg.model.pretrained),
        "checkpoint_path": cfg.model.checkpoint_path,
        "n_query": len(query),
        "n_gallery": len(gallery),
        "n_train": len(frame),
        "streaming": True if cfg.postproc.streaming is None else bool(cfg.postproc.streaming),
        "raw_metrics": retrieval_metrics(1 - qe @ ge.T, **metric_args),
        "metrics": retrieval_metrics(distance, **metric_args),
    }
    (out / "metrics.json").write_text(json.dumps(metadata, indent=2))
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
