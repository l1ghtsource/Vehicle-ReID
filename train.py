import json
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf, open_dict
from torch import nn

from dataset import ReIDDataModule
from dataset.folds import fingerprint, split_fingerprint
from eval import SERVING_FORMAT, load_result_keys
from modules.lightning_module import ReIDModule, is_partial_validation
from scripts.aggregate_cv import mean_stop_epochs


def container_dict(value: Any) -> dict[str, Any]:
    container = OmegaConf.to_container(value, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Expected a mapping configuration")
    return {str(key): item for key, item in container.items()}


def load_initial_weights(module: nn.Module, path: str | Path) -> str:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") == SERVING_FORMAT:
        payload = checkpoint.get("state_dict") or {}
        if not payload:
            raise ValueError("Serving payload is missing state_dict")
        state = {f"model.{key}": value for key, value in payload.items()}
        choice = str(checkpoint.get("weights") or "raw")
    else:
        choice = checkpoint.get("validation_weights", "raw")
        if choice == "ema" and "ema" in checkpoint:
            state = {f"model.{key}": value for key, value in checkpoint["ema"]["shadow"].items()}
        else:
            choice = "raw"
            state = {
                key: value for key, value in checkpoint["state_dict"].items() if key.startswith("model.")
            }
    try:
        missing, unexpected = load_result_keys(module.load_state_dict(state, strict=False))
    except RuntimeError as error:
        raise ValueError(
            "Initial checkpoint requires the same model, pooling, and head configuration"
        ) from error
    missing_model = [
        key for key in missing if key.startswith("model.") and not str(key).endswith("mask_token")
    ]
    if missing_model or unexpected:
        raise ValueError(
            f"Initial checkpoint model mismatch: missing={missing_model}, unexpected={unexpected}"
        )
    return choice


def apply_full_retrain(cfg) -> None:
    if not cfg.data.full_retrain:
        return
    with open_dict(cfg):
        if cfg.data.cv_dir:
            cfg.train.epochs = mean_stop_epochs(Path(str(cfg.data.cv_dir)), int(cfg.data.n_folds))
        cfg.trainer.limit_val_batches = 0
        cfg.trainer.num_sanity_val_steps = 0
        cfg.checkpointing.save_last = True


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
    if cfg.resume and cfg.init_checkpoint:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
    apply_full_retrain(cfg)
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("highest" if cfg.trainer.deterministic else "high")
    dm = ReIDDataModule(cfg)
    dm.prepare_data()
    dm.setup("fit")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    dm.save_split(out / "split")
    (out / "label_map.json").write_text(json.dumps(dm.label_map, indent=2))
    model = ReIDModule(
        cfg,
        dm.num_classes,
        initialize_pretrained=cfg.resume is None and cfg.init_checkpoint is None,
        data_module=dm,
    )
    initial_weights = None
    if cfg.init_checkpoint:
        initial_weights = load_initial_weights(model, cfg.init_checkpoint)
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("data_fingerprint") != fingerprint(dm.folds)
            or checkpoint.get("label_map") != dm.label_map
        ):
            raise ValueError("Resume checkpoint belongs to different data/fold; refusing unsafe resume")
        saved_split = checkpoint.get("split_fingerprint")
        if saved_split is not None and saved_split != split_fingerprint(dm.folds):
            raise ValueError("Resume checkpoint belongs to different data/fold; refusing unsafe resume")
    args = container_dict(cfg.trainer)

    partial = is_partial_validation(args["limit_val_batches"])
    ckpt_dir = out / "checkpoints"
    if cfg.resume:
        ckpt_dir = Path(cfg.resume).resolve().parent
    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch{epoch:03d}",
        monitor=None if partial else cfg.checkpointing.monitor,
        mode=cfg.checkpointing.mode,
        save_top_k=0 if partial else cfg.checkpointing.save_top_k,
        save_last=cfg.checkpointing.save_last,
        auto_insert_metric_name=False,
    )
    if args["devices"] != 1 and args["strategy"] == "auto":
        args["strategy"] = "ddp_find_unused_parameters_true"
    trainer = L.Trainer(
        **args,
        max_epochs=cfg.train.epochs,
        use_distributed_sampler=False,
        callbacks=[checkpoint],
        logger=CSVLogger(out, name="logs"),
        default_root_dir=out,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=cfg.resume)
    if trainer.is_global_zero:
        summary = {
            "output_dir": str(out),
            "init_checkpoint": str(cfg.init_checkpoint) if cfg.init_checkpoint else None,
            "initial_weights": initial_weights,
            "best_checkpoint": checkpoint.best_model_path,
            "last_checkpoint": checkpoint.last_model_path,
        }
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
