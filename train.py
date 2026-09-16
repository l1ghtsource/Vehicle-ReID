import json
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf

from dataset import ReIDDataModule
from dataset.folds import fingerprint
from modules.lightning_module import ReIDModule


def container_dict(value: Any) -> dict[str, Any]:
    container = OmegaConf.to_container(value, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Expected a mapping configuration")
    return {str(key): item for key, item in container.items()}


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg):
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
        initialize_pretrained=cfg.resume is None,
        data_module=dm,
    )
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("data_fingerprint") != fingerprint(dm.folds)
            or checkpoint.get("label_map") != dm.label_map
        ):
            raise ValueError("Resume checkpoint belongs to different data/fold; refusing unsafe resume")
    args = container_dict(cfg.trainer)

    partial = args["limit_val_batches"] != 1.0
    checkpoint = ModelCheckpoint(
        dirpath=out / "checkpoints",
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
            "best_checkpoint": checkpoint.best_model_path,
            "last_checkpoint": checkpoint.last_model_path,
        }
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
