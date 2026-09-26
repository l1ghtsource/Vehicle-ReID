import json
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf, open_dict

from dataset import PretrainDataModule
from dataset.folds import fingerprint, split_fingerprint
from dataset.pretrain import ssl_image_mode
from models.backbones import TOKEN_MASK_BACKENDS
from modules.lightning_module import ReIDModule, is_partial_validation


def container_dict(value: Any) -> dict[str, Any]:
    container = OmegaConf.to_container(value, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Expected a mapping configuration")
    return {str(key): item for key, item in container.items()}


def apply_ssl_pretrain(cfg) -> None:
    selected = [str(name).lower() for name in cfg.pretrain.datasets]
    modes = [ssl_image_mode(name) for name in selected]
    if not any(mode is not None for mode in modes):
        return
    if any(mode is None for mode in modes) or len(selected) != 1:
        raise ValueError("test_train_ssl cannot mix with labeled extra datasets")
    if not cfg.train.ema.enabled:
        raise ValueError("DINO SSL requires train.ema.enabled")
    dino = OmegaConf.load(Path(__file__).resolve().parent / "configs/loss/dino.yaml")
    with open_dict(cfg):
        cfg.loss = dino
        cfg.data.sampler.kind = "random"
        if str(cfg.model.backend) not in TOKEN_MASK_BACKENDS:
            for term in cfg.loss.terms:
                if term.name == "dino":
                    term.params.ibot_weight = 0.0


class PeriodicLastCheckpoint(Callback):
    def __init__(self, dirpath: Path, every_n_epochs: int, keep_epochs=()):
        self.dirpath = Path(dirpath)
        self.every = max(1, int(every_n_epochs))
        self.keep_epochs = frozenset(int(epoch) for epoch in keep_epochs)
        if any(epoch < 1 for epoch in self.keep_epochs):
            raise ValueError("checkpointing.keep_epochs must contain positive completed epoch counts")

    def _save(self, trainer, filename: str) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(self.dirpath / filename)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        completed = trainer.current_epoch + 1
        if completed % self.every == 0:
            self._save(trainer, "last.ckpt")
        if completed in self.keep_epochs:
            self._save(trainer, f"milestone_epoch{completed:03d}.ckpt")

    def on_train_end(self, trainer, pl_module) -> None:
        self._save(trainer, "last.ckpt")


@hydra.main(version_base="1.3", config_path="configs", config_name="pretrain")
def main(cfg):
    apply_ssl_pretrain(cfg)
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("highest" if cfg.trainer.deterministic else "high")
    dm = PretrainDataModule(cfg)
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
            raise ValueError("Resume checkpoint belongs to different pretraining data")
        saved_split = checkpoint.get("split_fingerprint")
        if saved_split is not None and saved_split != split_fingerprint(dm.folds):
            raise ValueError("Resume checkpoint belongs to different pretraining data")
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
        save_last=False,
        auto_insert_metric_name=False,
    )
    last_checkpoint_cb = PeriodicLastCheckpoint(
        ckpt_dir, cfg.checkpointing.last_every_n_epochs, cfg.checkpointing.keep_epochs
    )
    if args["devices"] != 1 and args["strategy"] == "auto":
        args["strategy"] = "ddp_find_unused_parameters_true"
    trainer = L.Trainer(
        **args,
        max_epochs=cfg.train.epochs,
        use_distributed_sampler=False,
        callbacks=[checkpoint, last_checkpoint_cb],
        logger=CSVLogger(out, name="logs"),
        default_root_dir=out,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=cfg.resume)
    if trainer.is_global_zero:
        if dm.validation_frame is None:
            raise RuntimeError("Validation data was not prepared")
        summary = {
            "output_dir": str(out),
            "datasets": list(cfg.pretrain.datasets),
            "train_images": len(dm.train_frame),
            "train_identities": dm.num_classes,
            "validation_images": len(dm.validation_frame),
            "best_checkpoint": checkpoint.best_model_path,
            "last_checkpoint": str(ckpt_dir / "last.ckpt"),
        }
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
