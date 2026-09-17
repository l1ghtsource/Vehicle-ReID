import math
import warnings
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from dataset.datamodule import ReIDDataModule
from dataset.folds import fingerprint
from models import ReIDModel

from .losses import LossCollection
from .metrics import retrieval_metrics
from .optim import build_optimizer, build_scheduler
from .regularization import EMA, awp


def is_partial_validation(limit: Any) -> bool:
    if limit is None:
        return False
    if isinstance(limit, bool):
        raise TypeError("trainer.limit_val_batches must be an int or float")
    if isinstance(limit, int):
        return True
    if isinstance(limit, float):
        return limit != 1.0
    raise TypeError("trainer.limit_val_batches must be an int or float")


class ReIDModule(L.LightningModule):
    def __init__(
        self,
        cfg,
        num_classes,
        initialize_pretrained=True,
        data_module: ReIDDataModule | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.data_module = data_module
        self.model = ReIDModel(cfg, initialize_pretrained)
        if cfg.model.backend == "hf":
            cfg.model.architecture = OmegaConf.create(self.model.backbone.net.config.to_dict())
        self.save_hyperparameters(
            {"cfg": OmegaConf.to_container(cfg, resolve=True), "num_classes": num_classes}
        )
        self.losses = LossCollection(cfg, self.model.embedding_dim, num_classes)
        self.automatic_optimization = False
        self.ema, self.ema_pending, self.ema_context = None, None, None
        self.val_outputs = []
        self.schedule = None
        if cfg.train.accumulate_grad_batches < 1:
            raise ValueError("accumulate_grad_batches must be >=1")
        if any(t.name == "adasp" for t in cfg.loss.terms) and cfg.data.sampler.kind != "pk":
            raise ValueError("AdaSP requires PK sampler")
        if cfg.train.awp.enabled and cfg.train.accumulate_grad_batches != 1:
            raise ValueError(
                "AWP currently requires train.accumulate_grad_batches=1 (clean per-batch perturbation)"
            )

    def forward(self, x):
        return self.model(x)["embedding"]

    def configure_optimizers(self):
        opt = build_optimizer(self, self.cfg)

        trainer = self._trainer
        if trainer is None:
            raise RuntimeError("configure_optimizers requires an attached Trainer")
        steps = math.ceil(
            trainer.estimated_stepping_batches
            / self.cfg.train.epochs
            / self.cfg.train.accumulate_grad_batches
        )
        self.schedule = build_scheduler(opt, self.cfg, max(1, steps))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": self.schedule, "interval": "step"}}

    def on_fit_start(self):
        if self.cfg.train.ema.enabled:
            self.ema = EMA(self.model, self.cfg.train.ema.decay)
            if self.ema_pending:
                self.ema.load_state_dict(self.ema_pending, self.device)
                self.ema_pending = None

    def on_train_epoch_start(self):
        if self.data_module is None:
            raise RuntimeError("Training requires a ReIDDataModule")
        sampler = self.data_module.train_sampler
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(self.current_epoch)
        self.model.freeze_backbone(self.current_epoch < self.cfg.train.freeze_backbone_epochs)

    def objective(self, batch):
        first = self.model(batch["image"])
        loss, components = self.losses(first, batch["label"])
        if self.cfg.train.rdrop.enabled:
            second = self.model(batch["image"])
            other, _ = self.losses(second, batch["label"])
            kl = self.losses.rdrop(first, second, self.cfg.train.rdrop.temperature)
            loss = 0.5 * (loss + other) + self.cfg.train.rdrop.weight * kl
            components["rdrop"] = kl.detach()
        return loss, components

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        if isinstance(opt, list):
            raise RuntimeError("ReIDModule supports exactly one optimizer")
        accum = self.cfg.train.accumulate_grad_batches
        total_batches = int(self.trainer.num_training_batches)
        group_start = batch_idx // accum * accum
        divisor = min(accum, total_batches - group_start)
        if batch_idx % accum == 0:
            opt.zero_grad(set_to_none=True)
        loss, components = self.objective(batch)
        self.manual_backward(loss / divisor)
        if self.cfg.train.awp.enabled and self.current_epoch >= self.cfg.train.awp.start_epoch:
            buffers = {k: b.clone() for k, b in self.model.named_buffers()}
            try:
                with awp(self.model, self.cfg.train.awp):
                    adv, _ = self.objective(batch)
                    self.manual_backward(self.cfg.train.awp.weight * adv / divisor)
                components["awp"] = adv.detach()
            finally:
                with torch.no_grad():
                    for k, b in self.model.named_buffers():
                        b.copy_(buffers[k])
        if (batch_idx + 1) % accum == 0 or batch_idx + 1 == total_batches:
            self.clip_gradients(
                opt.optimizer,
                gradient_clip_val=self.cfg.train.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
            scaler = getattr(self.trainer.precision_plugin, "scaler", None)
            before = scaler.get_scale() if scaler else None
            opt.step()
            succeeded = scaler is None or scaler.get_scale() >= before
            if succeeded:
                scheduler = self.lr_schedulers()
                if scheduler is None or isinstance(scheduler, list):
                    raise RuntimeError("ReIDModule supports exactly one scheduler")
                scheduler.step()
                if self.ema:
                    self.ema.update(self.model)
        self.log(
            "train/loss",
            loss.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch["label"]),
        )
        for name, value in components.items():
            self.log(
                f"train/{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(batch["label"]),
            )
        self.log("train/lr", opt.optimizer.param_groups[0]["lr"], on_step=True)
        return loss.detach()

    def on_validation_epoch_start(self):
        self.val_outputs = []
        if self.ema and self.cfg.train.ema.validate:
            self.ema_context = self.ema.apply(self.model)
            self.ema_context.__enter__()

    def validation_step(self, batch, batch_idx):
        emb = self(batch["image"])
        self.val_outputs.append((batch["index"].cpu(), emb.detach().float().cpu()))

    def on_validation_epoch_end(self):
        try:
            outputs = self.val_outputs
            if dist.is_available() and dist.is_initialized():
                gathered: list[list[tuple[torch.Tensor, torch.Tensor]] | None] = [
                    None
                ] * dist.get_world_size()
                dist.all_gather_object(gathered, outputs)
                outputs = [
                    item for rank_outputs in gathered if rank_outputs is not None for item in rank_outputs
                ]
            if not outputs:
                raise ValueError("Validation produced no embeddings")
            indices = torch.cat([x[0] for x in outputs]).numpy()
            emb = torch.cat([x[1] for x in outputs]).numpy()

            index, first = np.unique(indices, return_index=True)
            emb = emb[first]
            if self.data_module is None:
                raise RuntimeError("Validation requires a ReIDDataModule")
            dm = self.data_module
            if len(index) != len(dm.val_set):
                warnings.warn(
                    "Partial validation: skipping mAP; set trainer.limit_val_batches=1.0 for honest metrics",
                    stacklevel=2,
                )
                self.log("val/partial", 1.0, sync_dist=False)
                return
            n = dm.num_query
            q, g = dm.query_frame, dm.gallery_frame
            metrics = retrieval_metrics(
                1 - emb[:n] @ emb[n:].T,
                q.vehicle_id,
                g.vehicle_id,
                q.camera_id,
                g.camera_id,
                ranks=list(self.cfg.data.validation.ranks),
                cross_camera=self.cfg.data.validation.cross_camera,
                exclude_all_same_camera=self.cfg.data.validation.exclude_all_same_camera,
            )
            for k, v in metrics.items():
                self.log(f"val/{k}", float(v), sync_dist=False, prog_bar=k in {"mAP", "Rank-1"})
        finally:
            self.val_outputs.clear()
            if self.ema_context:
                self.ema_context.__exit__(None, None, None)
                self.ema_context = None

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint["ema"] = self.ema.state_dict()
        checkpoint["validation_weights"] = "ema" if self.ema and self.cfg.train.ema.validate else "raw"
        if self.data_module is not None:
            checkpoint["data_fingerprint"] = fingerprint(self.data_module.folds)
            checkpoint["label_map"] = self.data_module.label_map

    def on_load_checkpoint(self, checkpoint):
        self.ema_pending = checkpoint.get("ema")

    def on_exception(self, exception):
        if self.ema_context:
            self.ema_context.__exit__(type(exception), exception, exception.__traceback__)
            self.ema_context = None
