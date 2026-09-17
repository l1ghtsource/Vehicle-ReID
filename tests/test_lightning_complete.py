from types import SimpleNamespace
from typing import cast

import lightning as L
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import modules.lightning_module as lightning_module
from dataset import ReIDDataModule
from modules.lightning_module import ReIDModule, is_partial_validation, scheduler_horizon
from modules.regularization import EMA


class TinyConfig(nn.Module):
    def to_dict(self):
        return {"model_type": "tiny"}


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TinyConfig()


class TinyReID(nn.Module):
    def __init__(self, cfg, initialize_pretrained=True):
        super().__init__()
        self.linear = nn.Linear(3, 4)
        self.backbone = nn.Linear(3, 3)
        self.backbone.net = TinyNet()
        self.register_buffer("running", torch.ones(1))
        self.embedding_dim = 4
        self.frozen = False

    def freeze_backbone(self, frozen):
        self.frozen = frozen

    def forward(self, image):
        pooled = image.mean((-2, -1))
        raw = self.linear(pooled)
        return {"raw": raw, "neck": raw, "embedding": torch.nn.functional.normalize(raw, dim=1)}


def make_module(cfg, monkeypatch, data_module=None):
    monkeypatch.setattr(lightning_module, "ReIDModel", TinyReID)
    module = ReIDModule(cfg, 2, data_module=data_module)
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    return module


def trainer(**kwargs):
    values = {
        "estimated_stepping_batches": 4,
        "num_training_batches": 2,
        "current_epoch": 0,
        "precision_plugin": SimpleNamespace(scaler=None),
    }
    values.update(kwargs)
    return cast(L.Trainer, SimpleNamespace(**values))


def test_scheduler_horizon_uses_max_steps_without_double_accumulation(cfg):
    cfg.train.epochs = 2
    cfg.train.accumulate_grad_batches = 2
    per_epoch, total = scheduler_horizon(trainer(num_training_batches=3, max_steps=3), cfg)
    assert per_epoch == 2
    assert total == 3
    per_epoch, total = scheduler_horizon(trainer(num_training_batches=3, max_steps=100), cfg)
    assert per_epoch == 2
    assert total == 4
    per_epoch, total = scheduler_horizon(trainer(num_training_batches=float("inf"), max_steps=7), cfg)
    assert per_epoch == 1
    assert total == 7
    per_epoch, total = scheduler_horizon(
        trainer(num_training_batches=float("inf"), max_steps=5, train_dataloader=object()),
        cfg,
    )
    assert total == 5
    per_epoch, total = scheduler_horizon(trainer(num_training_batches=None, max_steps=None), cfg)
    assert per_epoch == 1
    assert total == 1


def test_scheduler_horizon_setups_train_loader_when_batches_unknown(cfg):
    fake = SimpleNamespace(num_training_batches=float("inf"), max_steps=-1, train_dataloader=None)

    class Loop:
        def setup_data(self):
            fake.num_training_batches = 3
            fake.train_dataloader = object()

    fake.fit_loop = Loop()
    cfg.train.epochs = 2
    cfg.train.accumulate_grad_batches = 2
    per_epoch, total = scheduler_horizon(cast(L.Trainer, fake), cfg)
    assert per_epoch == 2
    assert total == 4


def test_scheduler_horizon_from_trainer_fit(data_cfg, monkeypatch):
    monkeypatch.setattr(lightning_module, "ReIDModel", TinyReID)
    data_cfg.train.epochs = 2
    data_cfg.train.accumulate_grad_batches = 2
    data_cfg.data.sampler.steps_per_epoch = 3
    data_cfg.train.rdrop.enabled = False
    data_cfg.train.awp.enabled = False
    data_cfg.train.ema.enabled = False
    dm = ReIDDataModule(data_cfg)
    dm.prepare_data()
    dm.setup("fit")
    module = ReIDModule(data_cfg, dm.num_classes, data_module=dm)
    captured = {}
    original = lightning_module.build_scheduler

    def capture(optimizer, cfg, steps_per_epoch, total_steps=None):
        captured["per_epoch"] = steps_per_epoch
        captured["total"] = total_steps
        return original(optimizer, cfg, steps_per_epoch, total_steps)

    monkeypatch.setattr(lightning_module, "build_scheduler", capture)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        max_steps=-1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=dm)
    assert captured["per_epoch"] == 2
    assert captured["total"] == 4


def test_partial_validation_limit_types():
    assert is_partial_validation(None) is False
    assert is_partial_validation(1.0) is False
    assert is_partial_validation(1) is True
    assert is_partial_validation(2) is True
    assert is_partial_validation(0.5) is True
    with pytest.raises(TypeError, match="int or float"):
        is_partial_validation(True)
    with pytest.raises(TypeError, match="int or float"):
        is_partial_validation("1.0")


def batch():
    return {
        "image": torch.randn(4, 3, 2, 2),
        "label": torch.tensor([0, 0, 1, 1]),
        "index": torch.arange(4),
    }


def test_constructor_guards_and_hf_config(cfg, monkeypatch):
    cfg.model.backend = "hf"
    module = make_module(cfg, monkeypatch)
    assert cfg.model.architecture.model_type == "tiny"
    assert module(batch()["image"]).shape == (4, 4)

    cfg.train.accumulate_grad_batches = 0
    with pytest.raises(ValueError, match="accumulate"):
        make_module(cfg, monkeypatch)
    cfg.train.accumulate_grad_batches = 1
    cfg.loss.terms = [OmegaConf.create({"name": "adasp", "weight": 1, "feature": "raw", "params": {}})]
    cfg.data.sampler.kind = "random"
    with pytest.raises(ValueError, match="PK"):
        make_module(cfg, monkeypatch)
    cfg.loss.terms = [
        OmegaConf.create(
            {
                "name": "ce",
                "weight": 1,
                "feature": "raw",
                "params": {},
                "miner": None,
            }
        )
    ]
    cfg.data.sampler.kind = "pk"
    cfg.train.awp.enabled = True
    cfg.train.accumulate_grad_batches = 2
    with pytest.raises(ValueError, match="AWP"):
        make_module(cfg, monkeypatch)


def test_configure_and_fit_hooks(cfg, monkeypatch):
    module = make_module(cfg, monkeypatch)
    module._trainer = trainer()
    configured = module.configure_optimizers()
    assert configured["optimizer"]
    cfg.train.ema.enabled = True
    module.ema_pending = EMA(module.model).state_dict()
    module.on_fit_start()
    assert module.ema is not None
    assert module.ema_pending is None

    with pytest.raises(RuntimeError, match="ReIDDataModule"):
        module.on_train_epoch_start()
    sampler = SimpleNamespace(epoch=None, set_epoch=lambda epoch: setattr(sampler, "epoch", epoch))
    module.data_module = SimpleNamespace(train_sampler=sampler)
    cfg.train.freeze_backbone_epochs = 1
    module.on_train_epoch_start()
    assert sampler.epoch == 0
    assert module.model.frozen

    module._trainer = None
    with pytest.raises(RuntimeError, match="Trainer"):
        module.configure_optimizers()


class FakeOptimizer:
    def __init__(self, parameters):
        self.optimizer = torch.optim.SGD(parameters, lr=0.1)
        self.steps = 0

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.optimizer.step()
        self.steps += 1


def test_training_step_all_paths(cfg, monkeypatch):
    module = make_module(cfg, monkeypatch)
    module._trainer = trainer()
    optimizer = FakeOptimizer(module.parameters())
    scheduler = SimpleNamespace(steps=0, step=lambda: setattr(scheduler, "steps", scheduler.steps + 1))
    monkeypatch.setattr(module, "optimizers", lambda: optimizer)
    monkeypatch.setattr(module, "lr_schedulers", lambda: scheduler)
    monkeypatch.setattr(module, "manual_backward", lambda loss: loss.backward())
    monkeypatch.setattr(module, "clip_gradients", lambda *args, **kwargs: None)
    module.ema = EMA(module.model)
    cfg.train.rdrop.enabled = True
    cfg.train.accumulate_grad_batches = 2
    result = module.training_step(batch(), 0)
    assert torch.isfinite(result)
    assert optimizer.steps == 0
    result = module.training_step(batch(), 1)
    assert torch.isfinite(result)
    assert optimizer.steps == 1
    assert scheduler.steps == 1

    monkeypatch.setattr(module, "optimizers", lambda: [optimizer])
    with pytest.raises(RuntimeError, match="one optimizer"):
        module.training_step(batch(), 0)

    monkeypatch.setattr(module, "optimizers", lambda: optimizer)
    monkeypatch.setattr(module, "lr_schedulers", lambda: [])
    with pytest.raises(RuntimeError, match="one scheduler"):
        module.training_step(batch(), 1)


class Scale:
    def __init__(self, values):
        self.values = iter(values)

    def get_scale(self):
        return next(self.values)


def test_training_awp_and_skipped_scaler(cfg, monkeypatch):
    cfg.train.awp.enabled = True
    cfg.train.awp.start_epoch = 0
    module = make_module(cfg, monkeypatch)
    optimizer = FakeOptimizer(module.parameters())
    module._trainer = trainer(precision_plugin=SimpleNamespace(scaler=Scale([2.0, 1.0])))
    monkeypatch.setattr(module, "optimizers", lambda: optimizer)
    monkeypatch.setattr(module, "manual_backward", lambda loss: loss.backward())
    monkeypatch.setattr(module, "clip_gradients", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "lr_schedulers", lambda: SimpleNamespace(step=lambda: None))
    module.training_step(batch(), 1)
    assert optimizer.steps == 1


def validation_data():
    query = pd.DataFrame({"vehicle_id": [1, 2], "camera_id": [0, 0]})
    gallery = pd.DataFrame({"vehicle_id": [1, 2], "camera_id": [1, 1]})
    return SimpleNamespace(
        val_set=range(4),
        num_query=2,
        query_frame=query,
        gallery_frame=gallery,
        folds=pd.DataFrame({"image_id": ["a"], "vehicle_id": [1], "fold": [0]}),
        label_map={1: 0},
    )


def test_validation_full_partial_empty_and_distributed(cfg, monkeypatch):
    data = validation_data()
    module = make_module(cfg, monkeypatch, data)
    module._trainer = trainer()
    module.validation_step(batch(), 0)
    assert len(module.val_outputs) == 1
    module.on_validation_epoch_end()
    assert module.val_outputs == []

    module.val_outputs = [(torch.tensor([0]), torch.ones(1, 4))]
    with pytest.warns(UserWarning, match="Partial"):
        module.on_validation_epoch_end()

    with pytest.raises(ValueError, match="no embeddings"):
        module.on_validation_epoch_end()

    module.val_outputs = [(torch.arange(4), torch.eye(4))]
    monkeypatch.setattr(lightning_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(lightning_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(lightning_module.dist, "get_world_size", lambda: 1)

    def gather(target, outputs):
        target[0] = outputs

    monkeypatch.setattr(lightning_module.dist, "all_gather_object", gather)
    module.on_validation_epoch_end()

    module.data_module = None
    module.val_outputs = [(torch.arange(4), torch.eye(4))]
    with pytest.raises(RuntimeError, match="Validation"):
        module.on_validation_epoch_end()


def test_ema_validation_checkpoint_and_exception(cfg, monkeypatch):
    data = validation_data()
    module = make_module(cfg, monkeypatch, data)
    module._trainer = trainer()
    module.ema = EMA(module.model)
    cfg.train.ema.validate = True
    module.on_validation_epoch_start()
    assert module.ema_context is not None
    module.val_outputs = [(torch.arange(4), torch.eye(4))]
    module.on_validation_epoch_end()
    assert module.ema_context is None

    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["validation_weights"] == "ema"
    assert "data_fingerprint" in checkpoint
    assert "split_fingerprint" in checkpoint
    module.on_load_checkpoint(checkpoint)
    assert module.ema_pending is not None

    module.on_validation_epoch_start()
    error = RuntimeError("stop")
    module.on_exception(error)
    assert module.ema_context is None

    module.ema = None
    module.data_module = None
    raw = {}
    module.on_save_checkpoint(raw)
    assert raw["validation_weights"] == "raw"
