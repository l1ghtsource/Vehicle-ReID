from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, Dataset

import modules.inference as inference
import modules.losses.core as loss_core
from modules.inference import embed_loader
from modules.losses.core import AdaSP, LossCollection, Triplet
from modules.metrics import retrieval_metrics
from modules.optim import build_optimizer, build_scheduler
from modules.regularization import EMA, awp


class OptimModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
        self.head = nn.Linear(3, 2)
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)


def test_optimizer_groups_and_all_schedulers(cfg):
    module = OptimModule()
    cfg.optimizer = OmegaConf.create({"_target_": "torch.optim.SGD", "lr": 0.1, "weight_decay": 0.2})
    cfg.train.backbone_lr_multiplier = 0.5
    cfg.train.layer_decay = 0.8
    optimizer = build_optimizer(module, cfg)
    assert len(optimizer.param_groups) >= 2
    assert any(group["weight_decay"] == 0 for group in optimizer.param_groups)

    cfg.train.no_weight_decay_bias_norm = False
    optimizer = build_optimizer(module, cfg)
    assert all(group["weight_decay"] == 0.2 for group in optimizer.param_groups)
    for kind in ("cosine", "linear", "multistep", "constant"):
        cfg.scheduler.kind = kind
        cfg.scheduler.warmup_epochs = 1
        scheduler = build_scheduler(optimizer, cfg, 2, total_steps=3)
        values = []
        for _ in range(5):
            optimizer.step()
            scheduler.step()
            values.append(scheduler.get_last_lr()[0])
        assert all(np.isfinite(values))
    cfg.scheduler.kind = "invalid"
    scheduler = build_scheduler(optimizer, cfg, 2)
    with pytest.raises(ValueError, match="Unknown scheduler"):
        optimizer.step()
        scheduler.step()
        optimizer.step()
        scheduler.step()


class EmbedModel(nn.Module):
    def __init__(self, as_dict=True, nonfinite=False):
        super().__init__()
        self.as_dict = as_dict
        self.nonfinite = nonfinite

    def forward(self, x):
        embedding = x.mean((-2, -1))
        if self.nonfinite:
            embedding = embedding * torch.nan
        return {"embedding": embedding} if self.as_dict else embedding


class OneImageDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"image": torch.arange(48, dtype=torch.float32).reshape(3, 4, 4)}


class SizeRecordModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sizes = []

    def forward(self, x):
        self.sizes.append((int(x.shape[-2]), int(x.shape[-1])))
        return {"embedding": x.mean((-2, -1))}


class SquareImages(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"image": torch.zeros(3, self.size, self.size)}


def loader():
    return DataLoader(OneImageDataset(), batch_size=1)


def test_embed_loader_tta_and_guards(cfg, monkeypatch):
    cfg.eval.device = "cpu"
    cfg.eval.precision = "fp32"
    cfg.eval.tta.enabled = True
    cfg.eval.tta.scales = [1.0, 0.5]
    cfg.eval.tta.rotations = [0, 10]
    cfg.eval.tta.hflip = True
    result = embed_loader(EmbedModel(), loader(), cfg, torch.device("cpu"))
    assert result.shape == (1, 3)

    cfg.eval.tta.scales = [1.0, 0.9]
    cfg.model.spatial_multiple = 16
    sized = SizeRecordModel()
    embed_loader(sized, DataLoader(SquareImages(256), batch_size=1), cfg, torch.device("cpu"))
    assert (256, 256) in sized.sizes
    assert (224, 224) in sized.sizes
    assert all(height % 16 == 0 and width % 16 == 0 for height, width in sized.sizes)

    cfg.model.backend = "llm2clip"
    cfg.eval.tta.scales = [1.0, 0.9]
    cfg.eval.tta.rotations = [0]
    cfg.eval.tta.hflip = False
    llm = SizeRecordModel()
    embed_loader(llm, DataLoader(SquareImages(336), batch_size=1), cfg, torch.device("cpu"))
    assert llm.sizes == [(336, 336), (336, 336)]
    cfg.model.backend = "timm"

    cfg.eval.tta.enabled = False
    assert embed_loader(EmbedModel(as_dict=False), loader(), cfg, torch.device("cpu")).shape == (1, 3)
    cfg.eval.tta.enabled = True
    for scales, rotations in (([], [0]), ([0], [0]), ([1], [])):
        cfg.eval.tta.scales = scales
        cfg.eval.tta.rotations = rotations
        with pytest.raises(ValueError, match="TTA"):
            embed_loader(EmbedModel(), loader(), cfg, torch.device("cpu"))
    cfg.eval.tta.enabled = False
    with pytest.raises(ValueError, match="Empty"):
        embed_loader(EmbedModel(), [], cfg, torch.device("cpu"))

    cfg.eval.tta.enabled = False
    with pytest.raises(FloatingPointError):
        embed_loader(EmbedModel(nonfinite=True), loader(), cfg, torch.device("cpu"))

    original_to = torch.Tensor.to

    def fake_to(tensor, *args, **kwargs):
        if args and str(args[0]).startswith("cuda"):
            return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    monkeypatch.setattr(inference.torch, "autocast", lambda *args, **kwargs: nullcontext())
    cfg.eval.precision = "bf16"
    assert embed_loader(EmbedModel(), loader(), cfg, "cuda").shape == (1, 3)


def term(name, feature="raw", weight=1.0, params=None):
    return {
        "name": name,
        "feature": feature,
        "weight": weight,
        "params": params or {},
        "miner": None,
    }


def loss_cfg(*terms):
    return OmegaConf.create({"loss": {"terms": list(terms)}})


def test_loss_validation_and_rdrop(monkeypatch):
    with pytest.raises(ValueError, match="Unknown triplet"):
        Triplet(mining="invalid")
    zero = Triplet()(torch.randn(2, 3), torch.tensor([0, 0]))
    assert zero == 0
    with pytest.raises(ValueError, match="Invalid AdaSP"):
        AdaSP(temp=0)
    adasp = AdaSP()
    with pytest.raises(ValueError, match="requires"):
        adasp(torch.randn(3, 4), torch.tensor([0, 0, 1]))

    with pytest.raises(ValueError, match="Unknown loss"):
        LossCollection(loss_cfg(term("invalid")), 4, 2)
    with pytest.raises(ValueError, match="positive"):
        LossCollection(loss_cfg(term("ce", weight=0)), 4, 2)

    features = {"raw": torch.randn(4, 4), "neck": torch.randn(4, 4)}
    first = features
    second = {key: value + 0.1 for key, value in features.items()}
    classifiers = LossCollection(loss_cfg(term("ce"), term("arcface"), term("sphereface2")), 4, 2)
    assert classifiers.rdrop(first, second) >= 0
    metric = LossCollection(loss_cfg(term("triplet")), 4, 2)
    with pytest.raises(ValueError, match="R-Drop"):
        metric.rdrop(first, second)

    class Injected(nn.Module):
        def __init__(self, num_classes, embedding_size):
            super().__init__()
            self.value = num_classes + embedding_size

        def forward(self, x, labels):
            return x.sum() * 0 + self.value

    monkeypatch.setattr(loss_core, "get_class", lambda target: Injected)
    pml = term("pml")
    pml.update(target="unused", inject_dimensions=True)
    collection = LossCollection(loss_cfg(pml), 4, 2)
    assert collection(features, torch.tensor([0, 0, 1, 1]))[0] == 6

    class NanLoss(nn.Module):
        def forward(self, x, labels):
            return x.sum() * torch.nan

    collection.terms = nn.ModuleList([NanLoss()])
    with pytest.raises(FloatingPointError):
        collection(features, torch.tensor([0, 0, 1, 1]))


def test_metrics_validation_and_camera_policy():
    with pytest.raises(ValueError, match="distance"):
        retrieval_metrics([[np.nan]], [1], [1])
    with pytest.raises(ValueError, match="camera"):
        retrieval_metrics([[0.0]], [1], [1], [-1], [0])
    metrics = retrieval_metrics(
        [[0.0, 0.1]],
        [1],
        [1, 1],
        [0],
        [0, 1],
        exclude_all_same_camera=False,
    )
    assert metrics["mAP"] == 1.0


def test_ema_and_awp_all_paths():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    ema = EMA(model, decay=0.9)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    ema.update(model)
    changed = {key: value.clone() for key, value in model.state_dict().items()}
    with ema.apply(model):
        assert any(not torch.equal(model.state_dict()[key], changed[key]) for key in changed)
    for key, value in changed.items():
        assert torch.equal(model.state_dict()[key], value)
    state = ema.state_dict()
    restored = EMA(model)
    restored.load_state_dict(state, torch.device("cpu"))
    assert restored.updates == 1

    linear = nn.Linear(2, 1)
    linear(torch.ones(1, 2)).sum().backward()
    cfg = SimpleNamespace(parameter_pattern="weight", lr=0.1, eps=0.1)
    weight = linear.weight.detach().clone()
    with awp(linear, cfg):
        assert not torch.equal(linear.weight, weight)
    assert torch.equal(linear.weight, weight)
    assert linear.weight.grad is not None
    linear.weight.grad.fill_(torch.nan)
    with awp(linear, cfg):
        assert torch.equal(linear.weight, weight)
    assert before
