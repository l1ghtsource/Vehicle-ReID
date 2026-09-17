import inspect
import math

import torch
from hydra.utils import get_class, instantiate
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import TripletMarginLoss
from torch import nn
from torch.nn import functional as F

from third_party.adasp.adasp_portable import AdaSPLoss
from third_party.opensphere.sphereface2 import SphereFace2


class AngularClassifier(nn.Module):
    def __init__(self, dim, classes, kind="arcface", margin=0.3, scale=32, label_smoothing=0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, dim))
        nn.init.xavier_uniform_(self.weight)
        self.kind, self.margin, self.scale, self.smooth = kind, margin, scale, label_smoothing

    def logits(self, x):
        return self.scale * F.linear(F.normalize(x.float(), dim=1), F.normalize(self.weight.float(), dim=1))

    def forward(self, x, y):
        cosine = (self.logits(x) / self.scale).clamp(-1 + 1e-7, 1 - 1e-7)
        if self.kind == "arcface":
            m = self.margin
            phi = cosine * math.cos(m) - (1 - cosine.square()).sqrt() * math.sin(m)
            phi = torch.where(cosine > math.cos(math.pi - m), phi, cosine - math.sin(math.pi - m) * m)
        else:
            phi = cosine - self.margin
        one = F.one_hot(y, cosine.shape[1]).to(cosine)
        logits = self.scale * (one * phi + (1 - one) * cosine)
        return F.cross_entropy(logits, y, label_smoothing=self.smooth)


class CEClassifier(nn.Module):
    def __init__(self, dim, classes, label_smoothing=0.1):
        super().__init__()
        self.classifier = nn.Linear(dim, classes, bias=False)
        self.smooth = label_smoothing

    def logits(self, x):
        return self.classifier(x)

    def forward(self, x, y):
        return F.cross_entropy(self.logits(x), y, label_smoothing=self.smooth)


class Triplet(nn.Module):
    def __init__(self, margin=0.3, soft_margin=False, normalize=True, mining="hard"):
        super().__init__()
        self.margin, self.soft, self.normalize, self.mining = margin, soft_margin, normalize, mining
        if mining not in {"hard", "semihard", "all", "weighted"}:
            raise ValueError(f"Unknown triplet mining: {mining}")

    def forward(self, x, y):
        x = F.normalize(x.float(), dim=1) if self.normalize else x.float()
        dist = torch.cdist(x, x, p=2)
        pos = y[:, None].eq(y[None, :]) & ~torch.eye(len(y), device=y.device, dtype=torch.bool)
        neg = y[:, None].ne(y[None, :])
        valid = pos.any(1) & neg.any(1)
        if not valid.any():
            return x.sum() * 0
        if self.mining == "all":
            return TripletMarginLoss(
                margin=self.margin,
                smooth_loss=self.soft,
                distance=LpDistance(normalize_embeddings=False),
            )(x, y)
        if self.mining == "weighted":
            dp = (torch.softmax(dist.masked_fill(~pos, -1e9), 1) * dist).sum(1)
            dn = (torch.softmax((-dist).masked_fill(~neg, -1e9), 1) * dist).sum(1)
        else:
            dp = dist.masked_fill(~pos, -torch.inf).amax(1)
            if self.mining == "semihard":
                mask = neg & (dist > dp[:, None]) & (dist < dp[:, None] + self.margin)
                dn = dist.masked_fill(~mask, torch.inf).amin(1)

                dn = torch.where(torch.isfinite(dn), dn, dist.masked_fill(~neg, torch.inf).amin(1))
            else:
                dn = dist.masked_fill(~neg, torch.inf).amin(1)
        delta = dp[valid] - dn[valid]
        return F.softplus(delta).mean() if self.soft else F.relu(delta + self.margin).mean()


class AdaSP(nn.Module):
    def __init__(self, temp=0.04, loss_type="adasp"):
        super().__init__()
        if temp <= 0 or loss_type not in {"adasp", "sp-h", "sp-lh"}:
            raise ValueError("Invalid AdaSP parameters")
        self.loss = AdaSPLoss(temp=temp, loss_type=loss_type)

    def forward(self, x, y):
        _, count = y.unique(return_counts=True)
        if len(count) < 2 or count.min() < 2 or not torch.all(count == count[0]):
            raise ValueError("Original AdaSP requires >=2 identities and equal K>=2 per identity; use PK")
        order = y.argsort(stable=True)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.loss(x[order].double(), y[order]).float()


class LossCollection(nn.Module):
    def __init__(self, cfg, dim, classes):
        super().__init__()
        self.specs = cfg.loss.terms
        self.terms, self.miners = nn.ModuleList(), nn.ModuleList()
        for spec in self.specs:
            p = dict(spec.get("params", {}))
            if spec.name in {"arcface", "cosface"}:
                obj = AngularClassifier(dim, classes, kind=spec.name, **p)
            elif spec.name == "ce":
                obj = CEClassifier(dim, classes, **p)
            elif spec.name == "sphereface2":
                obj = SphereFace2(dim, classes, **p)
            elif spec.name == "triplet":
                obj = Triplet(**p)
            elif spec.name == "adasp":
                obj = AdaSP(**p)
            elif spec.name == "pml":
                cls = get_class(spec.target)
                if spec.get("inject_dimensions", False):
                    parameters = inspect.signature(cls).parameters
                    if "num_classes" in parameters:
                        p["num_classes"] = classes
                    if "embedding_size" in parameters:
                        p["embedding_size"] = dim
                obj = cls(**p)
            else:
                raise ValueError(f"Unknown loss {spec.name}")
            self.terms.append(obj)
            self.miners.append(instantiate(spec.miner) if spec.get("miner") else nn.Identity())
        if not self.terms or not any(s.weight > 0 for s in self.specs):
            raise ValueError("At least one positive loss weight is required")

    def forward(self, features, labels):
        total, components = features["raw"].sum() * 0, {}
        with torch.autocast(device_type=labels.device.type, enabled=False):
            for i, (spec, term, miner) in enumerate(zip(self.specs, self.terms, self.miners, strict=True)):
                x = features[spec.feature].float()
                if spec.name == "pml" and not isinstance(miner, nn.Identity):
                    loss = term(x, labels, miner(x, labels))
                else:
                    loss = term(x, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {spec.name} loss")
                components[f"{i}_{spec.name}"] = loss.detach()
                total = total + spec.weight * loss
        return total, components

    def rdrop(self, first, second, temperature=1.0):
        pairs = []
        for spec, term in zip(self.specs, self.terms, strict=True):
            if isinstance(term, (AngularClassifier, CEClassifier)):
                a, b = term.logits(first[spec.feature]), term.logits(second[spec.feature])
            elif isinstance(term, SphereFace2):
                a = (
                    term.r
                    * F.linear(
                        F.normalize(first[spec.feature].float(), dim=1), F.normalize(term.w.float(), dim=1)
                    )
                    + term.b
                )
                b = (
                    term.r
                    * F.linear(
                        F.normalize(second[spec.feature].float(), dim=1), F.normalize(term.w.float(), dim=1)
                    )
                    + term.b
                )
            else:
                continue
            a, b = a.float() / temperature, b.float() / temperature
            pairs.append(
                0.5
                * (
                    F.kl_div(a.log_softmax(-1), b.softmax(-1), reduction="batchmean")
                    + F.kl_div(b.log_softmax(-1), a.softmax(-1), reduction="batchmean")
                )
                * temperature**2
            )
        if not pairs:
            raise ValueError(
                "R-Drop KL needs CE/ArcFace/CosFace/SphereFace2; pure metric losses have no class logits"
            )
        return torch.stack(pairs).mean()
