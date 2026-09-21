import inspect
import math

import torch
import torch.distributed as dist
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


class DINOHead(nn.Module):
    def __init__(self, dim, hidden_dim, bottleneck_dim, out_dim, nlayers=3):
        super().__init__()
        if nlayers == 1:
            layers: list[nn.Module] = [nn.Linear(dim, bottleneck_dim)]
        else:
            layers = [nn.Linear(dim, hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.prototypes = nn.Linear(bottleneck_dim, out_dim, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.prototypes(F.normalize(self.mlp(x.float()), dim=-1, eps=1e-6))


def sample_ibot_masks(batch, patches, min_ratio=0.1, max_ratio=0.5, probability=0.5, generator=None):
    if batch < 1 or patches < 1:
        raise ValueError("Invalid iBOT mask size")
    if not 0 <= min_ratio < max_ratio <= 1 or not 0 <= probability <= 1:
        raise ValueError("Invalid iBOT mask parameters")
    masks = torch.zeros(batch, patches, dtype=torch.bool)
    for index in range(batch):
        if torch.rand((), generator=generator) < probability:
            ratio = min_ratio + (max_ratio - min_ratio) * float(torch.rand((), generator=generator))
            count = min(patches, max(1, round(patches * ratio)))
            chosen = torch.randperm(patches, generator=generator)[:count]
            masks[index, chosen] = True
    return masks


class DINO(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=2048,
        bottleneck_dim=256,
        out_dim=4096,
        nlayers=3,
        student_temp=0.1,
        teacher_temp=0.07,
        center_momentum=0.9,
        koleo_weight=0.1,
        ibot_weight=1.0,
        gram_weight=1.0,
        ibot_hidden_dim=None,
        ibot_bottleneck_dim=None,
        ibot_out_dim=None,
        ibot_nlayers=3,
        mask_ratio_min=0.1,
        mask_ratio_max=0.5,
        mask_probability=0.5,
        sinkhorn_iters=3,
        teacher_head_momentum=0.996,
        patch_dim=None,
    ):
        super().__init__()
        patch_dim = dim if patch_dim is None else patch_dim
        ibot_hidden = hidden_dim if ibot_hidden_dim is None else ibot_hidden_dim
        ibot_bottleneck = bottleneck_dim if ibot_bottleneck_dim is None else ibot_bottleneck_dim
        ibot_out = out_dim if ibot_out_dim is None else ibot_out_dim
        if (
            dim < 1
            or patch_dim < 1
            or hidden_dim < 1
            or bottleneck_dim < 1
            or out_dim < 2
            or nlayers < 1
            or ibot_hidden < 1
            or ibot_bottleneck < 1
            or ibot_out < 2
            or ibot_nlayers < 1
            or student_temp <= 0
            or teacher_temp <= 0
            or not 0 <= center_momentum < 1
            or koleo_weight < 0
            or ibot_weight < 0
            or gram_weight < 0
            or not 0 <= mask_ratio_min < mask_ratio_max <= 1
            or not 0 <= mask_probability <= 1
            or sinkhorn_iters < 1
            or not 0 <= teacher_head_momentum < 1
        ):
            raise ValueError("Invalid DINO parameters")
        self.student_temp = float(student_temp)
        self.teacher_temp = float(teacher_temp)
        self.center_momentum = float(center_momentum)
        self.koleo_weight = float(koleo_weight)
        self.ibot_weight = float(ibot_weight)
        self.gram_weight = float(gram_weight)
        self.mask_ratio_min = float(mask_ratio_min)
        self.mask_ratio_max = float(mask_ratio_max)
        self.mask_probability = float(mask_probability)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.teacher_head_momentum = float(teacher_head_momentum)
        self.student = DINOHead(dim, hidden_dim, bottleneck_dim, out_dim, nlayers)
        self.teacher = DINOHead(dim, hidden_dim, bottleneck_dim, out_dim, nlayers)
        self.ibot_student = DINOHead(patch_dim, ibot_hidden, ibot_bottleneck, ibot_out, ibot_nlayers)
        self.ibot_teacher = DINOHead(patch_dim, ibot_hidden, ibot_bottleneck, ibot_out, ibot_nlayers)
        self.teacher.load_state_dict(self.student.state_dict())
        self.ibot_teacher.load_state_dict(self.ibot_student.state_dict())
        for parameter in [*self.teacher.parameters(), *self.ibot_teacher.parameters()]:
            parameter.requires_grad_(False)
        self.register_buffer("prototype_center", torch.zeros(1, out_dim))
        self.register_buffer("ibot_center", torch.zeros(1, ibot_out))

    def sample_masks(self, batch, patches, device=None, generator=None):
        masks = sample_ibot_masks(
            batch,
            patches,
            self.mask_ratio_min,
            self.mask_ratio_max,
            self.mask_probability,
            generator,
        )
        return masks if device is None else masks.to(device)

    def _all_reduce(self, value):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value)
        return value

    def _sinkhorn(self, scores):
        q = torch.exp(scores.float() / self.teacher_temp).t()
        total = q.sum().clone()
        q = q / self._all_reduce(total).clamp_min(1e-12)
        n_proto, batch = q.shape
        for _ in range(self.sinkhorn_iters):
            rows = q.sum(dim=1, keepdim=True).clone()
            q = q / self._all_reduce(rows).clamp_min(1e-12)
            q = q / n_proto
            if batch:
                q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
                q = q / batch
        return (q * batch).t()

    def _koleo(self, x):
        if x.shape[0] < 2:
            return x.sum() * 0
        points = F.normalize(x.float(), dim=1)
        with torch.no_grad():
            dots = points @ points.t()
            n = points.shape[0]
            dots.view(-1)[:: n + 1] = -1
            indices = dots.argmax(1)
        distances = (points - points[indices]).norm(dim=1).clamp_min(1e-8)
        return -distances.log().mean()

    def _gram(self, student, teacher):
        student = F.normalize(student.float(), dim=-1)
        teacher = F.normalize(teacher.float(), dim=-1)
        return F.mse_loss(student @ student.transpose(-1, -2), teacher @ teacher.transpose(-1, -2))

    @torch.no_grad()
    def _update_teacher(self):
        momentum = self.teacher_head_momentum
        for source, target in zip(
            [*self.student.parameters(), *self.ibot_student.parameters()],
            [*self.teacher.parameters(), *self.ibot_teacher.parameters()],
            strict=True,
        ):
            target.lerp_(source, 1 - momentum)

    def forward(self, student, teacher, student_patches=None, teacher_patches=None, mask=None):
        if student.shape != teacher.shape or student.shape[0] % 2:
            raise ValueError("DINO expects an even concatenated pair of views")
        student_a, student_b = student.float().chunk(2)
        teacher_a, teacher_b = teacher.float().detach().chunk(2)
        logits_a = self.student(student_a)
        logits_b = self.student(student_b)
        with torch.no_grad():
            teacher_logits_a = self.teacher(teacher_a)
            teacher_logits_b = self.teacher(teacher_b)
            center = self.get_buffer("prototype_center")
            assigned_a = self._sinkhorn(teacher_logits_a - center)
            assigned_b = self._sinkhorn(teacher_logits_b - center)
            updated = torch.cat([teacher_logits_a, teacher_logits_b]).mean(0, keepdim=True)
            center.copy_(center.lerp(updated, 1 - self.center_momentum))
        loss = -0.5 * (
            (assigned_b * F.log_softmax(logits_a / self.student_temp, dim=-1)).sum(dim=1).mean()
            + (assigned_a * F.log_softmax(logits_b / self.student_temp, dim=-1)).sum(dim=1).mean()
        )
        loss = loss + 0.5 * self.koleo_weight * (self._koleo(student_a) + self._koleo(student_b))
        if (
            self.ibot_weight > 0
            and student_patches is not None
            and teacher_patches is not None
            and mask is not None
        ):
            masked_student = student_patches.float()[mask]
            masked_teacher = teacher_patches.float().detach()[mask]
            student_logits = self.ibot_student(masked_student)
            with torch.no_grad():
                teacher_logits = self.ibot_teacher(masked_teacher)
                ibot_center = self.get_buffer("ibot_center")
                assigned = self._sinkhorn(teacher_logits - ibot_center)
                if teacher_logits.shape[0]:
                    ibot_center.copy_(
                        ibot_center.lerp(teacher_logits.mean(0, keepdim=True), 1 - self.center_momentum)
                    )
            token_loss = -(assigned * F.log_softmax(student_logits / self.student_temp, dim=-1)).sum(dim=1)
            loss = loss + self.ibot_weight * (token_loss.sum() / max(token_loss.numel(), 1))
        if (
            self.gram_weight > 0
            and student_patches is not None
            and teacher_patches is not None
            and student_patches.shape[1] > 1
        ):
            loss = loss + self.gram_weight * self._gram(student_patches, teacher_patches.detach())
        with torch.no_grad():
            self._update_teacher()
        return loss


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
    def __init__(self, cfg, dim, classes, patch_dim=None):
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
            elif spec.name == "dino":
                obj = DINO(dim, patch_dim=dim if patch_dim is None else patch_dim, **p)
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

    def forward(self, features, labels, teacher_features=None):
        total, components = features["raw"].sum() * 0, {}
        with torch.autocast(device_type=labels.device.type, enabled=False):
            for i, (spec, term, miner) in enumerate(zip(self.specs, self.terms, self.miners, strict=True)):
                x = features[spec.feature].float()
                if spec.name == "dino":
                    teacher = features if teacher_features is None else teacher_features
                    loss = term(
                        x,
                        teacher[spec.feature].float(),
                        student_patches=features.get("patches"),
                        teacher_patches=teacher.get("patches"),
                        mask=features.get("mask"),
                    )
                elif spec.name == "pml" and not isinstance(miner, nn.Identity):
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
