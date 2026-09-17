import math

from hydra.utils import instantiate
from torch.optim.lr_scheduler import LambdaLR


def backbone_layer_map(backbone):
    last_numbered = "stem"
    seen_numbered = False
    stages = []
    mapping = {}
    for name, _ in backbone.named_parameters():
        parts = name.split(".")
        numbered = next(
            (".".join(parts[: index + 1]) for index, part in enumerate(parts) if part.isdigit()),
            None,
        )
        if numbered is not None:
            last_numbered = numbered
            seen_numbered = True
            stage = numbered
        elif seen_numbered:
            stage = last_numbered
        else:
            stage = "stem"
        mapping[name] = stage
        if stage not in stages:
            stages.append(stage)
    return stages, mapping


def build_optimizer(module, cfg):
    base_lr = float(cfg.optimizer.lr)
    decay = float(cfg.optimizer.get("weight_decay", 0))
    stages, mapping = backbone_layer_map(module.model.backbone)

    groups = {}
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        lr = base_lr
        if name.startswith("model.backbone."):
            lr *= cfg.train.backbone_lr_multiplier
            stage = mapping[name.removeprefix("model.backbone.")]
            layer = stages.index(stage)
            lr *= cfg.train.layer_decay ** (len(stages) - 1 - layer)
        wd = 0.0 if cfg.train.no_weight_decay_bias_norm and (p.ndim <= 1 or name.endswith("bias")) else decay
        groups.setdefault((lr, wd), []).append(p)
    return instantiate(
        cfg.optimizer,
        _convert_="all",
        params=[{"params": p, "lr": lr, "weight_decay": wd} for (lr, wd), p in groups.items()],
    )


def build_scheduler(optimizer, cfg, steps_per_epoch, total_steps=None):
    c = cfg.scheduler
    warm = round(c.warmup_epochs * steps_per_epoch)
    total = max(1, int(total_steps if total_steps is not None else cfg.train.epochs * steps_per_epoch))

    def factor(step):
        if step < warm:
            return c.warmup_start_factor + (1 - c.warmup_start_factor) * step / max(1, warm)
        progress = min(1.0, max(0.0, (step - warm) / max(1, total - warm)))
        if c.kind == "cosine":
            return c.min_lr_ratio + (1 - c.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        if c.kind == "linear":
            return c.min_lr_ratio + (1 - c.min_lr_ratio) * (1 - progress)
        if c.kind == "multistep":
            return c.gamma ** sum(step >= m * steps_per_epoch for m in c.milestones)
        if c.kind == "constant":
            return 1.0
        raise ValueError(f"Unknown scheduler {c.kind}")

    return LambdaLR(optimizer, factor)
