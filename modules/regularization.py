from contextlib import contextmanager

import torch


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].lerp_(value.detach().to(self.shadow[key]), 1 - decay)
            else:
                self.shadow[key].copy_(value.detach())

    @contextmanager
    def apply(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state, device):
        self.decay, self.updates = state["decay"], state["updates"]
        self.shadow = {k: v.to(device) for k, v in state["shadow"].items()}


@contextmanager
def awp(model, cfg):

    backup = {}
    try:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad or p.grad is None or cfg.parameter_pattern not in name:
                    continue
                norm = p.grad.float().norm()
                if not torch.isfinite(norm) or norm == 0:
                    continue
                backup[name] = p.detach().clone()
                perturb = cfg.lr * p.grad / (norm + 1e-6) * (p.detach().float().norm() + 1e-6)
                bound = cfg.eps * p.detach().abs()
                p.add_(perturb.clamp(-bound, bound))
        yield
    finally:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in backup:
                    p.copy_(backup[name])
