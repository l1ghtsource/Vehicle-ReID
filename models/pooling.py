import torch
from torch import nn


class Pool(nn.Module):
    def __init__(self, channels: int, cfg, num_prefix_tokens=0):
        super().__init__()
        self.kind = cfg.kind
        self.prefix = int(num_prefix_tokens)
        self.p = nn.Parameter(torch.tensor(float(cfg.p)), requires_grad=bool(cfg.trainable))
        if self.kind == "attn":
            self.attention = nn.Sequential(
                nn.Linear(channels, cfg.attention_hidden), nn.Tanh(), nn.Linear(cfg.attention_hidden, 1)
            )
        if self.kind not in {"gap", "gem", "signed_gem", "max", "avgmax", "attn", "cls"}:
            raise ValueError(f"Unknown pooling {self.kind}")
        self.out_dim: int = channels * (2 if self.kind == "avgmax" else 1)

    def forward(self, x):
        if x.ndim == 4:
            x = x.flatten(2).transpose(1, 2)
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected NCHW/NLC/NC, got {x.shape}")
        if self.kind == "cls":
            if self.prefix < 1:
                raise ValueError("CLS pooling requires a backbone with a class token")
            return x[:, 0]
        x = x[:, self.prefix :]
        if x.shape[1] == 0:
            raise ValueError("No spatial tokens after stripping prefix tokens")
        if self.kind == "gap":
            return x.mean(1)
        if self.kind == "max":
            return x.amax(1)
        if self.kind == "avgmax":
            return torch.cat([x.mean(1), x.amax(1)], 1)
        if self.kind == "attn":
            weights = self.attention(x).softmax(1)
            return (x * weights).sum(1)

        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            p = self.p.clamp(0.1, 10)
            if self.kind == "signed_gem":
                moment = (x.sign() * x.abs().clamp_min(1e-6).pow(p)).mean(1)
                return moment.sign() * moment.abs().clamp_min(1e-6).pow(1 / p)
            return x.clamp_min(1e-6).pow(p).mean(1).pow(1 / p)
