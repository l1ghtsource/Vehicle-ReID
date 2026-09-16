import torch
from torch import nn
from torch.nn import functional as F

from .backbones import Backbone
from .pooling import Pool


class ReIDModel(nn.Module):
    def __init__(self, cfg, initialize_pretrained=True):
        super().__init__()
        self.cfg = cfg.model
        self.backbone = Backbone(cfg.model, cfg.data.image_size, initialize_pretrained)
        pools = [Pool(d, cfg.model.pooling, self.backbone.prefix) for d in self.backbone.dims]
        self.pools = nn.ModuleList(pools)
        self.parts = int(cfg.model.head.local_parts)
        local_pools = [Pool(self.backbone.dims[-1], cfg.model.pooling, 0) for _ in range(self.parts)]
        self.local_pools = nn.ModuleList(local_pools)
        in_dim = sum(pool.out_dim for pool in pools) + sum(pool.out_dim for pool in local_pools)
        self.embedding_dim = int(cfg.model.head.embedding_dim or in_dim)
        self.projection = (
            nn.Linear(in_dim, self.embedding_dim, bias=False)
            if in_dim != self.embedding_dim
            else nn.Identity()
        )
        self.dropout = nn.Dropout(cfg.model.head.dropout)
        self.neck = nn.BatchNorm1d(self.embedding_dim) if cfg.model.head.bnneck else nn.Identity()
        if isinstance(self.neck, nn.BatchNorm1d):
            self.neck.bias.requires_grad_(False)
        self.frozen = False

    def freeze_backbone(self, frozen):
        self.frozen = frozen
        self.backbone.requires_grad_(not frozen)
        if frozen:
            self.backbone.eval()

    def forward(self, x):
        if self.frozen:
            self.backbone.eval()
            with torch.no_grad():
                levels = self.backbone(x)
        else:
            levels = self.backbone(x)
        pooled = [p(z) for p, z in zip(self.pools, levels, strict=True)]
        if self.parts:
            if levels[-1].ndim != 4 or levels[-1].shape[2] < self.parts:
                raise ValueError("local_parts requires spatial CNN features with sufficient height")
            pooled += [
                p(z)
                for p, z in zip(
                    self.local_pools,
                    torch.tensor_split(levels[-1], self.parts, dim=2),
                    strict=True,
                )
            ]
        raw = self.projection(self.dropout(torch.cat(pooled, dim=1)))
        neck = self.neck(raw)
        emb = neck if self.cfg.head.retrieval_feature == "neck" else raw
        return {"raw": raw, "neck": neck, "embedding": F.normalize(emb.float(), dim=1)}
