from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from interp import METHODS, interpret, normalize_maps, overlay
from interp.attention import _cls_maps, attention_rollout, chefer_attribution, last_attention
from interp.cam import eigen_cam, grad_cam
from interp.common import AttentionCapture, Unfrozen, as_nchw, embedding_score, prefix_tokens, upsample
from interp.pooling import pooling_attention
from models.pooling import Pool
from models.reid import ReIDModel


class TinyAttn(nn.Module):
    def __init__(self, dim=8, heads=2, dim_index=-1):
        super().__init__()
        self.num_heads = heads
        self.dim_index = dim_index
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, tokens, channels = x.shape
        heads = self.num_heads
        q = self.query(x).reshape(batch, tokens, heads, channels // heads).transpose(1, 2)
        k = self.key(x).reshape(batch, tokens, heads, channels // heads).transpose(1, 2)
        v = self.value(x).reshape(batch, tokens, heads, channels // heads).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (channels // heads) ** 0.5
        attn = scores.softmax(dim=self.dim_index, dtype=torch.float32)
        return self.proj((attn @ v).transpose(1, 2).reshape(batch, tokens, channels))


class TinyBlock(nn.Module):
    def __init__(self, dim_index=-1):
        super().__init__()
        self.attn = TinyAttn(dim_index=dim_index)
        self.ff = nn.Linear(8, 8)

    def forward(self, x):
        return self.ff(x + self.attn(x))


class TinyBackbone(nn.Module):
    def __init__(self, dim_index=-1):
        super().__init__()
        self.prefix = 1
        self.stem = nn.Conv2d(3, 8, kernel_size=4, stride=4)
        self.cls = nn.Parameter(torch.zeros(1, 1, 8))
        self.blocks = nn.ModuleList([TinyBlock(dim_index), TinyBlock(dim_index)])

    def forward(self, x):
        tokens = self.stem(x).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], 1)
        for block in self.blocks:
            tokens = block(tokens)
        return [tokens]


class TinyReID(nn.Module):
    def __init__(self, pool="attn", dim_index=-1):
        super().__init__()
        self.frozen = False
        self.backbone = TinyBackbone(dim_index=dim_index)
        self.pools = nn.ModuleList(
            [Pool(8, SimpleNamespace(kind=pool, p=3.0, trainable=False, attention_hidden=4), 1)]
        )
        self.proj = nn.Linear(8, 8)

    def freeze_backbone(self, frozen):
        self.frozen = frozen
        self.backbone.requires_grad_(not frozen)

    def forward(self, x):
        if self.frozen:
            self.backbone.eval()
            with torch.no_grad():
                levels = self.backbone(x)
        else:
            levels = self.backbone(x)
        pooled = self.pools[0](levels[0])
        raw = self.proj(pooled)
        return {"raw": raw, "neck": raw, "embedding": F.normalize(raw.float(), dim=1)}


class EmptyBackbone(nn.Module):
    def forward(self, x):
        return x


class CnnBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix = 0
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)

    def forward(self, x):
        return [self.conv(x)]


class CnnReID(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = False
        self.backbone = CnnBackbone()
        self.pools = nn.ModuleList(
            [Pool(8, SimpleNamespace(kind="gap", p=3.0, trainable=False, attention_hidden=4))]
        )
        self.proj = nn.Linear(8, 8)

    def freeze_backbone(self, frozen):
        self.frozen = frozen

    def forward(self, x):
        pooled = self.pools[0](self.backbone(x)[0])
        raw = self.proj(pooled)
        return {"embedding": F.normalize(raw.float(), dim=1)}


class UnusedFeatReID(CnnReID):
    def forward(self, x):
        self.backbone(x)
        return {"embedding": torch.ones(len(x), 8, requires_grad=True)}


class HooklessReID(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = False
        self.backbone = EmptyBackbone()
        self.pools = nn.ModuleList(
            [Pool(8, SimpleNamespace(kind="gap", p=3.0, trainable=False, attention_hidden=4))]
        )

    def freeze_backbone(self, frozen):
        self.frozen = frozen

    def forward(self, x):
        return {"embedding": torch.ones(len(x), 8, requires_grad=True)}


def images(batch=2, size=8):
    torch.manual_seed(0)
    return torch.randn(batch, 3, size, size)


def test_common_helpers_and_guards():
    square = as_nchw(torch.arange(8, dtype=torch.float32).reshape(1, 4, 2), prefix=0)
    assert square.shape == (1, 2, 2, 2)
    nchw = as_nchw(torch.ones(2, 3, 4, 5))
    assert nchw.shape == (2, 3, 4, 5)
    wide = as_nchw(torch.ones(1, 6, 3))
    assert wide.shape == (1, 3, 1, 6)
    assert as_nchw(torch.ones(2, 5)).shape == (2, 5, 1, 1)
    with pytest.raises(ValueError, match="Expected NCHW"):
        as_nchw(torch.ones(2, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="No spatial"):
        as_nchw(torch.ones(1, 1, 4), prefix=1)

    assert upsample(torch.ones(2, 2), (4, 4)).shape == (1, 4, 4)
    assert upsample(torch.ones(1, 2, 2), (3, 5)).shape == (1, 3, 5)
    assert upsample(torch.ones(1, 1, 2, 2), (4, 4)).shape == (1, 4, 4)
    with pytest.raises(ValueError, match="NHW/NCHW"):
        upsample(torch.ones(1), (2, 2))
    with pytest.raises(ValueError, match="positive"):
        upsample(torch.ones(1, 2, 2), (0, 2))

    assert normalize_maps(np.array([[0.0, 1.0], [0.5, 0.5]])).shape == (1, 2, 2)
    assert normalize_maps(np.zeros((1, 2, 2))).max() == 0
    with pytest.raises(ValueError, match="NHW"):
        normalize_maps(np.ones((2, 2, 2, 2)))

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    heat = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    blended = overlay(rgb, heat, alpha=0.5)
    assert blended.dtype == np.uint8 and blended.shape == (4, 4, 3)
    chw = overlay(np.zeros((3, 4, 4), dtype=np.float32), heat, alpha=0)
    assert chw.shape == (4, 4, 3)
    with pytest.raises(ValueError, match="alpha"):
        overlay(rgb, heat, alpha=2)
    with pytest.raises(ValueError, match="RGB"):
        overlay(np.zeros((4, 4)), heat)
    with pytest.raises(ValueError, match="spatial size"):
        overlay(rgb, np.ones((2, 2)))

    emb = {"embedding": F.normalize(torch.ones(2, 4), dim=1)}
    assert embedding_score(emb).ndim == 0
    neck = {"neck": torch.tensor([[2.0, 0.0], [0.0, 3.0]]), "embedding": F.normalize(torch.ones(2, 2), dim=1)}
    assert float(embedding_score(neck)) == 13.0
    assert embedding_score(emb["embedding"], torch.ones(4)).ndim == 0
    assert embedding_score(emb, torch.ones(2, 4)).ndim == 0
    with pytest.raises(ValueError, match="reference"):
        embedding_score(emb, torch.ones(3, 4))
    assert prefix_tokens(nn.Linear(2, 2)) == 0

    x = torch.randn(1, 2, 3, 3, requires_grad=True)
    with AttentionCapture() as capture:
        x.softmax(dim=3, dtype=torch.float32).sum().backward()
    assert capture.maps and capture.maps[0].shape[-1] == 3
    original = torch.Tensor.softmax
    try:
        with AttentionCapture():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert torch.Tensor.softmax is original


def test_all_methods_on_tiny_vit():
    model = TinyReID()
    x = images()
    for name in METHODS:
        maps = interpret(model, x, method=name)
        assert maps.shape == (2, 8, 8)
        assert np.isfinite(maps).all()
    model.freeze_backbone(True)
    maps = interpret(model, x, method="gradcam")
    assert model.frozen is True
    assert maps.shape == (2, 8, 8)
    ref_a = torch.zeros(8)
    ref_a[0] = 1
    ref_b = torch.zeros(8)
    ref_b[-1] = 1
    a = interpret(model, x[:1], method="gradcam", reference=ref_a)
    b = interpret(model, x[:1], method="gradcam", reference=ref_b)
    assert a.shape == b.shape
    with pytest.raises(ValueError, match="Unknown"):
        interpret(model, x, method="smoothgrad")
    with pytest.raises(ValueError, match="NCHW"):
        grad_cam(model, torch.randn(2, 8, 8))
    with pytest.raises(ValueError, match="NCHW"):
        last_attention(model, torch.randn(3, 8, 8))
    with pytest.raises(ValueError, match="NCHW"):
        pooling_attention(model, torch.randn(2, 1, 8, 8))


def test_pooling_variants_and_missing_modules():
    x = images()
    for kind in ("attn", "max", "gap"):
        maps = pooling_attention(TinyReID(pool=kind), x)
        assert maps.shape == (2, 8, 8)
    empty = TinyReID()
    empty.pools = nn.ModuleList()
    with pytest.raises(ValueError, match="pooling"):
        pooling_attention(empty, x)
    with pytest.raises(RuntimeError, match="hooked"):
        pooling_attention(HooklessReID(), x)
    with pytest.raises(RuntimeError, match="hooked"):
        eigen_cam(HooklessReID(), x)


def test_attention_methods_require_vit_softmax():
    x = images()
    with pytest.raises(ValueError, match="token-attention"):
        attention_rollout(CnnReID(), x)
    with pytest.raises(ValueError, match="token-attention"):
        chefer_attribution(CnnReID(), x)
    with pytest.raises(RuntimeError, match="gradient"):
        grad_cam(UnusedFeatReID(), x)
    with pytest.raises(ValueError, match="BNN"):
        _cls_maps(torch.ones(2, 4), 0, (4, 4))
    with pytest.raises(ValueError, match="No spatial"):
        _cls_maps(torch.ones(1, 3, 3), 3, (4, 4))
    maps = last_attention(TinyReID(dim_index=3), x)
    assert maps.shape == (2, 8, 8)


def test_cnn_cam_on_smoke_reid(cfg):
    cfg.model.pretrained = False
    cfg.model.pooling.kind = "gem"
    model = ReIDModel(cfg, initialize_pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    for name in ("gradcam", "hirescam", "layercam", "eigencam", "pooling"):
        maps = interpret(model, x, method=name)
        assert maps.shape == (1, 64, 64)
        assert np.isfinite(maps).all()
    with pytest.raises(ValueError, match="token-attention"):
        interpret(model, x, method="rollout")
    with Unfrozen(model):
        assert model.training is False
    model.freeze_backbone(True)
    with Unfrozen(model):
        assert model.frozen is False
    assert model.frozen is True
