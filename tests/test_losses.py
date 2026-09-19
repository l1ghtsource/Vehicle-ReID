from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from modules.losses.core import DINO, AdaSP, LossCollection, Triplet, sample_ibot_masks


@pytest.mark.parametrize(
    "loss_name",
    [
        "combined",
        "arcface",
        "sphereface2",
        "triplet",
        "adasp",
        "arcface_adasp",
        "circle",
        "multisimilarity",
        "supcon",
        "proxyanchor",
        "ce_triplet",
        "contrastive",
        "ntxent",
        "proxynca",
        "softtriple",
        "fastap",
        "lifted",
        "cosface",
        "dino",
        "triplet_distanceweighted",
        "triplet_semihard",
    ],
)
def test_loss_backward(loss_name):
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
    ):
        c = compose(config_name="config", overrides=[f"loss={loss_name}"])
    criterion = LossCollection(c, 16, 8)
    x = torch.randn(12, 16, requires_grad=True)
    labels = torch.arange(4).repeat_interleave(3)
    loss, _ = criterion({"raw": x, "neck": x}, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("mining", ["hard", "semihard", "weighted", "all"])
def test_triplet_mining(mining):
    x = torch.randn(8, 5, requires_grad=True)
    loss = Triplet(mining=mining)(x, torch.arange(4).repeat_interleave(2))
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()


def test_triplet_all_honors_normalize_false():
    x = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],
            [0.0, 3.0],
        ]
    )
    y = torch.tensor([0, 0, 1, 1])
    normalized = Triplet(margin=0.3, mining="all", normalize=True)(x, y)
    raw = Triplet(margin=0.3, mining="all", normalize=False)(x, y)
    assert float(normalized) == pytest.approx(0.0, abs=1e-6)
    assert float(raw) > 0.0


def test_adasp_rejects_unbalanced_and_permutation_invariance():
    loss = AdaSP()
    x = torch.randn(8, 5)
    y = torch.arange(4).repeat_interleave(2)
    permutation = torch.randperm(8)
    torch.testing.assert_close(loss(x, y), loss(x[permutation], y[permutation]))
    with pytest.raises(ValueError):
        loss(x[:7], y[:7])


def test_dino_views_koleo_and_invalid_params():
    with pytest.raises(ValueError, match="Invalid DINO"):
        DINO(4, student_temp=0)
    with pytest.raises(ValueError, match="Invalid iBOT mask"):
        sample_ibot_masks(0, 4)
    with pytest.raises(ValueError, match="Invalid iBOT mask"):
        sample_ibot_masks(2, 4, min_ratio=0.5, max_ratio=0.1)
    none = sample_ibot_masks(3, 8, probability=0.0)
    assert not none.any()
    always = sample_ibot_masks(2, 6, min_ratio=0.4, max_ratio=0.6, probability=1.0)
    assert always.any()
    tiny = DINO(4, hidden_dim=8, bottleneck_dim=4, out_dim=8, nlayers=1, sinkhorn_iters=1)
    pair = torch.randn(4, 4, requires_grad=True)
    loss = tiny(pair, pair.detach())
    loss.backward()
    assert torch.isfinite(loss)
    assert pair.grad is not None
    patches = torch.randn(4, 5, 4, requires_grad=True)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[0, 0] = True
    masked = tiny(pair.detach(), pair.detach(), patches, patches.detach(), mask)
    masked.backward()
    assert patches.grad is not None
    single_patch = torch.randn(4, 1, 4, requires_grad=True)
    skipped = tiny(pair.detach(), pair.detach(), single_patch, single_patch.detach(), mask[:, :1])
    skipped.backward()
    assert torch.isfinite(skipped)
    empty_mask = torch.zeros(4, 5, dtype=torch.bool)
    gram_only = tiny(pair.detach(), pair.detach(), patches.detach(), patches.detach(), empty_mask)
    assert torch.isfinite(gram_only)
    wide = DINO(
        4, hidden_dim=8, bottleneck_dim=4, out_dim=8, nlayers=1, patch_dim=6, sinkhorn_iters=1
    )
    wide_patches = torch.randn(4, 3, 6, requires_grad=True)
    wide_mask = torch.ones(4, 3, dtype=torch.bool)
    wide_loss = wide(pair.detach(), pair.detach(), wide_patches, wide_patches.detach(), wide_mask)
    wide_loss.backward()
    assert wide_patches.grad is not None
    single = torch.randn(2, 4, requires_grad=True)
    collapsed = tiny(single, single.detach())
    collapsed.backward()
    assert torch.isfinite(collapsed)
    with pytest.raises(ValueError, match="even concatenated"):
        tiny(torch.randn(3, 4), torch.randn(3, 4))
