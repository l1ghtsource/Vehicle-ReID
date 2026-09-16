from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from modules.losses.core import AdaSP, LossCollection, Triplet


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


def test_adasp_rejects_unbalanced_and_permutation_invariance():
    loss = AdaSP()
    x = torch.randn(8, 5)
    y = torch.arange(4).repeat_interleave(2)
    permutation = torch.randperm(8)
    torch.testing.assert_close(loss(x, y), loss(x[permutation], y[permutation]))
    with pytest.raises(ValueError):
        loss(x[:7], y[:7])
