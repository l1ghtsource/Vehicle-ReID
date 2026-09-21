import pytest
import torch

from models import ReIDModel
from modules.regularization import EMA, awp, eval_mode


@pytest.mark.parametrize("pool", ["gem", "gap", "signed_gem", "max", "avgmax", "attn"])
def test_model_pool_and_backward(cfg, pool):
    cfg.model.pooling.kind = pool
    model = ReIDModel(cfg)
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    assert out["embedding"].shape == (4, 512)
    torch.testing.assert_close(out["embedding"].norm(dim=1), torch.ones(4))
    out["raw"].square().mean().backward()
    assert any(p.grad is not None for p in model.backbone.parameters())


def test_multilevel_local_parts(cfg):
    cfg.model.out_indices = [2, 3, 4]
    cfg.model.head.local_parts = 2
    model = ReIDModel(cfg)
    assert model(torch.randn(2, 3, 128, 128))["raw"].shape == (2, 512)


def test_eval_mode_restores_mixed_training_flags():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5), torch.nn.BatchNorm1d(2))
    model.train()
    model[1].eval()
    before = [child.training for child in model.modules()]
    with pytest.raises(RuntimeError, match="boom"), eval_mode(model):
        assert all(not child.training for child in model.modules())
        raise RuntimeError("boom")
    assert [child.training for child in model.modules()] == before


def test_awp_restore_on_failure_and_ema(cfg):
    model = torch.nn.Linear(4, 2)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    model(torch.randn(2, 4)).sum().backward()
    with pytest.raises(RuntimeError), awp(model, cfg.train.awp):
        assert not torch.equal(model.weight, original["weight"])
        raise RuntimeError("test")
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, original[k])
    ema = EMA(model, 0.9)
    with torch.no_grad():
        model.weight.add_(1)
    ema.update(model)
    current = model.weight.clone()
    with ema.apply(model):
        assert not torch.equal(model.weight, current)
    torch.testing.assert_close(model.weight, current)
