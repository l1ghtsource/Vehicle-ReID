import multiprocessing as mp
import os
import pickle
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from models.kernels import (
    ORIGINAL_EVA_FORWARD,
    attn_kernel_name,
    configure_runtime,
    eva_attention_forward,
    is_eva_attention,
    maybe_compile,
    patch_eva_attention,
    prepare_inference_model,
)
from third_party.eva_clip.eva_vit_model import Attention


class DummyRope(nn.Module):
    def forward(self, tokens):
        return tokens + 0.01


def _copy_attn(src: Attention, **kwargs) -> Attention:
    dst = Attention(**kwargs)
    dst.load_state_dict(src.state_dict())
    dst.eval()
    return dst


def test_attn_kernel_name_defaults_and_guards():
    assert attn_kernel_name(OmegaConf.create({"backend": "llm2clip"})) == "sdpa"
    assert attn_kernel_name(OmegaConf.create({"backend": "timm"})) == "math"
    assert attn_kernel_name(OmegaConf.create({"backend": "llm2clip", "attn_kernel": "math"})) == "math"
    assert attn_kernel_name(SimpleNamespace(backend="llm2clip")) == "sdpa"
    with pytest.raises(ValueError, match="attn_kernel"):
        attn_kernel_name(OmegaConf.create({"attn_kernel": "flash"}))


def test_patch_skips_non_eva_and_rejects_kernel():
    linear = nn.Linear(8, 8)
    assert not is_eva_attention(linear)
    assert patch_eva_attention(linear, "sdpa") == 0
    with pytest.raises(ValueError, match="attn_kernel"):
        patch_eva_attention(linear, "flash")


def test_sdpa_matches_math_subln_and_mask():
    src = Attention(dim=32, num_heads=4, qkv_bias=True, subln=True)
    src.eval()
    tokens = torch.randn(2, 8, 32)
    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[:, -1] = False
    math_mod = _copy_attn(src, dim=32, num_heads=4, qkv_bias=True, subln=True)
    sdpa_mod = _copy_attn(src, dim=32, num_heads=4, qkv_bias=True, subln=True)
    assert patch_eva_attention(math_mod, "math") == 1
    assert patch_eva_attention(sdpa_mod, "sdpa") == 1
    with torch.inference_mode():
        baseline = ORIGINAL_EVA_FORWARD(src, tokens, attn_mask=mask)
        math_out = math_mod(tokens, attn_mask=mask)
        sdpa_out = sdpa_mod(tokens, attn_mask=mask)
    assert torch.allclose(math_out, baseline, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sdpa_out, math_out, atol=1e-5, rtol=1e-5)


def test_sdpa_keeps_rope_and_qkv_path():
    rope = DummyRope()
    src = Attention(dim=32, num_heads=4, qkv_bias=True, subln=False, rope=rope)
    src.eval()
    tokens = torch.randn(2, 6, 32)
    math_mod = _copy_attn(src, dim=32, num_heads=4, qkv_bias=True, subln=False, rope=DummyRope())
    sdpa_mod = _copy_attn(src, dim=32, num_heads=4, qkv_bias=True, subln=False, rope=DummyRope())
    patch_eva_attention(math_mod, "math")
    patch_eva_attention(sdpa_mod, "sdpa")
    with torch.inference_mode():
        baseline = ORIGINAL_EVA_FORWARD(src, tokens)
        assert torch.allclose(math_mod(tokens), baseline, atol=1e-5, rtol=1e-5)
        assert torch.allclose(sdpa_mod(tokens), math_mod(tokens), atol=1e-5, rtol=1e-5)


def test_sdpa_falls_back_for_relative_bias():
    src = Attention(dim=16, num_heads=4, qkv_bias=True, subln=True, window_size=(2, 2))
    src.eval()
    tokens = torch.randn(1, 5, 16)
    patched = _copy_attn(src, dim=16, num_heads=4, qkv_bias=True, subln=True, window_size=(2, 2))
    patch_eva_attention(patched, "sdpa")
    bias = torch.zeros(1, 4, 5, 5)
    with torch.inference_mode():
        table_out = patched(tokens)
        extra = patched(tokens, rel_pos_bias=bias)
        baseline = ORIGINAL_EVA_FORWARD(src, tokens)
        extra_base = ORIGINAL_EVA_FORWARD(src, tokens, rel_pos_bias=bias)
    assert torch.allclose(table_out, baseline, atol=1e-5, rtol=1e-5)
    assert torch.allclose(extra, extra_base, atol=1e-5, rtol=1e-5)


def _spawn_restore_attention(payload, conn):
    attn = pickle.loads(payload)
    out = attn(torch.ones(1, 4, 16))
    conn.send((tuple(out.shape), getattr(attn, "_attn_kernel", None)))


def test_patched_attention_pickle_and_spawn():
    attn = Attention(dim=16, num_heads=4, qkv_bias=True, subln=True)
    attn.eval()
    patch_eva_attention(attn, "sdpa")
    payload = pickle.dumps(attn)
    restored = pickle.loads(payload)
    tokens = torch.randn(1, 4, 16)
    with torch.inference_mode():
        assert restored(tokens).shape == (1, 4, 16)
        assert torch.allclose(restored(tokens), attn(tokens), atol=1e-5, rtol=1e-5)
    assert restored._attn_kernel == "sdpa"
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_spawn_restore_attention, args=(payload, child))
    proc.start()
    child.close()
    proc.join(timeout=60)
    assert proc.exitcode == 0
    assert parent.poll(timeout=5)
    shape, kernel = parent.recv()
    assert shape == (1, 4, 16)
    assert kernel == "sdpa"


def test_sdpa_training_dropout_and_direct_forward():
    attn = Attention(dim=16, num_heads=4, qkv_bias=True, subln=True, attn_drop=0.5)
    patch_eva_attention(attn, "sdpa")
    attn.train()
    out = eva_attention_forward(attn, torch.randn(2, 4, 16))
    assert out.shape == (2, 4, 16)


def test_compile_and_runtime_helpers(monkeypatch):
    linear = nn.Linear(4, 4)
    assert maybe_compile(linear, False) is linear
    with pytest.raises(ValueError, match="compile_mode"):
        maybe_compile(linear, True, mode="fast")
    compiled = maybe_compile(linear, True, mode="default")
    assert compiled is not linear
    assert compiled(torch.ones(2, 4)).shape == (2, 4)

    class DictNet(nn.Module):
        def forward(self, x):
            return {"embedding": x + 1, "raw": x}

    wrapped = maybe_compile(DictNet(), True, mode="default", dynamic=False, embedding=True)
    assert wrapped(torch.ones(2, 3)).shape == (2, 3)
    plain = maybe_compile(DictNet(), True, mode="default", embedding=False)
    assert isinstance(plain(torch.ones(2, 3)), dict)

    fast = OmegaConf.create({"eval": {"cpu_threads": 1, "fast_kernels": True}})
    configure_runtime(fast)
    assert torch.get_num_threads() == 1
    assert bool(torch.backends.cuda.matmul.allow_tf32) is True
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    slow = OmegaConf.create(
        {"eval": {"cpu_threads": 0, "fast_kernels": False}, "trainer": {"deterministic": False}}
    )
    configure_runtime(slow)
    assert bool(torch.backends.cuda.matmul.allow_tf32) is False
    assert bool(torch.backends.cudnn.benchmark) is False
    model = prepare_inference_model(nn.Linear(3, 3), OmegaConf.create({"model": {"compile": False}}))
    assert isinstance(model, nn.Linear)
    assert torch.are_deterministic_algorithms_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    compiled_cfg = OmegaConf.create(
        {"eval": {"fast_kernels": True}, "model": {"compile": True, "compile_mode": "default"}}
    )
    out = prepare_inference_model(nn.Linear(3, 3), compiled_cfg)
    assert out(torch.ones(2, 3)).shape == (2, 3)
    defaulted = prepare_inference_model(
        nn.Linear(2, 2), OmegaConf.create({"model": {"backend": "llm2clip", "compile_mode": "default"}})
    )
    assert defaulted(torch.ones(2, 2)).shape == (2, 2)
    skipped = prepare_inference_model(nn.Linear(2, 2), OmegaConf.create({"model": {"backend": "timm"}}))
    assert isinstance(skipped, nn.Linear)

    class DictNet(nn.Module):
        def forward(self, x):
            return {"embedding": x, "raw": x}

    flags = OmegaConf.create(
        {
            "eval": {"fast_kernels": True},
            "model": {
                "compile": True,
                "compile_mode": "default",
                "compile_dynamic": False,
                "compile_embedding": True,
            },
        }
    )
    compiled_embed = prepare_inference_model(DictNet(), flags)
    assert compiled_embed(torch.ones(2, 3)).shape == (2, 3)
    full = prepare_inference_model(
        DictNet(),
        OmegaConf.create(
            {
                "eval": {"fast_kernels": True},
                "model": {"compile": True, "compile_mode": "default", "compile_embedding": False},
            }
        ),
    )
    assert isinstance(full(torch.ones(2, 3)), dict)
    torch.use_deterministic_algorithms(False)
