from types import MethodType
from typing import Any, cast

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from third_party.eva_clip.eva_vit_model import Attention as EvaAttention

ATTN_KERNELS = frozenset({"math", "sdpa"})
COMPILE_MODES = frozenset({"default", "reduce-overhead", "max-autotune"})


def attn_kernel_name(cfg) -> str:
    value = cfg.get("attn_kernel") if hasattr(cfg, "get") else None
    if value is None:
        backend = str(getattr(cfg, "backend", ""))
        return "sdpa" if backend == "llm2clip" else "math"
    name = str(value)
    if name not in ATTN_KERNELS:
        raise ValueError("attn_kernel must be math/sdpa")
    return name


def is_eva_attention(module: nn.Module) -> bool:
    return (
        module.__class__.__name__ == "Attention"
        and hasattr(module, "xattn")
        and hasattr(module, "inner_attn_ln")
        and hasattr(module, "scale")
        and (hasattr(module, "q_proj") or hasattr(module, "qkv"))
    )


def _eva_qkv(attn: EvaAttention, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, tokens, _channels = x.shape
    if attn.subln:
        query = attn.q_proj(x)
        key = attn.k_proj(x)
        value = attn.v_proj(x)
        query = query.reshape(batch, tokens, attn.num_heads, -1).permute(0, 2, 1, 3)
        key = key.reshape(batch, tokens, attn.num_heads, -1).permute(0, 2, 1, 3)
        value = value.reshape(batch, tokens, attn.num_heads, -1).permute(0, 2, 1, 3)
        return query, key, value
    bias = None
    if attn.q_bias is not None:
        q_bias = cast(torch.Tensor, attn.q_bias)
        v_bias = cast(torch.Tensor, attn.v_bias)
        bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
    packed = F.linear(input=x, weight=attn.qkv.weight, bias=bias)
    packed = packed.reshape(batch, tokens, 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
    return packed[0], packed[1], packed[2]


def _eva_rope(
    attn: EvaAttention, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not attn.rope:
        return query, key
    rotated_q = attn.rope(query[:, :, 1:, :])
    query = torch.cat((query[:, :, :1, :], rotated_q), -2).type_as(value)
    rotated_k = attn.rope(key[:, :, 1:, :])
    key = torch.cat((key[:, :, :1, :], rotated_k), -2).type_as(value)
    return query, key


def _sdpa_key_mask(attn_mask: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    keep = attn_mask.bool()
    mask = torch.zeros(keep.shape[0], 1, 1, keep.shape[-1], dtype=dtype, device=device)
    return mask.masked_fill(~keep[:, None, None, :], torch.finfo(dtype).min)


def _math_attention(
    attn: EvaAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rel_pos_bias,
    attn_mask,
) -> torch.Tensor:
    scores = (query * attn.scale) @ key.transpose(-2, -1)
    if attn.relative_position_bias_table is not None:
        table = cast(torch.Tensor, attn.relative_position_bias_table)
        index = cast(torch.Tensor, attn.relative_position_index)
        window = cast(tuple[int, int], attn.window_size)
        tokens = window[0] * window[1] + 1
        relative = table[index.view(-1)].view(tokens, tokens, -1)
        scores = scores + relative.permute(2, 0, 1).contiguous().unsqueeze(0).type_as(scores)
    if rel_pos_bias is not None:
        scores = scores + rel_pos_bias.type_as(scores)
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask.bool()[:, None, None, :], float("-inf"))
    weights = attn.attn_drop(scores.softmax(dim=-1))
    return (weights @ value).transpose(1, 2).reshape(query.shape[0], query.shape[2], -1)


def eva_attention_forward(self: EvaAttention, x, rel_pos_bias=None, attn_mask=None):
    query, key, value = _eva_qkv(self, x)
    query, key = _eva_rope(self, query, key, value)
    use_sdpa = getattr(self, "_attn_kernel", "math") == "sdpa"
    can_sdpa = use_sdpa and self.relative_position_bias_table is None and rel_pos_bias is None
    if can_sdpa:
        mask = None if attn_mask is None else _sdpa_key_mask(attn_mask, query.dtype, query.device)
        drop = float(self.attn_drop.p) if self.training else 0.0
        context = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=drop, scale=self.scale
        )
        hidden = context.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
    else:
        hidden = _math_attention(self, query, key, value, rel_pos_bias, attn_mask)
    hidden = self.inner_attn_ln(hidden)
    return self.proj_drop(self.proj(hidden))


def patch_eva_attention(module: nn.Module, kernel: str) -> int:
    name = str(kernel)
    if name not in ATTN_KERNELS:
        raise ValueError("attn_kernel must be math/sdpa")
    count = 0
    for child in module.modules():
        if not is_eva_attention(child):
            continue
        patched = cast(Any, child)
        patched._attn_kernel = name
        patched.forward = MethodType(eva_attention_forward, patched)
        count += 1
    return count


def maybe_compile(model, enabled: bool, mode: str = "reduce-overhead"):
    if not enabled:
        return model
    if mode not in COMPILE_MODES:
        raise ValueError("compile_mode must be default/reduce-overhead/max-autotune")
    return torch.compile(model, mode=mode, dynamic=True)


def configure_runtime(cfg) -> None:
    threads = int(OmegaConf.select(cfg, "eval.cpu_threads", default=0) or 0)
    if threads > 0:
        torch.set_num_threads(threads)
    fast = bool(OmegaConf.select(cfg, "eval.fast_kernels", default=True))
    if fast:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        return
    torch.use_deterministic_algorithms(bool(OmegaConf.select(cfg, "trainer.deterministic", default=True)))
    torch.backends.cudnn.benchmark = False


def prepare_inference_model(model, cfg) -> Any:
    configure_runtime(cfg)
    enabled = OmegaConf.select(cfg, "model.compile", default=None)
    if enabled is None:
        enabled = str(OmegaConf.select(cfg, "model.backend", default="")) == "llm2clip"
    mode = str(OmegaConf.select(cfg, "model.compile_mode", default="reduce-overhead"))
    return maybe_compile(model, bool(enabled), mode)
