import json
from functools import partial
from pathlib import Path
from typing import Any, cast

import timm
import torch
from huggingface_hub import hf_hub_download
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel

from third_party.eva_clip.eva_vit_model import EVAVisionTransformer
from third_party.radio.hf_model import RADIOConfig, RADIOModel

from .input_size import spatial_multiple
from .kernels import attn_kernel_name, patch_eva_attention

TOKEN_MASK_BACKENDS = frozenset({"llm2clip"})


def llm2clip_level_indices(out_indices, depth: int) -> list[int]:
    if out_indices is None or len(list(out_indices)) == 0:
        raise ValueError("llm2clip multi-level features require model.out_indices")
    resolved: list[int] = []
    for raw in out_indices:
        index = int(raw)
        if index < 0:
            index += depth
        if index < 0 or index >= depth:
            raise ValueError(f"llm2clip out_indices entry {raw} is outside 0..{depth - 1}")
        resolved.append(index)
    if len(set(resolved)) != len(resolved):
        raise ValueError("llm2clip out_indices must be unique")
    return resolved


def llm2clip_fuse_name(cfg) -> str:
    raw = cfg.get("multilevel_fuse") if hasattr(cfg, "get") else getattr(cfg, "multilevel_fuse", None)
    name = "concat" if raw is None else str(raw).strip().lower()
    if name not in {"concat", "mean"}:
        raise ValueError("llm2clip multilevel_fuse must be concat/mean")
    return name


def llm2clip_levels(net, x, mask, indices: list[int], fuse: str):
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for index in set(indices):

        def keep(_module, _inputs, output, index=index):
            captured[index] = output

        hooks.append(net.blocks[index].register_forward_hook(keep))
    try:
        net.forward_features(x, return_all_features=True, bool_masked_pos=mask)
    finally:
        for hook in hooks:
            hook.remove()
    depth = len(net.blocks)
    levels = []
    for index in indices:
        tokens = captured[index]
        if fuse == "mean" or index == depth - 1:
            tokens = net.norm(tokens)
        levels.append(tokens)
    if fuse == "mean":
        return [torch.stack(levels).mean(0)]
    return levels


def container_dict(value: Any) -> dict[str, Any]:
    container = OmegaConf.to_container(value, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Expected a mapping configuration")
    return {str(key): item for key, item in container.items()}


def load_state(path):
    if str(path).endswith(".safetensors"):
        return load_file(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
    return {k.removeprefix("module."): v for k, v in state.items()}


class Backbone(nn.Module):
    def __init__(self, cfg, image_size, initialize_pretrained=True):
        super().__init__()
        self.net: Any
        self.level_indices: list[int] | None = None
        self.level_fuse = "concat"
        self.cfg, self.backend = cfg, cfg.backend
        self.prefix = cfg.num_prefix_tokens
        self.layout = cfg.layout
        pretrained = bool(cfg.pretrained and initialize_pretrained)
        sizes = tuple(int(value) for value in image_size)
        kw = container_dict(cfg.kwargs)
        if cfg.backend == "timm":
            multiple = spatial_multiple(cfg)
            if sizes[0] % multiple or sizes[1] % multiple:
                raise ValueError(
                    f"data.image_size {list(sizes)} must be divisible by {multiple} for {cfg.name}"
                )
            args: dict[str, Any] = dict(pretrained=pretrained and not cfg.checkpoint_path, **kw)
            if bool(cfg.get("bind_image_size", False)):
                args.setdefault("img_size", sizes)
            if cfg.drop_path_rate:
                args["drop_path_rate"] = cfg.drop_path_rate
            if cfg.features_only:
                args.update(features_only=True, out_indices=tuple(cfg.out_indices))
            else:
                args.update(num_classes=0, global_pool="")
            self.net = cast(Any, timm.create_model(cfg.name, **args))
            self.dims = (
                list(self.net.feature_info.channels()) if cfg.features_only else [self.net.num_features]
            )
            if not cfg.features_only:
                self.prefix = getattr(self.net, "num_prefix_tokens", cfg.num_prefix_tokens)
            if cfg.gradient_checkpointing:
                self.net.set_grad_checkpointing(True)
        elif cfg.backend == "radio":
            config_path = Path(__file__).resolve().parents[1] / "third_party/radio/config.json"
            values = json.loads(config_path.read_text())
            values["args"]["drop_path"] = float(cfg.drop_path_rate)

            if not pretrained:
                values["args"]["model_kwargs"]["weight_init"] = ""
            config = RADIOConfig(**values)
            self.net = RADIOModel(config)
            if pretrained:
                path = cfg.checkpoint_path
                if not path:
                    path = hf_hub_download(
                        cfg.name,
                        "model.safetensors",
                        revision=cfg.revision,
                        local_files_only=cfg.local_files_only,
                    )
                state = load_file(path) if str(path).endswith(".safetensors") else load_state(path)
                self.net.load_state_dict(state, strict=True)
            self.dims = [self.net.radio_model.embed_dim]
            if cfg.gradient_checkpointing:
                self.net.radio_model.model.set_grad_checkpointing(True)
        elif cfg.backend == "hf":
            common = dict(
                revision=cfg.revision,
                local_files_only=cfg.local_files_only,
                trust_remote_code=cfg.backend == "radio",
            )
            if (not pretrained or cfg.checkpoint_path) and cfg.get("architecture"):
                architecture = container_dict(cfg.architecture)
                kind = architecture.pop("model_type")
                config = AutoConfig.for_model(kind, **architecture)
            else:
                config = AutoConfig.from_pretrained(cfg.name, **common)
            if cfg.drop_path_rate:
                if not hasattr(config, "drop_path_rate"):
                    raise ValueError(
                        "This HF backbone has no drop_path_rate config; use 0 or a custom adapter"
                    )
                config.drop_path_rate = cfg.drop_path_rate
            if pretrained and not cfg.checkpoint_path:
                self.net = AutoModel.from_pretrained(cfg.name, config=config, **common, **kw)
            else:
                self.net = AutoModel.from_config(config, trust_remote_code=cfg.backend == "radio", **kw)
            if not cfg.feature_dims:
                raise ValueError("HF adapters require explicit model.feature_dims matching hidden_indices")
            self.dims = list(cfg.feature_dims)
            if cfg.gradient_checkpointing:
                self.net.gradient_checkpointing_enable()
        elif cfg.backend == "llm2clip":
            if list(image_size) != [336, 336]:
                raise ValueError(
                    "LLM2CLIP preset requires 336x336; position/RoPE interpolation is not implicit"
                )
            self.net = EVAVisionTransformer(
                img_size=336,
                patch_size=14,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=2.6667,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                num_classes=1280,
                use_mean_pooling=False,
                rope=True,
                pt_hw_seq_len=16,
                intp_freq=True,
                naiveswiglu=True,
                subln=True,
                xattn=False,
                drop_path_rate=cfg.drop_path_rate,
                grad_checkpointing=cfg.gradient_checkpointing,
                **kw,
            )
            if bool(cfg.features_only):
                self.level_indices = llm2clip_level_indices(cfg.out_indices, len(self.net.blocks))
                self.level_fuse = llm2clip_fuse_name(cfg)
                width = int(self.net.embed_dim)
                self.dims = [width] if self.level_fuse == "mean" else [width] * len(self.level_indices)
            else:
                self.dims = [1024]
            if pretrained:
                path = cfg.checkpoint_path
                if not path:
                    path = hf_hub_download(
                        cfg.name,
                        "LLM2CLIP-EVA02-L-14-336.pt",
                        revision=cfg.revision,
                        local_files_only=cfg.local_files_only,
                    )
                state = load_state(path)
                visual = {k[len("visual.") :]: v for k, v in state.items() if k.startswith("visual.")}
                self.net.load_state_dict(visual or state, strict=True)
            if not hasattr(self.net, "mask_token"):
                width = int(getattr(self.net, "embed_dim", self.dims[0]))
                self.net.mask_token = nn.Parameter(torch.zeros(1, 1, width))
                nn.init.trunc_normal_(self.net.mask_token, std=0.02)
        elif cfg.backend == "custom":
            self.net = instantiate(cfg.kwargs)
            self.dims = list(cfg.feature_dims)
        else:
            raise ValueError(f"Unknown backbone backend {cfg.backend}")
        if cfg.checkpoint_path and cfg.backend not in {"llm2clip", "radio"} and initialize_pretrained:
            self.net.load_state_dict(load_state(cfg.checkpoint_path), strict=True)
        patch_eva_attention(self.net, attn_kernel_name(cfg))

    @property
    def supports_token_mask(self) -> bool:
        return self.backend in TOKEN_MASK_BACKENDS

    def forward(self, x, mask=None):
        if mask is not None and not self.supports_token_mask:
            raise ValueError("iBOT token masks are only applied by llm2clip; set ibot_weight=0")
        c = self.cfg
        if self.backend == "timm":
            out = self.net(x) if c.features_only else self.net.forward_features(x)
        elif self.backend == "hf":
            out = self.net(pixel_values=x, output_hidden_states=True)
            if c.hidden_indices is not None:
                out = [out.hidden_states[i] for i in c.hidden_indices]
            else:
                out = out.last_hidden_state
        elif self.backend == "radio":
            _, out = self.net(x)
        elif self.backend == "llm2clip":
            if self.level_indices is None:
                tokens = self.net.forward_features(x, return_all_features=True, bool_masked_pos=mask)
                out = self.net.norm(tokens)
            else:
                out = llm2clip_levels(self.net, x, mask, self.level_indices, self.level_fuse)
        else:
            out = self.net(x)
        out = list(out) if isinstance(out, (list, tuple)) else [out]
        if self.layout == "NHWC":
            out = [z.permute(0, 3, 1, 2) if z.ndim == 4 else z for z in out]
        if len(out) != len(self.dims):
            raise ValueError(f"Backbone returned {len(out)} levels, expected {len(self.dims)}")
        for z, d in zip(out, self.dims, strict=True):
            actual = z.shape[1] if z.ndim == 4 else z.shape[-1]
            if actual != d:
                raise ValueError(f"Feature dimension {actual} != configured {d}; shape={z.shape}")
        return out
