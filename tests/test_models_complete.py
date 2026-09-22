from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from torch import nn

import models.backbones as backbones
import models.reid as reid_module
from models.backbones import Backbone, container_dict, load_state
from models.input_size import scaled_hw, spatial_multiple, validate_image_geometry
from models.pooling import Pool
from models.reid import ReIDModel
from third_party.eva_clip.eva_vit_model import EVAVisionTransformer


class DummyNet(nn.Module):
    def __init__(self, dims=(4,), output=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.num_features = dims[-1]
        self.num_prefix_tokens = 2
        self.feature_info = SimpleNamespace(channels=lambda: list(dims))
        self.radio_model = SimpleNamespace(
            embed_dim=dims[-1],
            model=SimpleNamespace(set_grad_checkpointing=self._set_checkpointing),
        )
        self.config = SimpleNamespace(to_dict=lambda: {"model_type": "dummy"})
        self.output = output
        self.checkpointing = False
        self.loaded = None

    def _set_checkpointing(self, enabled):
        self.checkpointing = enabled

    def set_grad_checkpointing(self, enabled):
        self.checkpointing = enabled

    def gradient_checkpointing_enable(self):
        self.checkpointing = True

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.loaded = (state_dict, strict)
        return SimpleNamespace()

    def forward_features(self, x, return_all_features=False, bool_masked_pos=None):
        if return_all_features:
            out = torch.ones(len(x), self.num_features, 2, 2)
            if bool_masked_pos is not None and bool_masked_pos.any():
                return out + 1
            return out
        return self.output if self.output is not None else torch.ones(len(x), self.num_features, 2, 2)

    def forward(self, x=None, pixel_values=None, output_hidden_states=False):
        value = pixel_values if pixel_values is not None else x
        if value is None:
            raise ValueError("DummyNet requires an input")
        if output_hidden_states:
            hidden = (
                torch.ones(len(value), 3, self.num_features),
                torch.ones(len(value), 3, self.num_features),
            )
            return SimpleNamespace(hidden_states=hidden, last_hidden_state=hidden[-1])
        if self.output is not None:
            return self.output
        return None, torch.ones(len(value), self.num_features, 2, 2)

    def norm(self, value):
        return value


class ConfigBackbone(nn.Module):
    def __init__(self, cfg, image_size, initialize_pretrained=True):
        super().__init__()
        self.dims = [4]
        self.prefix = 1

    def forward(self, x, mask=None):
        return [torch.ones(len(x), 4, 2, 2)]


def test_container_and_state_loading(tmp_path, monkeypatch):
    assert container_dict(OmegaConf.create({"a": 1})) == {"a": 1}
    with pytest.raises(TypeError):
        container_dict(OmegaConf.create([1]))

    tensor = torch.ones(1)
    monkeypatch.setattr(backbones, "load_file", lambda path: {"safe": tensor})
    assert load_state(tmp_path / "x.safetensors") == {"safe": tensor}
    path = tmp_path / "state.pt"
    torch.save({"state_dict": {"module.weight": tensor}}, path)
    assert list(load_state(path)) == ["weight"]
    torch.save({"model": {"module.bias": tensor}}, path)
    assert list(load_state(path)) == ["bias"]


def test_timm_backbone_paths(cfg, tmp_path, monkeypatch):
    calls = []

    def create_model(name, **kwargs):
        calls.append((name, kwargs))
        if kwargs.get("features_only"):
            output = [torch.ones(2, 3, 2, 2), torch.ones(2, 4, 2, 2)]
            return DummyNet((3, 4), output)
        return DummyNet((4,), torch.ones(2, 4, 2, 2))

    monkeypatch.setattr(backbones.timm, "create_model", create_model)
    cfg.model.drop_path_rate = 0.2
    cfg.model.gradient_checkpointing = True
    cfg.model.out_indices = [0, 1]
    model = Backbone(cfg.model, cfg.data.image_size)
    assert model.dims == [3, 4]
    assert model.net.checkpointing
    assert calls[0][1]["features_only"] is True
    assert "img_size" not in calls[0][1]
    cfg.model.bind_image_size = True
    Backbone(cfg.model, cfg.data.image_size)
    assert calls[-1][1]["img_size"] == tuple(int(value) for value in cfg.data.image_size)
    out = model(torch.zeros(2, 3, 8, 8))
    assert [item.shape[1] for item in out] == [3, 4]
    assert not model.supports_token_mask
    with pytest.raises(ValueError, match="token masks"):
        model(torch.zeros(2, 3, 8, 8), mask=torch.zeros(2, 4, dtype=torch.bool))

    cfg.model.features_only = False
    cfg.model.feature_dims = [4]
    model = Backbone(cfg.model, cfg.data.image_size)
    assert model.prefix == 2
    assert model(torch.zeros(2, 3, 8, 8))[0].shape[1] == 4

    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.ones(1)}, checkpoint)
    cfg.model.checkpoint_path = str(checkpoint)
    model = Backbone(cfg.model, cfg.data.image_size)
    assert model.net.loaded is not None


def test_radio_backbone_paths(cfg, tmp_path, monkeypatch):
    nets = []

    def make_radio(config):
        net = DummyNet((6,))
        nets.append(net)
        return net

    monkeypatch.setattr(backbones, "RADIOConfig", lambda **values: values)
    monkeypatch.setattr(backbones, "RADIOModel", make_radio)
    monkeypatch.setattr(backbones, "hf_hub_download", lambda *args, **kwargs: "radio.safetensors")
    monkeypatch.setattr(backbones, "load_file", lambda path: {"weight": torch.ones(1)})
    cfg.model.backend = "radio"
    cfg.model.pretrained = False
    cfg.model.gradient_checkpointing = True
    cfg.model.feature_dims = [6]
    model = Backbone(cfg.model, cfg.data.image_size)
    assert model.dims == [6]
    assert model(torch.zeros(2, 3, 8, 8))[0].shape[1] == 6
    assert not model.supports_token_mask
    with pytest.raises(ValueError, match="token masks"):
        model(torch.zeros(2, 3, 8, 8), mask=torch.zeros(2, 4, dtype=torch.bool))
    assert nets[-1].checkpointing

    cfg.model.pretrained = True
    cfg.model.checkpoint_path = None
    model = Backbone(cfg.model, cfg.data.image_size)
    assert model.net.loaded is not None
    path = tmp_path / "radio.pt"
    torch.save({"weight": torch.ones(1)}, path)
    cfg.model.checkpoint_path = str(path)
    Backbone(cfg.model, cfg.data.image_size)


def test_hf_backbone_paths(cfg, monkeypatch):
    configs = []

    def for_model(kind, **kwargs):
        config = SimpleNamespace(model_type=kind, drop_path_rate=0.0)
        configs.append(config)
        return config

    monkeypatch.setattr(backbones.AutoConfig, "for_model", for_model)
    monkeypatch.setattr(
        backbones.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(drop_path_rate=0.0),
    )
    monkeypatch.setattr(
        backbones.AutoModel,
        "from_config",
        lambda *args, **kwargs: DummyNet((5,)),
    )
    monkeypatch.setattr(
        backbones.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: DummyNet((5,)),
    )
    cfg.model.backend = "hf"
    cfg.model.architecture = {"model_type": "dummy"}
    cfg.model.feature_dims = [5]
    cfg.model.hidden_indices = [0]
    cfg.model.gradient_checkpointing = True
    cfg.model.drop_path_rate = 0.3
    model = Backbone(cfg.model, cfg.data.image_size)
    assert configs[0].drop_path_rate == 0.3
    assert model.net.checkpointing
    assert model(torch.zeros(2, 3, 8, 8))[0].shape[-1] == 5
    assert not model.supports_token_mask
    with pytest.raises(ValueError, match="token masks"):
        model(torch.zeros(2, 3, 8, 8), mask=torch.zeros(2, 4, dtype=torch.bool))

    cfg.model.hidden_indices = None
    cfg.model.pretrained = True
    cfg.model.architecture = None
    model = Backbone(cfg.model, cfg.data.image_size)
    assert model(torch.zeros(2, 3, 8, 8))[0].shape[-1] == 5

    cfg.model.feature_dims = []
    with pytest.raises(ValueError, match="feature_dims"):
        Backbone(cfg.model, cfg.data.image_size)
    cfg.model.feature_dims = [5]
    cfg.model.architecture = {"model_type": "dummy"}
    cfg.model.pretrained = False
    monkeypatch.setattr(backbones.AutoConfig, "for_model", lambda *args, **kwargs: SimpleNamespace())
    with pytest.raises(ValueError, match="drop_path_rate"):
        Backbone(cfg.model, cfg.data.image_size)


def test_llm_custom_unknown_and_forward_guards(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(backbones, "EVAVisionTransformer", lambda **kwargs: DummyNet((1024,)))
    monkeypatch.setattr(backbones, "hf_hub_download", lambda *args, **kwargs: "weights.pt")
    monkeypatch.setattr(
        backbones,
        "load_state",
        lambda path: {"visual.weight": torch.ones(1), "other": torch.ones(1)},
    )
    cfg.model.backend = "llm2clip"
    cfg.model.pretrained = True
    cfg.model.features_only = False
    cfg.model.checkpoint_path = None
    with pytest.raises(ValueError, match="336"):
        Backbone(cfg.model, [64, 64])
    model = Backbone(cfg.model, [336, 336])
    assert list(model.net.loaded[0]) == ["weight"]
    assert hasattr(model.net, "mask_token")
    assert model.supports_token_mask
    plain = model(torch.zeros(2, 3, 4, 4))
    assert plain[0].shape == (2, 1024, 2, 2)
    masked = model(torch.zeros(2, 3, 4, 4), mask=torch.ones(2, 4, dtype=torch.bool))
    assert masked[0].shape == (2, 1024, 2, 2)
    assert not torch.equal(plain[0], masked[0])

    cfg.model.backend = "custom"
    cfg.model.pretrained = False
    cfg.model.kwargs = {"_target_": "torch.nn.Identity"}
    cfg.model.feature_dims = [3]
    cfg.model.layout = "NHWC"
    model = Backbone(cfg.model, [8, 8])
    result = model(torch.zeros(2, 4, 4, 3))
    assert result[0].shape == (2, 3, 4, 4)
    assert not model.supports_token_mask
    with pytest.raises(ValueError, match="token masks"):
        model(torch.zeros(2, 4, 4, 3), mask=torch.zeros(2, 4, dtype=torch.bool))
    model.dims = [3, 3]
    with pytest.raises(ValueError, match="levels"):
        model(torch.zeros(2, 4, 4, 3))
    model.dims = [4]
    with pytest.raises(ValueError, match="dimension"):
        model(torch.zeros(2, 4, 4, 3))

    cfg.model.backend = "invalid"
    with pytest.raises(ValueError, match="Unknown backbone"):
        Backbone(cfg.model, [8, 8])


@pytest.mark.parametrize("kind", ["gap", "max", "avgmax", "attn", "gem", "signed_gem"])
def test_all_pooling_kinds(kind):
    cfg = SimpleNamespace(kind=kind, p=3.0, trainable=True, attention_hidden=2)
    pool = Pool(4, cfg)
    result = pool(torch.randn(2, 4, 3, 3))
    assert result.shape[0] == 2
    assert torch.isfinite(result).all()


def test_pooling_guards():
    with pytest.raises(ValueError, match="Unknown pooling"):
        Pool(4, SimpleNamespace(kind="bad", p=1, trainable=False, attention_hidden=2))
    cls = Pool(4, SimpleNamespace(kind="cls", p=1, trainable=False, attention_hidden=2), 0)
    with pytest.raises(ValueError, match="class token"):
        cls(torch.randn(2, 3, 4))
    cls.prefix = 1
    assert cls(torch.randn(2, 3, 4)).shape == (2, 4)
    assert cls(torch.randn(2, 4)).shape == (2, 4)
    with pytest.raises(ValueError, match="Expected"):
        cls(torch.randn(2, 1, 1, 1, 1))
    gap = Pool(4, SimpleNamespace(kind="gap", p=1, trainable=False, attention_hidden=2), 1)
    with pytest.raises(ValueError, match="No spatial"):
        gap(torch.randn(2, 1, 4))


def test_reid_freeze_and_local_guard(cfg):
    model = ReIDModel(cfg)
    model.train()
    bn = next(module for module in model.backbone.modules() if isinstance(module, nn.BatchNorm2d))
    model.freeze_backbone(True)
    assert model.frozen
    assert not model.backbone.training
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    before = bn.num_batches_tracked
    assert before is not None
    tracked = before.detach().clone()
    frozen = model(torch.randn(2, 3, 64, 64))
    assert frozen["embedding"].shape[0] == 2
    with pytest.raises(ValueError, match="token masks"):
        model(torch.randn(2, 3, 64, 64), mask=torch.zeros(2, 4, dtype=torch.bool))
    frozen_count = bn.num_batches_tracked
    assert frozen_count is not None
    assert torch.equal(frozen_count, tracked)
    model.freeze_backbone(False)
    assert model.backbone.training
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
    model(torch.randn(2, 3, 64, 64))
    after = bn.num_batches_tracked
    assert after is not None
    assert int(after.item()) == int(tracked.item()) + 1

    cfg.model.head.local_parts = 100
    model = ReIDModel(cfg)
    with pytest.raises(ValueError, match="local_parts"):
        model(torch.randn(2, 3, 64, 64))


@pytest.mark.parametrize(
    "model_name",
    [
        "convnext_tiny",
        "resnet50",
        "swin",
        "vit",
        "dinov3_convnext_base",
        "dinov3_convnext_large",
        "dinov3_convnext_multilevel",
        "radio",
        "llm2clip",
    ],
)
def test_every_model_config_forward(model_name, monkeypatch):
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
    ):
        cfg = compose(config_name="config", overrides=[f"model={model_name}", "model.pretrained=false"])
    if model_name == "llm2clip":
        cfg.data.image_size = [336, 336]
    monkeypatch.setattr(reid_module, "Backbone", ConfigBackbone)
    model = ReIDModel(cfg)
    output = model(torch.randn(2, 3, 16, 16))
    assert output["embedding"].shape == (2, cfg.model.head.embedding_dim)
    assert output["patches"].ndim == 3


def test_image_geometry_and_spatial_multiple(cfg):
    cfg.model.backend = "custom"
    assert spatial_multiple(cfg.model) == 1
    cfg.model.backend = "timm"
    cfg.model.spatial_multiple = None
    assert spatial_multiple(cfg.model) == 1
    cfg.model.spatial_multiple = 0
    with pytest.raises(ValueError, match="spatial_multiple"):
        spatial_multiple(cfg.model)
    cfg.model.spatial_multiple = 16
    assert scaled_hw(256, 256, 1.0, 16) == (256, 256)
    assert scaled_hw(256, 256, 0.9, 16) == (224, 224)
    with pytest.raises(ValueError, match="scales positive"):
        scaled_hw(256, 256, 0.0, 16)

    cfg.data.image_size = [0, 256]
    with pytest.raises(ValueError, match="height, width"):
        validate_image_geometry(cfg)
    cfg.data.image_size = [256]
    with pytest.raises(ValueError, match="height, width"):
        validate_image_geometry(cfg)
    cfg.data.image_size = [255, 255]
    with pytest.raises(ValueError, match="divisible"):
        validate_image_geometry(cfg)
    cfg.data.image_size = [256, 256]
    cfg.eval.tta.enabled = True
    cfg.eval.tta.scales = [1.0, 0.9]
    validate_image_geometry(cfg)
    cfg.eval.tta.scales = [0.0]
    with pytest.raises(ValueError, match="scales positive"):
        validate_image_geometry(cfg)
    cfg.eval.tta.scales = [1.0, 0.9]
    cfg.model.backend = "llm2clip"
    with pytest.raises(ValueError, match="336"):
        validate_image_geometry(cfg)
    cfg.data.image_size = [336, 336]
    cfg.model.spatial_multiple = 14
    with pytest.raises(ValueError, match="scale TTA"):
        validate_image_geometry(cfg)
    cfg.eval.tta.scales = [1.0]
    validate_image_geometry(cfg)


def test_backbone_rejects_incompatible_image_size(cfg, monkeypatch):
    monkeypatch.setattr(backbones.timm, "create_model", lambda *args, **kwargs: DummyNet((4,)))
    cfg.model.spatial_multiple = 32
    with pytest.raises(ValueError, match="divisible"):
        Backbone(cfg.model, [255, 255])


def test_swin_config_aligns_with_data_size():
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
    ):
        cfg = compose(config_name="config", overrides=["model=swin"])
    assert list(cfg.data.image_size) == [256, 256]
    assert cfg.model.spatial_multiple == 32
    assert cfg.model.bind_image_size is True
    assert cfg.model.kwargs.strict_img_size is False


def test_llm2clip_multilevel_uses_selected_blocks(cfg, monkeypatch):
    def factory(**kwargs):
        return EVAVisionTransformer(
            img_size=28,
            patch_size=14,
            embed_dim=32,
            depth=4,
            num_heads=2,
            mlp_ratio=2.0,
            qkv_bias=True,
            num_classes=0,
            use_mean_pooling=False,
            rope=False,
        )

    monkeypatch.setattr(backbones, "EVAVisionTransformer", factory)
    cfg.model.backend = "llm2clip"
    cfg.model.pretrained = False
    cfg.model.features_only = True
    cfg.model.out_indices = [1, -1]
    model = Backbone(cfg.model, [336, 336])
    assert model.level_indices == [1, 3]
    assert model.dims == [32, 32]
    images = torch.randn(2, 3, 28, 28)
    levels = model(images)
    assert len(levels) == 2
    final = model.net.norm(model.net.forward_features(images, return_all_features=True))
    torch.testing.assert_close(levels[1], final)
    assert not torch.equal(levels[0], levels[1])
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[0, 0] = True
    masked = model(images, mask=mask)
    assert not torch.equal(masked[0], levels[0])
    with pytest.raises(ValueError, match="require"):
        backbones.llm2clip_level_indices(None, 4)
    with pytest.raises(ValueError, match="require"):
        backbones.llm2clip_level_indices([], 4)
    with pytest.raises(ValueError, match="outside"):
        backbones.llm2clip_level_indices([4], 4)
    with pytest.raises(ValueError, match="unique"):
        backbones.llm2clip_level_indices([1, -3], 4)
    with open_dict(cfg.model):
        cfg.model.multilevel_fuse = "mean"
    averaged = Backbone(cfg.model, [336, 336])
    assert averaged.dims == [32]
    mean_levels = averaged(images)
    assert len(mean_levels) == 1
    concat = backbones.llm2clip_levels(averaged.net, images, None, [1, 3], "concat")
    torch.testing.assert_close(mean_levels[0], torch.stack([averaged.net.norm(concat[0]), concat[1]]).mean(0))
    with open_dict(cfg.model):
        cfg.model.multilevel_fuse = "stack"
    with pytest.raises(ValueError, match="concat/mean"):
        Backbone(cfg.model, [336, 336])
    assert backbones.llm2clip_fuse_name(SimpleNamespace()) == "concat"
    assert backbones.llm2clip_fuse_name(SimpleNamespace(multilevel_fuse="mean")) == "mean"


def test_eva02_multilevel_keeps_trial23_recipe():
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
    ):
        tuned = compose(config_name="config", overrides=["experiment=current_best_tuned"])
        cfg = compose(config_name="config", overrides=["experiment=eva02_multilevel"])
        late = compose(config_name="config", overrides=["experiment=eva02_multilevel_late"])
        k4 = compose(config_name="config", overrides=["experiment=eva02_k4"])
    assert cfg.name == "eva02_multilevel"
    assert cfg.model.backend == "llm2clip"
    assert cfg.model.features_only is True
    assert list(cfg.model.out_indices) == [11, 17, 23]
    assert cfg.model.multilevel_fuse == "concat"
    assert cfg.model.head.local_parts == 0
    assert cfg.data.sampler.instances == 2
    assert cfg.model.pooling.kind == tuned.model.pooling.kind
    assert cfg.train.epochs == tuned.train.epochs
    assert list(cfg.data.image_size) == [336, 336]
    assert cfg.eval.weights == "ema"
    assert tuned.model.features_only is False
    assert late.name == "eva02_multilevel_late"
    assert list(late.model.out_indices) == [20, 21, 22, 23]
    assert late.model.multilevel_fuse == "mean"
    assert late.data.sampler.instances == 2
    assert late.train.epochs == tuned.train.epochs
    assert k4.name == "eva02_k4"
    assert k4.model.backend == "llm2clip"
    assert k4.model.features_only is False
    assert k4.data.sampler.identities == tuned.data.sampler.identities
    assert k4.data.sampler.instances == 4
    assert tuned.data.sampler.instances == 4
    assert k4.train.epochs == tuned.train.epochs
    assert k4.train.accumulate_grad_batches == tuned.train.accumulate_grad_batches
    assert list(k4.data.image_size) == [336, 336]


def test_eva_mask_token_and_vector_features(cfg, monkeypatch):
    net = EVAVisionTransformer(
        img_size=28,
        patch_size=14,
        embed_dim=32,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        qkv_bias=True,
        num_classes=0,
        use_mean_pooling=False,
        rope=False,
    )
    x = torch.randn(2, 3, 28, 28)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[0, 1] = True
    with pytest.raises(ValueError, match="mask_token"):
        net.forward_features(x, return_all_features=True, bool_masked_pos=mask)
    net.mask_token = nn.Parameter(torch.zeros(1, 1, 32))
    plain = net.forward_features(x, return_all_features=True)
    masked = net.forward_features(x, return_all_features=True, bool_masked_pos=mask)
    assert not torch.equal(plain, masked)

    class FlatBackbone(nn.Module):
        def __init__(self, model_cfg, image_size, initialize_pretrained=True):
            super().__init__()
            self.dims = [4]
            self.prefix = 0

        def forward(self, x, mask=None):
            return [torch.ones(len(x), 4)]

    monkeypatch.setattr(reid_module, "Backbone", FlatBackbone)
    model = ReIDModel(cfg)
    output = model(torch.randn(2, 3, 8, 8))
    assert "patches" not in output
    assert output["embedding"].shape[0] == 2

    class TokenBackbone(nn.Module):
        def __init__(self, model_cfg, image_size, initialize_pretrained=True):
            super().__init__()
            self.dims = [4]
            self.prefix = 1

        def forward(self, x, mask=None):
            return [torch.ones(len(x), 5, 4)]

    monkeypatch.setattr(reid_module, "Backbone", TokenBackbone)
    tokens = ReIDModel(cfg)(torch.randn(2, 3, 8, 8))
    assert tokens["patches"].shape == (2, 4, 4)
