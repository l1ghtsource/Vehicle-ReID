import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import lightning as L
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from PIL import Image
from torch import nn

import pretrain as pretrain_module
import train
from dataset.folds import fingerprint
from dataset.pretrain import (
    READERS,
    SSL_CROP,
    SSL_FULL,
    SSL_SOURCE,
    PretrainDataModule,
    load_external_data,
    read_test_train_ssl,
    read_veri,
    read_vric,
    ssl_image_mode,
)


def write_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 16), (12, 34, 56)).save(path)


def write_veri(root: Path, identities: int = 4, declared: bool = True):
    image_dir = root / "image_train"
    items = []
    for identity in range(identities):
        for camera in range(2):
            name = f"{identity:04d}_c{camera + 1:03d}_0.jpg"
            write_image(image_dir / name)
            items.append(
                f'        <Item imageName="{name}" vehicleID="{identity:04d}" cameraID="c{camera + 1:03d}" />'
            )
    body = "<TrainingImages>\n    <Items>\n" + "\n".join(items) + "\n    </Items>\n</TrainingImages>\n"
    prefix = '<?xml version="1.0" encoding="gb2312" ?>\n' if declared else ""
    (root / "train_label.xml").write_bytes((prefix + body).encode("gb18030"))
    return root


def write_vric(root: Path, identities: int = 4):
    image_dir = root / "train_images"
    lines = []
    for identity in range(identities):
        for camera in range(2):
            name = f"img_{identity}_{camera}.jpg"
            write_image(image_dir / name)
            lines.append(f"{name} {identity + 1} {camera + 1}")
    (root / "vric_train.txt").write_text("\n".join(lines) + "\n")
    return root


def attach_pretrain(cfg, tmp_path, datasets=("veri", "vric")):
    veri_root = write_veri(tmp_path / "VeRi")
    vric_root = write_vric(tmp_path / "VRIC")
    with open_dict(cfg):
        cfg.pretrain = {
            "datasets": list(datasets),
            "validation_csv": cfg.data.train_csv,
            "veri": {"root": str(veri_root)},
            "vric": {"root": str(vric_root)},
        }
    return cfg


def test_pretrain_config_composes():
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
    ):
        cfg = compose(config_name="pretrain", overrides=["experiment=smoke"])
    assert list(cfg.pretrain.datasets) == ["veri", "vric"]
    assert cfg.pretrain.validation_csv == cfg.data.train_csv
    assert cfg.init_checkpoint is None


def test_current_best_tuned_matches_eva02_oof():
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")
    ):
        cfg = compose(config_name="pretrain", overrides=["experiment=current_best_tuned"])
        swapped = compose(
            config_name="pretrain",
            overrides=[
                "experiment=current_best_tuned",
                "model=dinov3_convnext_base",
                "pretrain.datasets=[vric]",
            ],
        )
        ssl = compose(
            config_name="pretrain",
            overrides=[
                "experiment=current_best_tuned",
                "pretrain.datasets=[test_train_ssl]",
            ],
        )
    pretrain_module.apply_ssl_pretrain(ssl)
    assert cfg.model.name == "microsoft/LLM2CLIP-EVA02-L-14-336"
    assert cfg.train.epochs == 27
    assert cfg.train.accumulate_grad_batches == 4
    assert cfg.train.ema.enabled is True
    assert cfg.model.pooling.kind == "attn"
    assert cfg.model.head.local_parts == 0
    assert cfg.model.head.embedding_dim == 256
    assert list(cfg.data.image_size) == [336, 336]
    assert list(cfg.eval.tta.scales) == [1.0]
    assert cfg.data.sampler.identities == 16
    assert cfg.data.sampler.instances == 2
    assert cfg.scheduler.kind == "linear"
    assert list(cfg.scheduler.milestones) == [27, 28]
    assert cfg.eval.weights == "ema"
    assert cfg.loss.terms[0].params.margin == pytest.approx(0.4789452160852028)
    assert cfg.optimizer.lr == pytest.approx(0.0006983608478082421)
    assert "dinov3-convnext-base" in swapped.model.name
    assert list(swapped.data.image_size) == [336, 336]
    assert swapped.model.head.local_parts == 0
    assert list(swapped.pretrain.datasets) == ["vric"]
    assert list(ssl.pretrain.datasets) == ["test_train_ssl"]
    assert ssl.loss.terms[0].name == "dino"
    assert ssl.data.sampler.kind == "random"
    assert ssl_image_mode("test_train_ssl") == "crop"
    assert ssl_image_mode("test_train_ssl_crop") == "crop"
    assert ssl_image_mode("test_train_ssl_full") == "full"
    assert ssl_image_mode("vric") is None


def test_read_veri_and_vric_formats(tmp_path):
    veri = read_veri(write_veri(tmp_path / "declared"))
    assert veri.source.unique().tolist() == ["veri"]
    assert veri.full_image.all()
    assert veri.image_id.str.startswith("veri:").all()
    assert Path(veri.image_path.iloc[0]).is_file()

    undeclared = read_veri(write_veri(tmp_path / "plain", declared=False))
    assert len(undeclared) == len(veri)

    vric = read_vric(write_vric(tmp_path / "vric"))
    assert vric.source.unique().tolist() == ["vric"]
    assert vric.identity_key.str.startswith("vric:").all()


def test_external_reader_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Incomplete VeRi"):
        read_veri(tmp_path / "missing-veri")
    empty_veri = tmp_path / "empty-veri"
    (empty_veri / "image_train").mkdir(parents=True)
    (empty_veri / "train_label.xml").write_bytes(b"<TrainingImages><Items/></TrainingImages>")
    with pytest.raises(ValueError, match="empty"):
        read_veri(empty_veri)

    with pytest.raises(FileNotFoundError, match="Incomplete VRIC"):
        read_vric(tmp_path / "missing-vric")
    empty_vric = tmp_path / "empty-vric"
    (empty_vric / "train_images").mkdir(parents=True)
    (empty_vric / "vric_train.txt").write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_vric(empty_vric)
    (empty_vric / "vric_train.txt").write_text("image 1\n")
    with pytest.raises(ValueError, match="expected image, identity, camera"):
        read_vric(empty_vric)


def test_load_external_data_mix_and_guards(data_cfg, tmp_path, monkeypatch):
    cfg = attach_pretrain(data_cfg, tmp_path)
    mixed = load_external_data(cfg)
    assert set(mixed.source) == {"veri", "vric"}
    assert mixed.vehicle_id.nunique() == mixed.identity_key.nunique()
    assert mixed.camera_id.nunique() == mixed.camera_key.nunique()
    assert mixed.image_id.is_unique

    cfg.pretrain.datasets = ["veri"]
    assert load_external_data(cfg).source.unique().tolist() == ["veri"]

    cfg.pretrain.datasets = []
    with pytest.raises(ValueError, match="unique dataset names"):
        load_external_data(cfg)
    cfg.pretrain.datasets = ["veri", "veri"]
    with pytest.raises(ValueError, match="unique dataset names"):
        load_external_data(cfg)
    cfg.pretrain.datasets = ["madcars"]
    with pytest.raises(ValueError, match="Unknown pretraining datasets"):
        load_external_data(cfg)

    cfg.pretrain.datasets = ["veri", "vric"]
    cloned = load_external_data(cfg)
    monkeypatch.setitem(
        READERS,
        "vric",
        lambda root: cloned.loc[cloned.source == "veri"].assign(source="vric"),
    )
    with pytest.raises(ValueError, match="unique"):
        load_external_data(cfg)


def test_test_train_ssl_reader_and_guards(data_cfg, tmp_path, monkeypatch):
    cfg = attach_pretrain(data_cfg, tmp_path)
    cfg.pretrain.datasets = [SSL_SOURCE]
    frame = read_test_train_ssl(cfg)
    assert set(frame.source) == {SSL_CROP}
    assert frame.full_image.eq(False).all()
    mixed = load_external_data(cfg)
    assert mixed.image_id.is_unique
    assert mixed.vehicle_id.nunique() == len(mixed)

    full = read_test_train_ssl(cfg, "full")
    assert set(full.source) == {SSL_FULL}
    assert full.full_image.all()
    cfg.pretrain.datasets = [SSL_FULL]
    assert load_external_data(cfg).full_image.all()
    with pytest.raises(ValueError, match="crop or full"):
        read_test_train_ssl(cfg, "both")

    cfg.pretrain.datasets = ["veri", SSL_SOURCE]
    with pytest.raises(ValueError, match="cannot mix"):
        load_external_data(cfg)
    cfg.pretrain.datasets = [SSL_CROP, SSL_FULL]
    with pytest.raises(ValueError, match="cannot mix"):
        load_external_data(cfg)

    query = pd.read_csv(cfg.data.query_csv)
    query.loc[0, "image_id"] = pd.read_csv(cfg.data.train_csv).image_id.iloc[0]
    query.to_csv(cfg.data.query_csv, index=False)
    with pytest.raises(ValueError, match="unique"):
        read_test_train_ssl(cfg)
    deduped = read_test_train_ssl(cfg, "full")
    assert deduped.image_id.is_unique

    empty_frame = pd.DataFrame(
        {
            "image_id": pd.Series(dtype=str),
            "x": pd.Series(dtype=float),
            "y": pd.Series(dtype=float),
            "w": pd.Series(dtype=float),
            "h": pd.Series(dtype=float),
            "camera_id": pd.Series(dtype=int),
            "vehicle_id": pd.Series(dtype=int),
            "row_id": pd.Series(dtype=int),
        }
    )
    monkeypatch.setattr("dataset.pretrain.read_annotations", lambda *args, **kwargs: empty_frame)
    with pytest.raises(ValueError, match="empty"):
        read_test_train_ssl(cfg)


def test_ssl_pretrain_datamodule_two_views(data_cfg, tmp_path):
    cfg = attach_pretrain(data_cfg, tmp_path)
    cfg.pretrain.datasets = [SSL_SOURCE]
    cfg.data.sampler.kind = "random"
    dm = PretrainDataModule(cfg)
    dm.prepare_data()
    dm.setup("fit")
    item = dm.train_set[0]
    assert "view" in item
    assert item["view"].shape == item["image"].shape
    dm.trainer = cast(L.Trainer, SimpleNamespace(global_rank=0, world_size=1))
    batch = next(iter(dm.train_dataloader()))
    assert "view" in batch
    assert batch["image"].ndim == 4
    cfg.pretrain.datasets = [SSL_FULL]
    full_dm = PretrainDataModule(cfg)
    full_dm.setup("fit")
    assert full_dm.train_frame.full_image.all()


def test_apply_ssl_pretrain_guards(data_cfg, tmp_path):
    cfg = attach_pretrain(data_cfg, tmp_path)
    assert cfg.data.sampler.kind == "pk"
    pretrain_module.apply_ssl_pretrain(cfg)
    assert cfg.data.sampler.kind == "pk"
    cfg.pretrain.datasets = [SSL_SOURCE, "veri"]
    with pytest.raises(ValueError, match="cannot mix"):
        pretrain_module.apply_ssl_pretrain(cfg)
    cfg.pretrain.datasets = [SSL_CROP, SSL_FULL]
    with pytest.raises(ValueError, match="cannot mix"):
        pretrain_module.apply_ssl_pretrain(cfg)
    cfg.pretrain.datasets = [SSL_SOURCE]
    cfg.train.ema.enabled = False
    with pytest.raises(ValueError, match="ema"):
        pretrain_module.apply_ssl_pretrain(cfg)
    with open_dict(cfg):
        cfg.train.ema.enabled = True
        cfg.data.sampler.kind = "pk"
    pretrain_module.apply_ssl_pretrain(cfg)
    assert cfg.data.sampler.kind == "random"
    assert [str(term.name) for term in cfg.loss.terms] == ["dino"]
    cfg.pretrain.datasets = [SSL_FULL]
    pretrain_module.apply_ssl_pretrain(cfg)
    assert [str(term.name) for term in cfg.loss.terms] == ["dino"]


def test_pretrain_main_applies_ssl(data_cfg, tmp_path, monkeypatch):
    cfg = attach_pretrain(data_cfg, tmp_path, datasets=(SSL_SOURCE,))
    cfg.output_dir = str(tmp_path / "ssl-run")
    cfg.trainer.accelerator = "cpu"
    cfg.trainer.devices = 1
    cfg.resume = None
    cfg.train.ema.enabled = True
    cfg.data.sampler.kind = "pk"
    monkeypatch.setattr(pretrain_module, "PretrainDataModule", FakePretrainDataModule)
    monkeypatch.setattr(pretrain_module, "ReIDModule", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(pretrain_module, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(pretrain_module, "CSVLogger", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(pretrain_module.L, "Trainer", FakeTrainer)
    monkeypatch.setattr(pretrain_module.L, "seed_everything", lambda *args, **kwargs: None)
    FakeTrainer.instances.clear()
    FakeTrainer.global_zero = True
    pretrain_module.main.__wrapped__(cfg)
    assert [str(term.name) for term in cfg.loss.terms] == ["dino"]
    assert cfg.data.sampler.kind == "random"
    summary = json.loads((tmp_path / "ssl-run/run_summary.json").read_text())
    assert summary["datasets"] == [SSL_SOURCE]


def test_pretrain_datamodule_uses_full_competition_train(data_cfg, tmp_path):
    cfg = attach_pretrain(data_cfg, tmp_path)
    dm = PretrainDataModule(cfg)
    dm.prepare_data()
    dm.setup("fit")
    validation = pd.read_csv(cfg.data.train_csv)
    assert dm.validation_frame is not None
    assert len(dm.train_frame) == 16
    assert dm.num_classes == 8
    assert len(dm.validation_frame) == len(validation)
    assert set(dm.query_frame.image_id).isdisjoint(dm.gallery_frame.image_id)
    used = set(dm.query_frame.image_id).union(dm.gallery_frame.image_id)
    assert used <= set(validation.image_id.astype(str))
    assert fingerprint(dm.folds) == fingerprint(dm.train_frame)
    item = dm.train_set[0]
    assert item["image"].ndim == 3
    dm.trainer = cast(L.Trainer, SimpleNamespace(global_rank=0, world_size=1))
    assert next(iter(dm.train_dataloader()))["image"].ndim == 4
    assert next(iter(dm.val_dataloader()))["image"].ndim == 4


class FakePretrainDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.folds = pd.DataFrame({"image_id": ["a", "b"], "vehicle_id": [1, 2]})
        self.label_map = {1: 0, 2: 1}
        self.num_classes = 2
        self.train_frame = self.folds
        self.validation_frame = pd.DataFrame({"image_id": ["q"]})
        self.saved = None

    def prepare_data(self):
        self.prepared = True

    def setup(self, stage=None):
        self.stage = stage

    def save_split(self, path):
        self.saved = path
        Path(path).mkdir(parents=True, exist_ok=True)


class FakeCheckpoint:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.best_model_path = "best.ckpt"
        self.last_model_path = "last.ckpt"


class FakeTrainer:
    instances = []
    global_zero = True

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.is_global_zero = self.global_zero
        self.fit_args = None
        self.instances.append(self)

    def fit(self, model, datamodule=None, ckpt_path=None):
        self.fit_args = (model, datamodule, ckpt_path)


def test_pretrain_main_all_paths(data_cfg, tmp_path, monkeypatch):
    cfg = attach_pretrain(data_cfg, tmp_path)
    cfg.output_dir = str(tmp_path / "pretrain-run")
    cfg.trainer.accelerator = "cpu"
    cfg.trainer.devices = 2
    cfg.trainer.strategy = "auto"
    cfg.resume = None
    monkeypatch.setattr(pretrain_module, "PretrainDataModule", FakePretrainDataModule)
    monkeypatch.setattr(pretrain_module, "ReIDModule", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(pretrain_module, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(pretrain_module, "CSVLogger", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(pretrain_module.L, "Trainer", FakeTrainer)
    monkeypatch.setattr(pretrain_module.L, "seed_everything", lambda *args, **kwargs: None)
    FakeTrainer.instances.clear()
    FakeTrainer.global_zero = True
    pretrain_module.main.__wrapped__(cfg)
    summary = json.loads((tmp_path / "pretrain-run/run_summary.json").read_text())
    assert summary["datasets"] == ["veri", "vric"]
    assert summary["train_identities"] == 2
    assert FakeTrainer.instances[-1].kwargs["strategy"] == "ddp_find_unused_parameters_true"

    dm = FakePretrainDataModule(cfg)
    resume = tmp_path / "resume.ckpt"
    torch.save({"data_fingerprint": fingerprint(dm.folds), "label_map": dm.label_map}, resume)
    cfg.resume = str(resume)
    cfg.trainer.devices = 1
    cfg.trainer.limit_val_batches = 1.0
    summary_path = tmp_path / "pretrain-run/run_summary.json"
    summary_path.unlink()
    FakeTrainer.global_zero = False
    pretrain_module.main.__wrapped__(cfg)
    assert FakeTrainer.instances[-1].fit_args[2] == str(resume)
    callback = FakeTrainer.instances[-1].kwargs["callbacks"][0]
    assert callback.kwargs["dirpath"] == resume.resolve().parent
    assert not summary_path.exists()

    torch.save(
        {
            "data_fingerprint": fingerprint(dm.folds),
            "split_fingerprint": "wrong",
            "label_map": dm.label_map,
        },
        resume,
    )
    with pytest.raises(ValueError, match="different pretraining data"):
        pretrain_module.main.__wrapped__(cfg)

    torch.save({"data_fingerprint": "wrong", "label_map": dm.label_map}, resume)
    with pytest.raises(ValueError, match="different pretraining data"):
        pretrain_module.main.__wrapped__(cfg)
    with pytest.raises(TypeError, match="mapping"):
        pretrain_module.container_dict(OmegaConf.create([1]))

    FakeTrainer.global_zero = True
    cfg.resume = None
    monkeypatch.setattr(
        pretrain_module,
        "PretrainDataModule",
        lambda cfg: SimpleNamespace(
            folds=dm.folds,
            label_map=dm.label_map,
            num_classes=2,
            train_frame=dm.train_frame,
            validation_frame=None,
            prepare_data=lambda: None,
            setup=lambda stage=None: None,
            save_split=lambda path: Path(path).mkdir(parents=True, exist_ok=True),
        ),
    )
    with pytest.raises(RuntimeError, match="Validation data was not prepared"):
        pretrain_module.main.__wrapped__(cfg)


class InitModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(state_dict, strict=strict)


def test_load_initial_weights_choices_and_mismatches(tmp_path):
    module = InitModule()
    path = tmp_path / "init.ckpt"
    torch.save(
        {
            "state_dict": dict(module.state_dict()),
            "validation_weights": "raw",
        },
        path,
    )
    assert train.load_initial_weights(module, path) == "raw"

    torch.save(
        {
            "state_dict": dict(module.state_dict()),
            "ema": {"shadow": dict(module.model.state_dict())},
            "validation_weights": "ema",
        },
        path,
    )
    assert train.load_initial_weights(module, path) == "ema"

    torch.save({"state_dict": {"model.weight": torch.ones(2, 1)}, "validation_weights": "raw"}, path)
    with pytest.raises(ValueError, match="same model, pooling, and head"):
        train.load_initial_weights(module, path)

    class Partial(InitModule):
        def load_state_dict(self, state_dict, strict=True, assign=False):
            return ["model.missing", "losses.term"], ["extra"]

    with pytest.raises(ValueError, match="model mismatch"):
        train.load_initial_weights(Partial(), path)

    class LossesOnly(InitModule):
        def load_state_dict(self, state_dict, strict=True, assign=False):
            return ["losses.term"], []

    torch.save(
        {"state_dict": {"model.weight": module.model.weight.detach()}, "validation_weights": "ema"},
        path,
    )
    assert train.load_initial_weights(LossesOnly(), path) == "raw"

    torch.save(
        {
            "format": train.SERVING_FORMAT,
            "state_dict": dict(module.model.state_dict()),
            "weights": "ema",
        },
        path,
    )
    assert train.load_initial_weights(module, path) == "ema"
    torch.save({"format": train.SERVING_FORMAT, "state_dict": {}}, path)
    with pytest.raises(ValueError, match="missing state_dict"):
        train.load_initial_weights(module, path)

    class Masked(InitModule):
        def __init__(self):
            super().__init__()
            self.model.mask_token = nn.Parameter(torch.zeros(1, 1))

    torch.save(
        {
            "format": train.SERVING_FORMAT,
            "state_dict": dict(module.model.state_dict()),
            "weights": "raw",
        },
        path,
    )
    assert train.load_initial_weights(Masked(), path) == "raw"


def test_train_main_init_checkpoint(cfg, tmp_path, monkeypatch):
    class TrackingModule(InitModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.kwargs = kwargs

    class LocalTrainer:
        def __init__(self, **kwargs):
            self.is_global_zero = True
            self.fit_args = None

        def fit(self, model, datamodule=None, ckpt_path=None):
            self.fit_args = (model, datamodule, ckpt_path)

    module = TrackingModule()
    path = tmp_path / "init.ckpt"
    torch.save(
        {
            "state_dict": dict(module.state_dict()),
            "ema": {"shadow": dict(module.model.state_dict())},
            "validation_weights": "ema",
        },
        path,
    )
    monkeypatch.setattr(
        train,
        "ReIDDataModule",
        lambda cfg: SimpleNamespace(
            folds=pd.DataFrame({"image_id": ["a"], "vehicle_id": [1]}),
            label_map={1: 0},
            num_classes=1,
            prepare_data=lambda: None,
            setup=lambda stage=None: None,
            save_split=lambda split: Path(split).mkdir(parents=True, exist_ok=True),
        ),
    )
    created = {}

    def make_module(*args, **kwargs):
        created.update(kwargs)
        return TrackingModule(*args, **kwargs)

    monkeypatch.setattr(train, "ReIDModule", make_module)
    monkeypatch.setattr(train, "ModelCheckpoint", FakeCheckpoint)
    monkeypatch.setattr(train, "CSVLogger", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(train.L, "Trainer", LocalTrainer)
    monkeypatch.setattr(train.L, "seed_everything", lambda *args, **kwargs: None)
    cfg.output_dir = str(tmp_path / "init-run")
    cfg.resume = None
    cfg.init_checkpoint = str(path)
    cfg.trainer.devices = 1
    train.main.__wrapped__(cfg)
    assert created["initialize_pretrained"] is False
    summary = json.loads((tmp_path / "init-run/run_summary.json").read_text())
    assert summary["init_checkpoint"] == str(path)
    assert summary["initial_weights"] == "ema"
