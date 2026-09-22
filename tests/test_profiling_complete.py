from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset.images import VehicleDataset
from modules.inference import autocast_context, embed_frame, extract_frame, records_frame, tta_context_pcts
from profiling import (
    CONTEST_SUFFIXES,
    EXTRA_SUFFIXES,
    LIMIT_BYTES,
    STAGES,
    allocated_bytes,
    as_device,
    as_jsonable,
    bytes_text,
    checkpoint_breakdown,
    compare_embeddings,
    contest_summary,
    decode_rgb,
    embed_tensor,
    inventory,
    is_cuda,
    latency_score,
    make_runner,
    measure_latency,
    measure_stages,
    measure_throughput,
    measure_vram,
    mem_info,
    memory_snapshot,
    nested_tensor_bytes,
    peak_bytes,
    performance_scores,
    pick_device,
    profile_extract,
    read_file,
    reserved_bytes,
    reset_peak,
    serving_bytes,
    summarize_ms,
    synchronize,
    throughput_score,
    time_load,
    timed,
)
from profiling import device as device_mod
from profiling.extract import crop_record, decode_images, extract, preprocess_record
from profiling.run import BatchStream, RepeatDataset, make_loader_runner
from profiling.weights import is_contest_weight, is_extra_weight, iter_files


class DummyReID(nn.Module):
    def __init__(self, dim=8, bad=False, tensor_out=False):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, 3, padding=1)
        self.fc = nn.Linear(dim, dim)
        self.bad = bad
        self.tensor_out = tensor_out

    def forward(self, x):
        raw = self.fc(self.conv(x).mean((2, 3)))
        if self.bad:
            raw = raw + torch.nan
        emb = F.normalize(raw.float(), dim=1)
        return emb if self.tensor_out else {"embedding": emb}


def tiny_transform(arr):
    tensor = torch.from_numpy(np.ascontiguousarray(arr).copy()).permute(2, 0, 1).float() / 255
    return F.interpolate(tensor.unsqueeze(0), size=(16, 16), mode="bilinear", align_corners=False).squeeze(0)


def write_jpeg(path: Path, size=(20, 16), mode="RGB"):
    Image.new(mode, size, 40).save(path, format="JPEG")
    return path


def records(tmp_path: Path, n: int):
    paths = []
    bboxes = []
    for i in range(n):
        path = write_jpeg(tmp_path / f"{i}.jpg")
        paths.append(path)
        bboxes.append([1, 1, 16, 12])
    return paths, bboxes


def test_weight_inventory(tmp_path):
    root = tmp_path / "weights"
    nested = root / "aux"
    nested.mkdir(parents=True)
    (root / "SHA256SUMS").write_text("x")
    (root / ".gitkeep").write_text("")
    (root / "notes.txt").write_text("no")
    extra = root / "head.cbm"
    extra.write_bytes(b"cbm")
    sizes = {}
    for suffix in CONTEST_SUFFIXES:
        path = nested / f"w{suffix}"
        path.write_bytes(b"a" * (len(suffix) + 3))
        sizes[suffix] = path.stat().st_size
    missing = tmp_path / "absent"
    report = inventory([root, root, missing, nested / "w.pt"])
    assert report["within_limit"]
    assert report["limit_bytes"] == LIMIT_BYTES == 2 * 1024**3
    assert report["total_bytes"] == sum(sizes.values())
    assert report["extra_bytes"] == extra.stat().st_size
    assert {item["suffix"] for item in report["files"]} == set(CONTEST_SUFFIXES)
    assert report["extra_files"][0]["name"] == "head.cbm"
    assert EXTRA_SUFFIXES == (".cbm",)
    assert not is_contest_weight(root)
    assert is_extra_weight(extra)
    assert iter_files([missing]) == []
    monkey = tmp_path / "tiny.pt"
    monkey.write_bytes(b"abcd")
    tight = inventory([monkey], limit_bytes=3)
    assert not tight["within_limit"]
    assert tight["headroom_bytes"] < 0


def test_device_cpu_and_pick(monkeypatch):
    cpu = as_device(None)
    assert cpu.type == "cpu" and not is_cuda(cpu)
    assert as_device(cpu) is cpu
    synchronize("cpu")
    reset_peak("cpu")
    assert allocated_bytes("cpu") == peak_bytes("cpu") == reserved_bytes("cpu") == 0
    assert mem_info("cpu") is None
    snap = memory_snapshot("cpu")
    assert snap["free_bytes"] is None and snap["allocated_bytes"] == 0
    value, ms = timed(lambda: 7, "cpu")
    assert value == 7 and ms >= 0
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pick_device() == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    def fake_info(index):
        table = {0: (96, 100), 1: (10, 100), 2: (99, 100)}
        return table[int(index)]

    monkeypatch.setattr(torch.cuda, "mem_get_info", fake_info)
    assert pick_device(prefer=(2,), exclude=(0, 1)) == torch.device("cuda:2")
    assert pick_device(exclude=(0,)) == torch.device("cuda:2")
    assert pick_device(prefer=(1,), exclude=(0, 2)) == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (1, 100))
    assert pick_device(min_free_frac=0.95) == torch.device("cpu")


def test_device_cuda_hooks(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: 11)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 22)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: 33)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (44, 55))
    cuda = torch.device("cuda")
    synchronize(cuda)
    reset_peak(cuda)
    assert allocated_bytes(cuda) == 11
    assert peak_bytes(cuda) == 22
    assert reserved_bytes(cuda) == 33
    assert mem_info(cuda) == (44, 55)
    snap = memory_snapshot(cuda)
    assert snap["peak_bytes"] == 22 and snap["free_bytes"] == 44
    assert device_mod.is_cuda(cuda)


def test_extract_stages_and_guards(tmp_path):
    paths, bboxes = records(tmp_path, 3)
    write_jpeg(tmp_path / "gray.jpg", mode="L")
    model = DummyReID()
    model.eval()
    out = extract(paths[:2], bboxes[:2], tiny_transform, model, "cpu", context_pct=5)
    assert out.shape == (2, 8) and np.isfinite(out).all()
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    timed_out, detail = extract(
        paths[:1],
        bboxes[:1],
        tiny_transform,
        model,
        "cpu",
        timed=True,
        full_images=[True],
    )
    assert timed_out.shape == (1, 8)
    assert set(detail["batch_ms"]) == set(STAGES)
    assert detail["n"] == 1 and detail["total_ms"] >= 0
    payload = read_file(paths[0])
    image = decode_rgb(payload)
    cropped = crop_record(image, bboxes[0], 0.0, full_image=False)
    tensor = preprocess_record(cropped, tiny_transform)
    assert tensor.shape[0] == 3
    with pytest.raises(ValueError, match="Empty extract"):
        extract([], [], tiny_transform, model, "cpu")
    with pytest.raises(ValueError, match="full_images"):
        extract(paths[:1], bboxes[:1], tiny_transform, model, "cpu", full_images=[False, True])
    with pytest.raises(ValueError):
        extract(paths[:1], bboxes[:2], tiny_transform, model, "cpu")
    with pytest.raises(FileNotFoundError):
        extract([tmp_path / "missing.jpg"], [[0, 0, 4, 4]], tiny_transform, model, "cpu")
    with pytest.raises(ValueError, match="BBox"):
        extract(paths[:1], [[0, 0, 0, 0]], tiny_transform, model, "cpu")
    with pytest.raises(ValueError, match="precision"):
        extract(paths[:1], bboxes[:1], tiny_transform, model, "cpu", precision="fp8")
    with pytest.raises(FloatingPointError):
        extract(paths[:1], bboxes[:1], tiny_transform, DummyReID(bad=True), "cpu")
    tensor_model = DummyReID(tensor_out=True)
    tensor_model.eval()
    again = extract(paths[:1], bboxes[:1], tiny_transform, tensor_model, "cpu")
    assert again.shape == (1, 8)
    gray = extract([tmp_path / "gray.jpg"], [[0, 0, 8, 8]], tiny_transform, model, "cpu", precision="bf16")
    assert gray.shape == (1, 8)
    payload = read_file(paths[0])
    cv_image = decode_rgb(payload, "cv2")
    assert cv_image.mode == "RGB" and cv_image.size == decode_rgb(payload).size
    assert len(decode_images([payload, payload], backend="cv2", workers=2)) == 2
    assert len(decode_images([payload], backend="pil", workers=4)) == 1
    jpeg = decode_rgb(payload, "jpeg")
    assert jpeg.mode == "RGB" and jpeg.size == decode_rgb(payload).size
    with pytest.raises(ValueError, match="jpeg_cuda"):
        decode_rgb(payload, "jpeg_cuda")
    with pytest.raises(ValueError, match="pil/cv2/jpeg"):
        decode_rgb(payload, "turbo")
    with pytest.raises(ValueError, match="Failed to decode"):
        decode_rgb(b"not-an-image", "cv2")
    parallel = extract(
        paths[:2],
        bboxes[:2],
        tiny_transform,
        model,
        "cpu",
        decode_backend="cv2",
        decode_workers=2,
    )
    assert parallel.shape == (2, 8)
    jpeg_out = extract(paths[:1], bboxes[:1], tiny_transform, model, "cpu", decode_backend="jpeg")
    assert jpeg_out.shape == (1, 8)
    with pytest.raises(ValueError, match="jpeg_cuda"):
        extract(paths[:1], bboxes[:1], tiny_transform, model, "cpu", decode_backend="jpeg_cuda")
    with pytest.raises(ValueError, match="decode_workers"):
        extract(paths[:1], bboxes[:1], tiny_transform, model, "cpu", decode_workers=-1)


class CountingReID(DummyReID):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return super().forward(x)


def identity_tta(**kwargs):
    values = {"enabled": True, "scales": [1.0], "rotations": [0], "hflip": False, "context_pcts": None}
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_extract_context_tta_matches_eval_forwards(tmp_path):
    paths, bboxes = records(tmp_path, 1)
    model = CountingReID()
    model.eval()
    extract(
        paths,
        bboxes,
        tiny_transform,
        model,
        "cpu",
        context_pct=0.0,
        tta=identity_tta(context_pcts=[0, 50]),
    )
    assert model.calls == 2
    assert tta_context_pcts(identity_tta(context_pcts=[0, 50]), 10.0) == [0.0, 50.0]
    model.calls = 0
    extract(
        paths,
        bboxes,
        tiny_transform,
        model,
        "cpu",
        context_pct=5.0,
        tta=identity_tta(enabled=False, context_pcts=[0, 50]),
    )
    assert model.calls == 1
    model.calls = 0
    extract(paths, bboxes, tiny_transform, model, "cpu", context_pct=5.0, tta=identity_tta())
    assert model.calls == 1
    model.calls = 0
    extract(
        paths,
        bboxes,
        tiny_transform,
        model,
        "cpu",
        context_pct=5.0,
        tta=identity_tta(context_pcts=[]),
    )
    assert model.calls == 1
    shared = CountingReID()
    shared.eval()
    a = extract(paths, bboxes, tiny_transform, shared, "cpu", context_pct=0.0)
    b = extract(paths, bboxes, tiny_transform, shared, "cpu", context_pct=50.0)
    both = extract(
        paths,
        bboxes,
        tiny_transform,
        shared,
        "cpu",
        context_pct=99.0,
        tta=identity_tta(context_pcts=[0, 50]),
    )
    mean = (a + b) / 2
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    assert np.allclose(both, mean, atol=1e-5)


def test_extract_frame_matches_sequential_extract(cfg, tmp_path):
    paths, bboxes = records(tmp_path, 2)
    model = DummyReID()
    model.eval()
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.eval.precision = "fp32"
    cfg.eval.tta.enabled = False
    sequential = extract(paths, bboxes, tiny_transform, model, "cpu", context_pct=5.0)
    frame = records_frame(paths, bboxes)
    loaded = extract_frame(
        model,
        cfg,
        "cpu",
        frame,
        batch_size=2,
        transform=tiny_transform,
        context_pct=5.0,
    )
    assert np.allclose(sequential, loaded, atol=1e-5)


def test_loader_runner_streams_one_iterator(cfg, tmp_path, monkeypatch):
    paths, bboxes = records(tmp_path, 4)
    seen = []
    original = VehicleDataset.__getitem__

    def spy(self, index):
        seen.append(int(index))
        return original(self, index)

    monkeypatch.setattr(VehicleDataset, "__getitem__", spy)
    model = DummyReID()
    model.eval()
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False
    cfg.eval.precision = "fp32"
    cfg.eval.tta.enabled = False
    runner = make_loader_runner(paths, bboxes, tiny_transform, model, "cpu", cfg)
    for _ in range(8):
        assert runner(1).shape == (1, 8)
    assert seen[:8] == [0, 1, 2, 3, 0, 1, 2, 3]
    cold = runner.cold_start_s()
    assert len(cold) == 1
    assert cold[0]["batch_size"] == 1
    assert cold[0]["seconds"] >= 0
    runner.release()
    assert runner.cold_start_s()[0]["batch_size"] == 1
    with pytest.raises(ValueError, match="repeat dataset"):
        RepeatDataset([])
    closed = BatchStream(DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1))
    assert closed.iterator is None
    closed.close()


def test_loader_runner_matches_embed_frame_context_tta(cfg, tmp_path, monkeypatch):
    paths, bboxes = records(tmp_path, 2)
    model = CountingReID()
    model.eval()
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.context_pct = 0.0
    cfg.eval.precision = "fp32"
    cfg.eval.tta.enabled = True
    cfg.eval.tta.context_pcts = [0, 50]
    cfg.eval.tta.scales = [1.0]
    cfg.eval.tta.rotations = [0]
    cfg.eval.tta.hflip = False
    monkeypatch.setattr("modules.inference.build_transforms", lambda cfg: tiny_transform)
    runner = make_loader_runner(paths, bboxes, tiny_transform, model, "cpu", cfg)
    model.calls = 0
    got = runner(2)
    assert model.calls == 2
    frame = records_frame(paths, bboxes)
    model.calls = 0
    expected = embed_frame(model, cfg, "cpu", frame)
    assert model.calls == 2
    assert np.max(np.abs(got - expected)) < 1e-5


def test_loader_runner_releases_workers_between_batch_sizes(cfg, tmp_path, monkeypatch):
    paths, bboxes = records(tmp_path, 4)
    live = []

    class Iterator:
        def __init__(self, loader):
            self.loader = loader
            self.closed = False
            live.append(self)

        def __next__(self):
            if self.closed:
                raise RuntimeError("iterator already closed")
            return {"image": torch.zeros(self.loader.batch_size, 3, 8, 8)}

        def _shutdown_workers(self):
            self.closed = True
            live.remove(self)

    class Loader:
        def __init__(self, dataset, batch_size, **kwargs):
            self.batch_size = batch_size
            self.kwargs = kwargs

        def __iter__(self):
            return Iterator(self)

    monkeypatch.setattr("profiling.run.DataLoader", Loader)
    model = DummyReID()
    model.eval()
    cfg.data.num_workers = 2
    cfg.data.pin_memory = False
    cfg.eval.precision = "fp32"
    cfg.eval.tta.enabled = True
    cfg.eval.tta.context_pcts = [0, 50]
    cfg.eval.tta.scales = [1.0]
    cfg.eval.tta.rotations = [0]
    cfg.eval.tta.hflip = False
    runner = make_loader_runner(paths, bboxes, tiny_transform, model, "cpu", cfg)
    runner(1)
    runner(1)
    assert len(live) == 2
    runner(8)
    assert len(live) == 2
    assert {item.loader.batch_size for item in live} == {8}
    runner.release()
    assert live == []


def test_repeated_context_uses_the_same_images(cfg, tmp_path, monkeypatch):
    paths, bboxes = records(tmp_path, 4)
    for index, path in enumerate(paths):
        Image.new("RGB", (20, 16), (index * 40, 20, 80)).save(path, format="JPEG")
    seen = []
    original = VehicleDataset.__getitem__

    def spy(self, index):
        seen.append(int(index))
        return original(self, index)

    monkeypatch.setattr(VehicleDataset, "__getitem__", spy)
    monkeypatch.setattr("modules.inference.build_transforms", lambda cfg: tiny_transform)
    model = DummyReID()
    model.eval()
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.context_pct = 0.0
    cfg.eval.precision = "fp32"
    cfg.eval.tta.enabled = True
    cfg.eval.tta.context_pcts = [0, 0]
    cfg.eval.tta.scales = [1.0]
    cfg.eval.tta.rotations = [0]
    cfg.eval.tta.hflip = False
    runner = make_loader_runner(paths, bboxes, tiny_transform, model, "cpu", cfg)
    got = runner(2)
    assert seen[:4] == [0, 1, 0, 1]
    frame = records_frame(paths[:2], bboxes[:2])
    expected = embed_frame(model, cfg, "cpu", frame)
    assert np.max(np.abs(got - expected)) < 1e-5
    cold = runner.cold_start_s()
    assert [row["view"] for row in cold] == [0, 1]
    assert [row["context_pct"] for row in cold] == [0.0, 0.0]


def test_loader_runner_cold_start_skips_failed_open(cfg, tmp_path, monkeypatch):
    paths, bboxes = records(tmp_path, 2)

    class Iterator:
        def __next__(self):
            raise RuntimeError("worker failed")

        def _shutdown_workers(self):
            return None

    class Loader:
        def __init__(self, dataset, batch_size, **kwargs):
            self.batch_size = batch_size

        def __iter__(self):
            return Iterator()

    monkeypatch.setattr("profiling.run.DataLoader", Loader)
    cfg.data.num_workers = 0
    cfg.eval.precision = "fp32"
    cfg.eval.tta.enabled = False
    runner = make_loader_runner(paths, bboxes, tiny_transform, DummyReID(), "cpu", cfg)
    with pytest.raises(RuntimeError, match="worker failed"):
        runner(1)
    assert runner.cold_start_s() == []
    runner.release()


@pytest.mark.parametrize("stage", ["throughput", "vram"])
def test_profile_extract_releases_loader_when_measurement_raises(cfg, tmp_path, monkeypatch, stage):
    paths, bboxes = records(tmp_path, 2)
    live = []
    opened = []

    class Iterator:
        def __init__(self, loader):
            self.loader = loader
            self.steps = 0
            live.append(self)
            opened.append(self)

        def __next__(self):
            self.steps += 1
            if stage == "throughput" and len(opened) == 1 and self.steps == 2:
                raise RuntimeError("oom")
            if stage == "vram" and len(opened) >= 2:
                raise RuntimeError("oom")
            return {"image": torch.zeros(self.loader.batch_size, 3, 8, 8)}

        def _shutdown_workers(self):
            live.remove(self)

    class Loader:
        def __init__(self, dataset, batch_size, **kwargs):
            self.batch_size = batch_size

        def __iter__(self):
            return Iterator(self)

    monkeypatch.setattr("profiling.run.DataLoader", Loader)
    model = DummyReID()
    model.eval()
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.eval.precision = "fp32"
    cfg.eval.tta.enabled = False
    with pytest.raises(RuntimeError, match="oom"):
        profile_extract(
            roots=[tmp_path],
            paths=paths,
            bboxes=bboxes,
            transform=tiny_transform,
            model=model,
            device="cpu",
            load_ms=0.0,
            warmup=0,
            repeats=1,
            batch_sizes=(1,),
            min_seconds=0.0,
            vram_repeats=1,
            stage_repeats=1,
            cfg=cfg,
        )
    assert opened
    assert live == []
    assert len(opened) == (1 if stage == "throughput" else 2)


def test_embed_tta_and_amp(monkeypatch):
    model = DummyReID()
    model.eval()
    batch = torch.rand(2, 3, 16, 16)
    plain = embed_tensor(model, batch, "cpu")
    assert plain.shape == (2, 8)
    tta = SimpleNamespace(enabled=True, scales=[1.0], rotations=[15], hflip=True)
    flipped = embed_tensor(model, batch, "cpu", tta=tta)
    assert flipped.shape == (2, 8)
    off = SimpleNamespace(enabled=False, scales=[3.0], rotations=[90], hflip=True)
    assert embed_tensor(model, batch, "cpu", tta=off).shape == (2, 8)
    cfg = OmegaConf.create({"backend": "timm", "spatial_multiple": 1, "name": "t"})
    scaled = SimpleNamespace(enabled=True, scales=[1.25], rotations=[0], hflip=False)
    resized = embed_tensor(model, batch, "cpu", tta=scaled, model_cfg=cfg)
    assert resized.shape == (2, 8)
    llm = OmegaConf.create({"backend": "llm2clip", "spatial_multiple": 14, "name": "e"})
    ignore_scale = SimpleNamespace(enabled=True, scales=[2.0], rotations=[0], hflip=False)
    kept = embed_tensor(model, batch, "cpu", tta=ignore_scale, model_cfg=llm)
    assert kept.shape == (2, 8)
    with pytest.raises(ValueError, match="TTA"):
        embed_tensor(
            model,
            batch,
            "cpu",
            tta=SimpleNamespace(enabled=True, scales=[], rotations=[0], hflip=False),
        )
    with pytest.raises(ValueError, match="TTA"):
        embed_tensor(
            model,
            batch,
            "cpu",
            tta=SimpleNamespace(enabled=True, scales=[0.0], rotations=[0], hflip=False),
        )
    with autocast_context("cpu", "fp32"):
        pass
    with autocast_context("cpu", "bf16"):
        pass
    with pytest.raises(ValueError, match="precision"):
        autocast_context("cpu", "fp8")

    class DummyCtx:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    seen = []

    def fake_autocast(**kwargs):
        seen.append(kwargs["dtype"])
        return DummyCtx()

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    with autocast_context("cuda", "bf16"):
        pass
    with autocast_context("cuda", "fp16"):
        pass
    assert seen == [torch.bfloat16, torch.float16]


def test_protocol_and_profile(cfg, tmp_path):
    paths, bboxes = records(tmp_path, 4)
    model = DummyReID()
    model.eval()
    ckpt = {
        "state_dict": {"a": torch.ones(4)},
        "ema": {"shadow": {"a": torch.ones(4, 2)}},
        "opt": [torch.zeros(3)],
    }
    blob, load_ms, snap = time_load(lambda: model, "cpu")
    assert blob is model and load_ms >= 0 and snap["allocated_bytes"] == 0
    report = profile_extract(
        roots=[tmp_path],
        paths=paths,
        bboxes=bboxes,
        transform=tiny_transform,
        model=model,
        device="cpu",
        load_ms=load_ms,
        checkpoint=ckpt,
        weights="ema",
        context_pct=0.0,
        full_images=[False] * 4,
        warmup=0,
        repeats=3,
        batch_sizes=(1, 2),
        min_seconds=0.0,
        vram_repeats=1,
        stage_repeats=2,
        cfg=None,
    )
    assert report["contest"]["latency_b1_ms"] == report["latency"]["p50_ms"]
    assert report["throughput"]["best_fps"] > 0
    assert report["determinism"]["allclose_fp32"]
    assert report["stages_per_image_ms"]["forward"] >= 0
    assert serving_bytes(report["checkpoint"], "ema") == nested_tensor_bytes(ckpt["ema"]["shadow"])
    none_ckpt = profile_extract(
        roots=[tmp_path],
        paths=paths,
        bboxes=bboxes,
        transform=tiny_transform,
        model=model,
        device="cpu",
        load_ms=0.0,
        warmup=0,
        repeats=1,
        batch_sizes=(1,),
        min_seconds=0.0,
        vram_repeats=1,
        stage_repeats=1,
    )
    assert none_ckpt["checkpoint"] is None
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.decode_backend = "cv2"
    cfg.eval.precision = "fp32"
    cfg.eval.tta.enabled = False
    serving_report = profile_extract(
        roots=[tmp_path],
        paths=paths,
        bboxes=bboxes,
        transform=tiny_transform,
        model=model,
        device="cpu",
        load_ms=0.0,
        warmup=0,
        repeats=1,
        batch_sizes=(1, 2),
        min_seconds=0.0,
        vram_repeats=1,
        stage_repeats=1,
        cfg=cfg,
    )
    assert serving_report["throughput"]["best_fps"] > 0
    assert serving_report["throughput"]["cold_start_s"]
    runner = make_runner(paths, bboxes, tiny_transform, model, "cpu")
    assert runner(1).shape[0] == 1
    latency = measure_latency(lambda: runner(1), "cpu", warmup=1, repeats=2)
    assert latency["n"] == 2
    order = []

    def mark_warm():
        order.append("warm")

    def mark_stage():
        order.append("stage")
        return None, {"per_image_ms": {"forward": 1.0, "decode": 2.0}}

    stages = measure_stages(mark_warm, mark_stage, "cpu", warmup=2, repeats=3)
    assert order == ["warm", "warm", "stage", "stage", "stage"]
    assert stages == {"forward": 1.0, "decode": 2.0}
    with pytest.raises(ValueError, match="warmup"):
        measure_stages(mark_warm, mark_stage, "cpu", warmup=-1, repeats=1)
    with pytest.raises(ValueError, match="repeats"):
        measure_stages(mark_warm, mark_stage, "cpu", warmup=0, repeats=0)
    thr = measure_throughput(
        lambda size: runner(size), "cpu", batch_sizes=(1,), min_seconds=0, warmup_batches=0
    )
    assert thr["by_batch"][0]["images"] >= 1
    vram = measure_vram(lambda size: runner(size), "cpu", batch_sizes=(1,), repeats=1)
    assert vram[0]["peak_bytes"] == 0
    same = compare_embeddings(np.ones((2, 3)), np.ones((2, 3)))
    assert same["bit_identical"] and same["min_cosine"] > 0.99
    shifted = compare_embeddings(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert not shifted["bit_identical"]
    with pytest.raises(ValueError, match="warmup"):
        measure_latency(lambda: None, "cpu", warmup=-1, repeats=1)
    with pytest.raises(ValueError, match="repeats"):
        measure_latency(lambda: None, "cpu", warmup=0, repeats=0)
    with pytest.raises(ValueError, match="empty"):
        summarize_ms([])
    with pytest.raises(ValueError, match="batch_sizes"):
        measure_throughput(lambda size: None, "cpu", batch_sizes=(0,))
    with pytest.raises(ValueError, match="min_seconds"):
        measure_throughput(lambda size: None, "cpu", batch_sizes=(1,), min_seconds=-1)
    with pytest.raises(ValueError, match="warmup_batches"):
        measure_throughput(lambda size: None, "cpu", batch_sizes=(1,), warmup_batches=-1)
    with pytest.raises(ValueError, match="no images"):
        measure_throughput(
            lambda size: np.zeros((0, 2)),
            "cpu",
            batch_sizes=(1,),
            min_seconds=0,
            warmup_batches=0,
        )
    with pytest.raises(ValueError, match="repeats"):
        measure_vram(lambda size: None, "cpu", repeats=0)
    with pytest.raises(ValueError, match="shapes"):
        compare_embeddings(np.ones((2, 2)), np.ones((3, 2)))
    with pytest.raises(ValueError, match="empty"):
        compare_embeddings(np.zeros((0, 2)), np.zeros((0, 2)))
    with pytest.raises(ValueError, match="need at least"):
        profile_extract(
            roots=[],
            paths=paths[:1],
            bboxes=bboxes[:1],
            transform=tiny_transform,
            model=model,
            device="cpu",
            load_ms=0,
            batch_sizes=(1, 8),
            stage_repeats=1,
            warmup=0,
            repeats=1,
            min_seconds=0,
            vram_repeats=1,
        )
    with pytest.raises(ValueError, match="stage_repeats"):
        profile_extract(
            roots=[],
            paths=paths,
            bboxes=bboxes,
            transform=tiny_transform,
            model=model,
            device="cpu",
            load_ms=0,
            batch_sizes=(1,),
            stage_repeats=0,
            warmup=0,
            repeats=1,
            min_seconds=0,
            vram_repeats=1,
        )
    with pytest.raises(ValueError, match="batch_sizes"):
        profile_extract(
            roots=[],
            paths=paths,
            bboxes=bboxes,
            transform=tiny_transform,
            model=model,
            device="cpu",
            load_ms=0,
            batch_sizes=(0,),
            stage_repeats=1,
            warmup=0,
            repeats=1,
            min_seconds=0,
            vram_repeats=1,
        )


def test_report_helpers(tmp_path):
    assert bytes_text(None) == "n/a"
    assert bytes_text(0) == "0 B"
    assert bytes_text(1023) == "1023 B"
    assert bytes_text(1024) == "1.00 KiB"
    assert bytes_text(1024**2) == "1.00 MiB"
    assert bytes_text(1024**3) == "1.00 GiB"
    assert bytes_text(1024**4) == "1.00 TiB"
    assert bytes_text(1024**5) == "1024.00 TiB"
    assert bytes_text(-2048) == "-2.00 KiB"
    assert as_jsonable("ok") == "ok"
    blob = {"state_dict": {"w": torch.ones(3, 4)}, "other": 1, "seq": [torch.zeros(2)]}
    parts = checkpoint_breakdown(blob)
    assert parts["state_dict_bytes"] == 3 * 4 * 4
    assert parts["ema_shadow_bytes"] == 0
    assert serving_bytes(parts, "ema") == parts["state_dict_bytes"]
    assert serving_bytes(parts, "raw") == parts["state_dict_bytes"]
    empty = checkpoint_breakdown([1, 2])
    assert empty["keys"] == [] and nested_tensor_bytes("x") == 0
    stats = summarize_ms([1.0, 2.0, 3.0])
    assert stats["p50_ms"] == 2.0
    encoded = as_jsonable(
        {
            "path": tmp_path,
            "arr": np.array([1.5], dtype=np.float32),
            "flag": np.bool_(True),
            "n": np.int64(3),
            "raw": b"ab",
            "tup": (1, 2),
        }
    )
    assert encoded["path"] == str(tmp_path)
    assert encoded["raw"] == 2 and encoded["flag"] is True
    summary = contest_summary(
        inventory={"total_bytes": 10, "within_limit": True},
        latency={"latency_b1_ms": 1.5},
        throughput={"best_fps": 8.0, "best_batch_size": 16},
        vram=[],
        load_ms=3.0,
        load_snapshot={"peak_bytes": 9},
        determinism={"bit_identical": False, "allclose_fp32": True},
        serving_weight_bytes=4,
    )
    assert summary["peak_vram_bytes"] == 9
    assert summary["deterministic"] and summary["within_2gib"]
    assert summary["performance_eligible"]
    assert summary["latency_score"] == 1.0
    assert summary["throughput_score"] == 0.0
    assert summary["performance_score"] == pytest.approx(0.10)
    assert latency_score(40) == 1.0 and latency_score(80) == 0.0
    assert latency_score(60) == pytest.approx(0.5)
    assert throughput_score(100) == 1.0 and throughput_score(50) == 0.0
    assert throughput_score(75) == pytest.approx(0.5)
    blocked = performance_scores(20.0, 200.0, within_limit=False, n_test=100)
    assert blocked["performance_eligible"] is False
    assert blocked["performance_score"] is None
    assert blocked["time_budget_s"] == pytest.approx(6.0)
    full = performance_scores(31.2, 120.0, within_limit=True, n_test=10)
    assert full["latency_score"] == 1.0 and full["throughput_score"] == 1.0
    assert full["performance_score"] == pytest.approx(0.20)
    heavy = contest_summary(
        inventory={"total_bytes": 9, "within_limit": False},
        latency={"latency_b1_ms": 20.0},
        throughput={"best_fps": 200.0, "best_batch_size": 32},
        vram=[],
        load_ms=1.0,
        load_snapshot={"peak_bytes": 0},
        determinism={"bit_identical": True, "allclose_fp32": True},
        serving_weight_bytes=1,
        n_test=50,
    )
    assert heavy["performance_eligible"] is False
    assert heavy["time_budget_s"] == pytest.approx(3.0)
    assert summary["time_budget_s"] is None
    peaky = contest_summary(
        inventory={"total_bytes": 10, "within_limit": True},
        latency={"latency_b1_ms": 1.5},
        throughput={"best_fps": 8.0, "best_batch_size": 16},
        vram=[{"peak_bytes": 5}, {"peak_bytes": 12}],
        load_ms=3.0,
        load_snapshot={"peak_bytes": None},
        determinism={"bit_identical": True, "allclose_fp32": False},
        serving_weight_bytes=4,
    )
    assert peaky["peak_vram_bytes"] == 12
