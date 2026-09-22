from time import perf_counter
from typing import Any, cast

from torch.utils.data import DataLoader, Dataset

from dataset.images import VehicleDataset, decode_backend_name
from modules.inference import (
    embed_frame,
    embed_tensor,
    fuse_context_views,
    inference_loader_kwargs,
    records_frame,
    tta_context_pcts,
)

from .device import memory_snapshot, reset_peak, timed
from .extract import extract
from .protocol import (
    CONTEST_BATCH_SIZES,
    CONTEST_MIN_SECONDS,
    CONTEST_REPEATS,
    CONTEST_WARMUP,
    compare_embeddings,
    measure_latency,
    measure_stages,
    measure_throughput,
    measure_vram,
)
from .report import checkpoint_breakdown, contest_summary, serving_bytes
from .weights import inventory

STREAM_REPEATS = 100_000


def _slice_flags(full_images, size: int):
    if full_images is None:
        return None
    return list(full_images)[:size]


def make_runner(paths, bboxes, transform, model, device, **kwargs):
    full_images = kwargs.pop("full_images", None)

    def run(size: int, *, timed: bool = False):
        return extract(
            list(paths)[:size],
            list(bboxes)[:size],
            transform,
            model,
            device,
            full_images=_slice_flags(full_images, size),
            timed=timed,
            **kwargs,
        )

    return run


class RepeatDataset(Dataset):
    def __init__(self, dataset, repeats: int = STREAM_REPEATS):
        self.dataset = dataset
        self.repeats = int(repeats)
        if self.repeats < 1 or len(dataset) < 1:
            raise ValueError("repeat dataset needs a nonempty source and repeats >= 1")

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, index):
        return self.dataset[int(index) % len(self.dataset)]


class BatchStream:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = None
        self.cold_s = None

    def next_batch(self):
        if self.iterator is None:
            started = perf_counter()
            self.iterator = iter(cast(Any, self.loader))
            batch = next(self.iterator)
            self.cold_s = perf_counter() - started
            return batch
        return next(self.iterator)

    def close(self) -> None:
        iterator = self.iterator
        self.iterator = None
        self.loader = None
        if iterator is None:
            return
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()


def make_loader_runner(paths, bboxes, transform, model, device, cfg, *, full_images=None):
    frame = records_frame(list(paths), list(bboxes), full_images=full_images)
    contexts = tta_context_pcts(cfg.eval.tta, cfg.data.context_pct)
    streams = {}
    finished = []
    active_size = None

    def stream_for(size: int, view: int, context: float) -> BatchStream:
        key = (int(size), int(view))
        if key not in streams:
            base = VehicleDataset(frame, cfg, transform, context_pct=float(context))
            kwargs = inference_loader_kwargs(cfg, device)
            loader = DataLoader(RepeatDataset(base), batch_size=int(size), **kwargs)
            streams[key] = BatchStream(loader)
        return streams[key]

    def remember(size: int, view: int, context: float, stream: BatchStream) -> None:
        if stream.cold_s is None:
            return
        finished.append(
            {
                "batch_size": int(size),
                "view": int(view),
                "context_pct": float(context),
                "seconds": float(stream.cold_s),
            }
        )

    class Runner:
        def _drop(self) -> None:
            nonlocal active_size
            for (size, view), stream in list(streams.items()):
                remember(size, view, contexts[view], stream)
                stream.close()
            streams.clear()
            active_size = None

        def __call__(self, size: int):
            nonlocal active_size
            if active_size != int(size):
                self._drop()
                active_size = int(size)
            views = []
            for view, context in enumerate(contexts):
                batch = stream_for(size, view, context).next_batch()
                image = batch["image"].to(device, non_blocking=True)
                views.append(
                    embed_tensor(
                        model,
                        image,
                        device,
                        precision=str(cfg.eval.precision),
                        tta=cfg.eval.tta,
                        model_cfg=cfg.model,
                    )
                    .cpu()
                    .numpy()
                )
            return fuse_context_views(views)

        def cold_start_s(self):
            rows = list(finished)
            for (size, view), stream in streams.items():
                if stream.cold_s is None:
                    continue
                rows.append(
                    {
                        "batch_size": int(size),
                        "view": int(view),
                        "context_pct": float(contexts[view]),
                        "seconds": cast(float, stream.cold_s),
                    }
                )
            return rows

        def release(self) -> None:
            self._drop()

    return Runner()


def release_runner(runner) -> None:
    release = getattr(runner, "release", None)
    if callable(release):
        release()


def profile_extract(
    *,
    roots,
    paths,
    bboxes,
    transform,
    model,
    device,
    load_ms: float,
    checkpoint=None,
    weights: str = "raw",
    context_pct: float = 0.0,
    full_images=None,
    precision: str = "fp32",
    tta=None,
    model_cfg=None,
    decode_backend: str = "pil",
    decode_workers: int = 0,
    warmup: int = CONTEST_WARMUP,
    repeats: int = CONTEST_REPEATS,
    batch_sizes=CONTEST_BATCH_SIZES,
    min_seconds: float = CONTEST_MIN_SECONDS,
    vram_repeats: int = 3,
    stage_repeats: int = 8,
    n_test=None,
    cfg=None,
):
    sizes = [int(size) for size in batch_sizes]
    if not sizes or min(sizes) < 1:
        raise ValueError("batch_sizes must be nonempty positive integers")
    if len(paths) < max(sizes):
        raise ValueError("need at least max(batch_sizes) records")
    if stage_repeats < 1:
        raise ValueError("stage_repeats must be >= 1")
    if cfg is not None:
        decode_backend = decode_backend_name(cfg, default=decode_backend)
    kwargs = {
        "context_pct": context_pct,
        "precision": precision,
        "tta": tta,
        "model_cfg": model_cfg,
        "decode_backend": decode_backend,
        "decode_workers": decode_workers,
    }
    run = make_runner(paths, bboxes, transform, model, device, full_images=full_images, **kwargs)
    serving = (
        make_loader_runner(paths, bboxes, transform, model, device, cfg, full_images=full_images)
        if cfg is not None
        else run
    )
    weights_info = inventory(roots)
    breakdown = checkpoint_breakdown(checkpoint) if checkpoint is not None else None
    payload = serving_bytes(breakdown, weights) if breakdown is not None else 0
    load_snapshot = memory_snapshot(device)

    def run_one():
        run(1)

    def run_timed():
        return run(1, timed=True)

    def run_batch(size: int):
        return serving(size)

    stage_mean = measure_stages(
        run_one,
        run_timed,
        device,
        warmup=warmup,
        repeats=stage_repeats,
    )
    reset_peak(device)
    latency = measure_latency(run_one, device, warmup=warmup, repeats=repeats)
    try:
        throughput = measure_throughput(run_batch, device, batch_sizes=sizes, min_seconds=min_seconds)
        cold_start = getattr(serving, "cold_start_s", None)
        if callable(cold_start):
            throughput["cold_start_s"] = cold_start()
    finally:
        release_runner(serving)
    try:
        vram = measure_vram(run_batch, device, batch_sizes=sizes, repeats=vram_repeats)
    finally:
        release_runner(serving)
    sample = min(8, len(paths))
    if cfg is None:
        first = run(sample)
        second = run(sample)
    else:
        frame = records_frame(
            list(paths)[:sample],
            list(bboxes)[:sample],
            full_images=_slice_flags(full_images, sample),
        )
        first = embed_frame(model, cfg, device, frame)
        second = embed_frame(model, cfg, device, frame)
    determinism = compare_embeddings(first, second)
    summary = contest_summary(
        inventory=weights_info,
        latency=latency,
        throughput=throughput,
        vram=vram,
        load_ms=load_ms,
        load_snapshot=load_snapshot,
        determinism=determinism,
        serving_weight_bytes=payload,
        n_test=n_test,
    )
    return {
        "weights": weights_info,
        "checkpoint": breakdown,
        "serving_choice": weights,
        "load_ms": float(load_ms),
        "load_snapshot": load_snapshot,
        "stages_per_image_ms": stage_mean,
        "latency": latency,
        "throughput": throughput,
        "vram": vram,
        "determinism": determinism,
        "contest": summary,
    }


def time_load(load_fn, device):
    reset_peak(device)
    model, elapsed = timed(load_fn, device)
    return model, elapsed, memory_snapshot(device)
