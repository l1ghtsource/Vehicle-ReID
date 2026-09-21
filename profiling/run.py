from torch.utils.data import DataLoader

from dataset.images import VehicleDataset
from modules.inference import embed_loader, inference_loader_kwargs, records_frame

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


def make_loader_runner(paths, bboxes, transform, model, device, cfg, *, full_images=None, context_pct=0.0):
    loaders = {}

    def run_batch(size: int):
        if size not in loaders:
            frame = records_frame(
                list(paths)[:size],
                list(bboxes)[:size],
                full_images=_slice_flags(full_images, size),
            )
            kwargs = inference_loader_kwargs(cfg, device)
            if kwargs["num_workers"] > 0:
                kwargs["persistent_workers"] = True
            loaders[size] = DataLoader(
                VehicleDataset(frame.reset_index(drop=True), cfg, transform, context_pct=float(context_pct)),
                batch_size=size,
                **kwargs,
            )
        return embed_loader(model, loaders[size], cfg, device)

    return run_batch


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
        decode_backend = str(cfg.data.get("decode_backend", decode_backend))
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
        make_loader_runner(
            paths,
            bboxes,
            transform,
            model,
            device,
            cfg,
            full_images=full_images,
            context_pct=context_pct,
        )
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
        serving(size)

    stage_mean = measure_stages(
        run_one,
        run_timed,
        device,
        warmup=warmup,
        repeats=stage_repeats,
    )
    reset_peak(device)
    latency = measure_latency(run_one, device, warmup=warmup, repeats=repeats)
    throughput = measure_throughput(run_batch, device, batch_sizes=sizes, min_seconds=min_seconds)
    vram = measure_vram(run_batch, device, batch_sizes=sizes, repeats=vram_repeats)
    first = serving(min(8, len(paths)))
    second = serving(min(8, len(paths)))
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
