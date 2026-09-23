"""Export a serving checkpoint to ONNX and build an optional TensorRT engine."""

import gc
import json
import time
from pathlib import Path

import hydra
import torch

from eval import load_model
from models.kernels import EmbeddingForward
from modules.tensorrt_backend import engine_manifest_path, require_tensorrt, sha256_file


def export_onnx(model, image_size: tuple[int, int], path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, 3, *image_size), dtype=torch.float32)
    started = time.perf_counter()
    torch.onnx.export(
        EmbeddingForward(model.cpu().eval()),
        (example,),
        str(path),
        input_names=["images"],
        output_names=["embedding"],
        opset_version=18,
        dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
        dynamo=False,
    )
    return time.perf_counter() - started


def build_engine(onnx_path: Path, engine_path: Path, image_size: tuple[int, int], spec) -> float:
    api = require_tensorrt()
    minimum, optimum, maximum = (int(spec.min_batch), int(spec.opt_batch), int(spec.max_batch))
    if not 1 <= minimum <= optimum <= maximum:
        raise ValueError("TensorRT batch profile must satisfy 1 <= min <= opt <= max")
    precision = str(spec.precision)
    if precision not in {"fp16", "fp32"}:
        raise ValueError("TensorRT precision must be fp16/fp32")
    logger = api.Logger(api.Logger.WARNING)
    builder = api.Builder(logger)
    network = builder.create_network(0)
    parser = api.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise ValueError(f"TensorRT ONNX parser failed:\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(api.MemoryPoolType.WORKSPACE, int(float(spec.workspace_gb) * (1 << 30)))
    if precision == "fp16":
        if not hasattr(api.BuilderFlag, "FP16"):
            raise RuntimeError("FP16 builder flag is unavailable; install pinned TensorRT 10.x")
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This GPU does not support fast FP16 TensorRT kernels")
        config.set_flag(api.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images", (minimum, 3, *image_size), (optimum, 3, *image_size), (maximum, 3, *image_size)
    )
    if profile.get_shape("images") != [
        (minimum, 3, *image_size),
        (optimum, 3, *image_size),
        (maximum, 3, *image_size),
    ]:
        raise ValueError("TensorRT rejected the batch optimization profile")
    config.add_optimization_profile(profile)
    started = time.perf_counter()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT failed to build the engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(engine)
    return time.perf_counter() - started


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg) -> None:
    if not cfg.checkpoint:
        raise ValueError("Pass checkpoint=/path/to/weights.pt")
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT export requires a CUDA GPU")
    model, effective, checkpoint, choice = load_model(cfg)
    spec = effective.eval.tensorrt
    onnx_path = Path(str(spec.onnx_path))
    engine_path = Path(str(spec.engine_path))
    image_size = tuple(int(value) for value in effective.data.image_size)
    if len(image_size) != 2:
        raise ValueError("TensorRT export requires a fixed height and width")
    export_s = export_onnx(model, image_size, onnx_path)
    del model, checkpoint
    gc.collect()
    build_s = build_engine(onnx_path, engine_path, image_size, spec)
    api = require_tensorrt()
    manifest = {
        "checkpoint_sha256": sha256_file(cfg.checkpoint),
        "onnx_sha256": sha256_file(onnx_path),
        "input_name": "images",
        "output_name": "embedding",
        "image_size": image_size,
        "min_batch": int(spec.min_batch),
        "opt_batch": int(spec.opt_batch),
        "max_batch": int(spec.max_batch),
        "precision": str(spec.precision),
        "weights": choice,
        "tensorrt_version": str(api.__version__),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "onnx_bytes": onnx_path.stat().st_size,
        "engine_bytes": engine_path.stat().st_size,
        "export_s": export_s,
        "build_s": build_s,
    }
    engine_manifest_path(engine_path).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
