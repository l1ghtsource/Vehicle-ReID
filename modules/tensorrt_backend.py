import hashlib
import importlib
import json
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch

trt: Any = importlib.import_module("tensorrt") if find_spec("tensorrt") is not None else None


def require_tensorrt() -> Any:
    if trt is None:
        raise RuntimeError("TensorRT is optional; install requirements/tensorrt.txt on a CUDA host")
    return trt


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engine_manifest_path(engine_path: str | Path) -> Path:
    return Path(f"{engine_path}.json")


def read_engine_manifest(engine_path: str | Path, checkpoint_path: str | Path) -> dict:
    path = engine_manifest_path(engine_path)
    if not path.is_file():
        raise FileNotFoundError(f"TensorRT engine manifest is missing: {path}")
    manifest = json.loads(path.read_text())
    expected = manifest.get("checkpoint_sha256")
    if not expected or expected != sha256_file(checkpoint_path):
        raise ValueError("TensorRT engine was built from different checkpoint bytes")
    return manifest


class TensorRTEmbedding:
    def __init__(self, engine_path: str | Path, checkpoint_path: str | Path, device: torch.device):
        api = require_tensorrt()
        if device.type != "cuda":
            raise ValueError("TensorRT inference requires a CUDA device")
        self.manifest = read_engine_manifest(engine_path, checkpoint_path)
        device_index = torch.cuda.current_device() if device.index is None else device.index
        self.device = torch.device("cuda", device_index)
        self.api = api
        if str(self.manifest["tensorrt_version"]) != str(api.__version__):
            raise ValueError("TensorRT engine and runtime versions differ")
        if tuple(self.manifest["compute_capability"]) != torch.cuda.get_device_capability(self.device):
            raise ValueError("TensorRT engine was built for a different GPU compute capability")
        if str(self.manifest["gpu"]) != torch.cuda.get_device_name(self.device):
            raise ValueError("TensorRT engine was built for a different GPU model")
        self.logger = api.Logger(api.Logger.WARNING)
        self.runtime = api.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT could not deserialize the engine on this GPU/runtime")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context")
        self.stream = torch.cuda.Stream(device=self.device)
        self.input_name = str(self.manifest["input_name"])
        self.output_name = str(self.manifest["output_name"])
        if self.engine.get_tensor_dtype(self.input_name) != api.DataType.FLOAT:
            raise ValueError("TensorRT engine input must be float32")
        if self.engine.get_tensor_dtype(self.output_name) != api.DataType.FLOAT:
            raise ValueError("TensorRT engine output must be float32")

    def eval(self):
        return self

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        shape = tuple(images.shape)
        expected_hw = tuple(int(value) for value in self.manifest["image_size"])
        batch = int(shape[0]) if shape else 0
        if (
            images.device.type != "cuda"
            or images.device.index != self.device.index
            or images.dtype != torch.float32
            or len(shape) != 4
            or shape[1:] != (3, *expected_hw)
            or not int(self.manifest["min_batch"]) <= batch <= int(self.manifest["max_batch"])
        ):
            raise ValueError("TensorRT input must match the configured CUDA device, shape, dtype, and batch")
        inp = images.contiguous()
        if not self.context.set_input_shape(self.input_name, shape):
            raise RuntimeError(f"TensorRT rejected input shape {shape}")
        output_shape = tuple(int(value) for value in self.context.get_tensor_shape(self.output_name))
        if any(value < 1 for value in output_shape):
            raise RuntimeError(f"TensorRT returned unresolved output shape {output_shape}")
        output = torch.empty(output_shape, device=self.device, dtype=torch.float32)
        if not self.context.set_tensor_address(self.input_name, inp.data_ptr()):
            raise RuntimeError("TensorRT rejected the input buffer")
        if not self.context.set_tensor_address(self.output_name, output.data_ptr()):
            raise RuntimeError("TensorRT rejected the output buffer")
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        inp.record_stream(self.stream)
        output.record_stream(self.stream)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT inference failed")
        current.wait_stream(self.stream)
        return output
