"""CPU-only contract tests for the optional TensorRT export and runtime."""

import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import hydra
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import eval as eval_module
import modules.tensorrt_backend as backend
import scripts.export_tensorrt as exporter


def test_manifest_checksum_and_optional_dependency(tmp_path, monkeypatch):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    engine = tmp_path / "model.engine"
    assert backend.sha256_file(weights) == hashlib.sha256(b"weights").hexdigest()
    assert backend.engine_manifest_path(engine) == tmp_path / "model.engine.json"
    monkeypatch.setattr(backend, "trt", None)
    with pytest.raises(RuntimeError, match="optional"):
        backend.require_tensorrt()
    with pytest.raises(FileNotFoundError, match="manifest"):
        backend.read_engine_manifest(engine, weights)
    manifest = {"checkpoint_sha256": "wrong"}
    backend.engine_manifest_path(engine).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different checkpoint"):
        backend.read_engine_manifest(engine, weights)
    manifest["checkpoint_sha256"] = backend.sha256_file(weights)
    backend.engine_manifest_path(engine).write_text(json.dumps(manifest))
    assert backend.read_engine_manifest(engine, weights) == manifest
    monkeypatch.setattr(backend, "trt", object())
    assert backend.require_tensorrt() is backend.trt


def fake_runtime(monkeypatch, tmp_path):
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")
    manifest = {
        "tensorrt_version": "10.16",
        "compute_capability": [8, 6],
        "gpu": "fake GPU",
        "input_name": "images",
        "output_name": "embedding",
        "image_size": [336, 336],
        "min_batch": 1,
        "max_batch": 32,
    }
    monkeypatch.setattr(backend, "read_engine_manifest", lambda *args: manifest)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 6))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "fake GPU")
    stream = MagicMock(cuda_stream=123)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_: stream)
    current = MagicMock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: current)
    context = MagicMock()
    context.set_input_shape.return_value = True
    context.get_tensor_shape.return_value = (8, 256)
    context.set_tensor_address.return_value = True
    context.execute_async_v3.return_value = True
    engine = MagicMock()
    engine.create_execution_context.return_value = context
    engine.get_tensor_dtype.return_value = 1
    runtime = MagicMock()
    runtime.deserialize_cuda_engine.return_value = engine
    api = SimpleNamespace(
        __version__="10.16",
        Logger=MagicMock(WARNING=1),
        Runtime=lambda _: runtime,
        DataType=SimpleNamespace(FLOAT=1),
    )
    monkeypatch.setattr(backend, "trt", api)
    return engine_path, manifest, api, runtime, engine, context, stream, current


def test_tensorrt_runtime_checks_and_execution(tmp_path, monkeypatch):
    path, manifest, api, runtime, engine, context, stream, current = fake_runtime(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="CUDA"):
        backend.TensorRTEmbedding(path, "weights.pt", torch.device("cpu"))
    manifest["tensorrt_version"] = "old"
    with pytest.raises(ValueError, match="versions"):
        backend.TensorRTEmbedding(path, "weights.pt", torch.device("cuda"))
    manifest["tensorrt_version"] = api.__version__
    manifest["compute_capability"] = [9, 0]
    with pytest.raises(ValueError, match="compute capability"):
        backend.TensorRTEmbedding(path, "weights.pt", torch.device("cuda"))
    manifest["compute_capability"] = [8, 6]
    manifest["gpu"] = "other GPU"
    with pytest.raises(ValueError, match="GPU model"):
        backend.TensorRTEmbedding(path, "weights.pt", torch.device("cuda"))
    manifest["gpu"] = "fake GPU"
    runtime.deserialize_cuda_engine.return_value = None
    with pytest.raises(RuntimeError, match="deserialize"):
        backend.TensorRTEmbedding(path, "weights.pt", torch.device("cuda"))
    runtime.deserialize_cuda_engine.return_value = engine
    engine.create_execution_context.return_value = None
    with pytest.raises(RuntimeError, match="execution context"):
        backend.TensorRTEmbedding(path, "weights.pt", torch.device("cuda"))
    engine.create_execution_context.return_value = context
    engine.get_tensor_dtype.return_value = 2
    with pytest.raises(ValueError, match="input must be float32"):
        backend.TensorRTEmbedding(path, "weights.pt", torch.device("cuda"))
    engine.get_tensor_dtype.side_effect = [1, 2]
    with pytest.raises(ValueError, match="output must be float32"):
        backend.TensorRTEmbedding(path, "weights.pt", torch.device("cuda"))
    engine.get_tensor_dtype.side_effect = None
    engine.get_tensor_dtype.return_value = 1
    model = backend.TensorRTEmbedding(path, "weights.pt", torch.device("cuda"))
    assert model.eval() is model
    image = MagicMock()
    image.shape = (8, 3, 336, 336)
    image.device = torch.device("cuda:0")
    image.dtype = torch.float32
    image.contiguous.return_value = image
    image.data_ptr.return_value = 111
    output = MagicMock()
    output.data_ptr.return_value = 222
    monkeypatch.setattr(backend.torch, "empty", lambda *args, **kwargs: output)
    assert model(image) is output
    context.set_input_shape.assert_called_with("images", image.shape)
    context.execute_async_v3.assert_called_with(123)
    stream.wait_stream.assert_called_with(current)
    current.wait_stream.assert_called_with(stream)
    for attr, value in (("dtype", torch.float16), ("shape", (8, 3, 320, 320))):
        old = getattr(image, attr)
        setattr(image, attr, value)
        with pytest.raises(ValueError, match="input must match"):
            model(image)
        setattr(image, attr, old)
    image.shape = (33, 3, 336, 336)
    with pytest.raises(ValueError, match="input must match"):
        model(image)
    image.shape = (8, 3, 336, 336)
    context.set_input_shape.return_value = False
    with pytest.raises(RuntimeError, match="rejected input shape"):
        model(image)
    context.set_input_shape.return_value = True
    context.get_tensor_shape.return_value = (-1, 256)
    with pytest.raises(RuntimeError, match="unresolved output"):
        model(image)
    context.get_tensor_shape.return_value = (8, 256)
    context.set_tensor_address.side_effect = [False]
    with pytest.raises(RuntimeError, match="input buffer"):
        model(image)
    context.set_tensor_address.side_effect = [True, False]
    with pytest.raises(RuntimeError, match="output buffer"):
        model(image)
    context.set_tensor_address.side_effect = None
    context.set_tensor_address.return_value = True
    context.execute_async_v3.return_value = False
    with pytest.raises(RuntimeError, match="inference failed"):
        model(image)


def fake_builder():
    parser = MagicMock()
    parser.parse_from_file.return_value = True
    parser.num_errors = 1
    parser.get_error.return_value = "broken graph"
    profile = MagicMock()
    profile.get_shape.return_value = [(1, 3, 336, 336), (16, 3, 336, 336), (32, 3, 336, 336)]
    builder = MagicMock()
    builder.platform_has_fast_fp16 = True
    builder.create_optimization_profile.return_value = profile
    builder.build_serialized_network.return_value = b"serialized"
    api = SimpleNamespace(
        Logger=MagicMock(WARNING=1),
        Builder=lambda _: builder,
        OnnxParser=lambda *args: parser,
        MemoryPoolType=SimpleNamespace(WORKSPACE=1),
        BuilderFlag=SimpleNamespace(FP16=2),
    )
    return api, builder, parser, profile


def test_export_and_build_contracts(tmp_path, monkeypatch):
    model = nn.Identity()
    recorded = []
    monkeypatch.setattr(
        exporter.torch.onnx, "export", lambda *args, **kwargs: recorded.append((args, kwargs))
    )
    assert exporter.export_onnx(model, (336, 336), tmp_path / "new/model.onnx") >= 0
    assert recorded[0][0][1][0].shape == (1, 3, 336, 336)
    assert recorded[0][1]["dynamo"] is False
    api, builder, parser, profile = fake_builder()
    monkeypatch.setattr(exporter, "require_tensorrt", lambda: api)
    spec = SimpleNamespace(min_batch=1, opt_batch=16, max_batch=32, precision="fp16", workspace_gb=1)
    onnx_path = tmp_path / "new/model.onnx"
    engine_path = tmp_path / "eng/model.engine"
    assert exporter.build_engine(onnx_path, engine_path, (336, 336), spec) >= 0
    assert engine_path.read_bytes() == b"serialized"
    builder.create_builder_config.return_value.set_flag.assert_called_with(2)
    spec.min_batch = 0
    with pytest.raises(ValueError, match="batch profile"):
        exporter.build_engine(onnx_path, engine_path, (336, 336), spec)
    spec.min_batch = 1
    spec.precision = "int8"
    with pytest.raises(ValueError, match="precision"):
        exporter.build_engine(onnx_path, engine_path, (336, 336), spec)
    spec.precision = "fp16"
    parser.parse_from_file.return_value = False
    with pytest.raises(ValueError, match="broken graph"):
        exporter.build_engine(onnx_path, engine_path, (336, 336), spec)
    parser.parse_from_file.return_value = True
    api.BuilderFlag = SimpleNamespace()
    with pytest.raises(RuntimeError, match="FP16 builder flag"):
        exporter.build_engine(onnx_path, engine_path, (336, 336), spec)
    api.BuilderFlag = SimpleNamespace(FP16=2)
    builder.platform_has_fast_fp16 = False
    with pytest.raises(RuntimeError, match="fast FP16"):
        exporter.build_engine(onnx_path, engine_path, (336, 336), spec)
    builder.platform_has_fast_fp16 = True
    profile.get_shape.return_value = []
    with pytest.raises(ValueError, match="batch optimization"):
        exporter.build_engine(onnx_path, engine_path, (336, 336), spec)
    profile.get_shape.return_value = [(1, 3, 336, 336), (16, 3, 336, 336), (32, 3, 336, 336)]
    builder.build_serialized_network.return_value = None
    with pytest.raises(RuntimeError, match="failed to build"):
        exporter.build_engine(onnx_path, engine_path, (336, 336), spec)
    spec.precision = "fp32"
    builder.build_serialized_network.return_value = b"fp32"
    assert exporter.build_engine(onnx_path, engine_path, (336, 336), spec) >= 0
    assert engine_path.read_bytes() == b"fp32"


def test_export_entrypoint(tmp_path, monkeypatch):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"checkpoint")
    cfg = OmegaConf.create(
        {
            "checkpoint": str(weights),
            "data": {"image_size": [336, 336]},
            "eval": {
                "tensorrt": {
                    "onnx_path": str(tmp_path / "model.onnx"),
                    "engine_path": str(tmp_path / "model.engine"),
                    "precision": "fp16",
                    "min_batch": 1,
                    "opt_batch": 16,
                    "max_batch": 32,
                }
            },
        }
    )
    monkeypatch.setattr(exporter.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(exporter.torch.cuda, "get_device_name", lambda: "fake GPU")
    monkeypatch.setattr(exporter.torch.cuda, "get_device_capability", lambda: (8, 6))
    monkeypatch.setattr(exporter, "load_model", lambda *_: (nn.Identity(), cfg, {}, "ema"))
    monkeypatch.setattr(exporter, "export_onnx", lambda *args: (Path(args[2]).write_bytes(b"onnx"), 1.0)[1])
    monkeypatch.setattr(
        exporter, "build_engine", lambda *args: (Path(args[1]).write_bytes(b"engine"), 2.0)[1]
    )
    monkeypatch.setattr(exporter, "require_tensorrt", lambda: SimpleNamespace(__version__="10.16"))
    exporter.main.__wrapped__(cfg)
    result = json.loads((tmp_path / "model.engine.json").read_text())
    assert result["checkpoint_sha256"] == backend.sha256_file(weights)
    assert result["weights"] == "ema"
    assert result["image_size"] == [336, 336]
    cfg.checkpoint = None
    with pytest.raises(ValueError, match="checkpoint"):
        exporter.main.__wrapped__(cfg)
    cfg.checkpoint = str(weights)
    monkeypatch.setattr(exporter.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        exporter.main.__wrapped__(cfg)
    monkeypatch.setattr(exporter.torch.cuda, "is_available", lambda: True)
    cfg.data.image_size = [336]
    with pytest.raises(ValueError, match="fixed height"):
        exporter.main.__wrapped__(cfg)
    monkeypatch.setattr(sys, "argv", ["export_tensorrt"])
    called = []
    monkeypatch.setattr(hydra, "main", lambda **kwargs: lambda _: lambda: called.append(True))
    runpy.run_module("scripts.export_tensorrt", run_name="__main__")
    assert called == [True]


def test_eval_backend_overlay_and_runtime(cfg, data_cfg, tmp_path, monkeypatch):
    cfg.eval.inference_backend = "tensorrt"
    cfg.eval.tensorrt.engine_path = str(tmp_path / "model.engine")
    saved = OmegaConf.create({"eval": {"split": "val"}, "data": {"fold": 0}})
    effective = eval_module.overlay_eval_config(saved, cfg, [])
    assert effective.eval.inference_backend == "tensorrt"
    assert effective.eval.tensorrt.engine_path == cfg.eval.tensorrt.engine_path
    class StubModel:
        manifest = {"max_batch": 2, "weights": "ema", "image_size": list(data_cfg.data.image_size)}

        def eval(self):
            return self

    recorded = []
    monkeypatch.setattr(
        eval_module, "load_model", lambda *_: (nn.Identity(), data_cfg, {"state_dict": {}}, "ema")
    )
    monkeypatch.setattr(
        eval_module, "TensorRTEmbedding", lambda *args: (recorded.append(args), StubModel())[1]
    )
    monkeypatch.setattr(eval_module, "configure_runtime", lambda *_: None)
    monkeypatch.setattr(eval_module.L, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        eval_module, "embed_frame", lambda _m, _c, _d, frame: torch.ones(len(frame), 8).numpy()
    )
    data_cfg.eval.inference_backend = "tensorrt"
    data_cfg.eval.device = "cuda:0"
    data_cfg.eval.split = "test"
    data_cfg.eval.output_dir = str(tmp_path / "out")
    data_cfg.eval.tensorrt.engine_path = str(tmp_path / "model.engine")
    data_cfg.eval.tta.enabled = True
    data_cfg.eval.tta.scales = [1.1]
    with pytest.raises(ValueError, match="fixed input size"):
        eval_module.main.__wrapped__(data_cfg)
    data_cfg.eval.tta.scales = [1.0]
    data_cfg.data.batch_size_eval = 64
    data_cfg.refusal.kind = "none"
    eval_module.main.__wrapped__(data_cfg)
    assert data_cfg.data.batch_size_eval == 2
    assert recorded[0][0] == data_cfg.eval.tensorrt.engine_path
    StubModel.manifest["weights"] = "raw"
    with pytest.raises(ValueError, match="weights differ"):
        eval_module.main.__wrapped__(data_cfg)
    StubModel.manifest["weights"] = "ema"
    StubModel.manifest["image_size"] = [224, 224]
    with pytest.raises(ValueError, match="image_size differs"):
        eval_module.main.__wrapped__(data_cfg)
    data_cfg.eval.inference_backend = "invalid"
    with pytest.raises(ValueError, match="torch/tensorrt"):
        eval_module.main.__wrapped__(data_cfg)
