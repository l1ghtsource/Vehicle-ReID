import argparse
import json
from pathlib import Path

import torch

from eval import build_serving_payload
from scripts.verify_weights import write_sha256sums

DEFAULT_SUMMARY = Path("runs/full/eva02_k4_cam/run_summary.json")
DEFAULT_METRICS = Path("runs/cv/eva02_k4_cam/fold0/val/metrics.json")
DEFAULT_OUTPUT = Path("weights/finetuned/eva02.pt")


def listed_checkpoint(payload: dict) -> Path:
    for key in ("last_checkpoint", "best_checkpoint", "checkpoint"):
        value = payload.get(key)
        if value:
            return Path(str(value))
    raise ValueError("No checkpoint path in default metadata")


def source_checkpoint(path: Path | None) -> Path:
    if path is not None:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        return source
    for default in (DEFAULT_SUMMARY, DEFAULT_METRICS):
        if not default.is_file():
            continue
        source = listed_checkpoint(json.loads(default.read_text()))
        if not source.is_file():
            raise FileNotFoundError(source)
        return source
    raise ValueError("Pass --checkpoint; no full-retrain run_summary.json or fold-0 metrics.json was found")


def export_serving(source: Path, destination: Path, weights: str = "ema") -> Path:
    blob = torch.load(Path(source), map_location="cpu", weights_only=False)
    payload = build_serving_payload(blob, weights)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weights", choices=("auto", "raw", "ema"), default="ema")
    parser.add_argument("--sha256", action="store_true")
    args = parser.parse_args()
    source = source_checkpoint(args.checkpoint)
    target = export_serving(source, args.output, args.weights)
    print(f"{source} -> {target} ({target.stat().st_size} bytes)")
    if args.sha256:
        written = write_sha256sums(target.parent)
        print(written)


if __name__ == "__main__":
    main()
