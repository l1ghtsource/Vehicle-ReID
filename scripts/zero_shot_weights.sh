#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <cuda_device> [hydra overrides...]" >&2
  exit 1
fi

DEVICE="$1"
shift
PYTHON="${PYTHON:-.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-artifacts/zero_shot}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

run_one() {
  local model="$1"
  local checkpoint="$2"
  shift 2
  mkdir -p "$OUT_ROOT/$model"
  echo "zero-shot $model on $DEVICE"
  "$PYTHON" scripts/zero_shot.py "$model" \
    "model.checkpoint_path=$checkpoint" \
    "model.local_files_only=true" \
    "eval.device=$DEVICE" \
    "eval.output_dir=$OUT_ROOT/$model" \
    "$@"
}

run_one dinov3_convnext_base weights/dinov3_base/model.safetensors "$@"
run_one dinov3_convnext_large weights/dinov3_large/model.safetensors "$@"
run_one radio weights/radio/model.safetensors "$@"
run_one llm2clip weights/llm2clip/LLM2CLIP-EVA02-L-14-336.pt \
  "data.image_size=[336,336]" \
  "model.pooling.kind=cls" \
  "$@"

"$PYTHON" - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
models = ["dinov3_convnext_base", "dinov3_convnext_large", "radio", "llm2clip"]
summary = []
for name in models:
    path = root / name / "metrics.json"
    summary.append(json.loads(path.read_text()))
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(f"summary: {root / 'summary.json'}")
PY
