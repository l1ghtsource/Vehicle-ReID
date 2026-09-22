#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  echo "usage: $0 <cuda_device> [pseudo_label.py args...]" >&2
  echo "  cuda_device: cuda:2 or 2" >&2
  echo "  embeds test query+gallery with weights/finetuned/eva02.pt, clusters with HDBSCAN" >&2
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

RAW_DEVICE="$1"
shift

if [[ "$RAW_DEVICE" == cuda:* ]]; then
  GPU="${RAW_DEVICE#cuda:}"
elif [[ "$RAW_DEVICE" =~ ^[0-9]+$ ]]; then
  GPU="$RAW_DEVICE"
else
  echo "device must be cuda:N or N" >&2
  usage
  exit 1
fi

PYTHON="${PYTHON:-.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-weights/finetuned/eva02.pt}"
ITER="${ITER:-1}"
OUTPUT="${OUTPUT:-runs/pseudo/iter$(printf '%03d' "$ITER")}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-4}"
MIN_SAMPLES="${MIN_SAMPLES:-4}"
export CUDA_VISIBLE_DEVICES="$GPU"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "pseudo-label test embeddings with $CHECKPOINT on GPU $GPU -> $OUTPUT"
"$PYTHON" scripts/pseudo_label.py \
  --checkpoint "$CHECKPOINT" \
  --device cuda \
  --iter "$ITER" \
  --output "$OUTPUT" \
  --min-cluster-size "$MIN_CLUSTER_SIZE" \
  --min-samples "$MIN_SAMPLES" \
  "$@"
