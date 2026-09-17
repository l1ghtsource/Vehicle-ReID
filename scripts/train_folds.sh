#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
EXPERIMENT="${EXPERIMENT:-dino_base}"
RUN_ROOT="${RUN_ROOT:-runs/cv/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-weights/dinov3_base/model.safetensors}"
GPUS=(3 4 5 6 7)
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p "$RUN_ROOT"
"$PYTHON" -m scripts.prepare_folds "$@" >"$RUN_ROOT/folds.log" 2>&1

pids=()
for fold in 0 1 2 3 4; do
  (
    fold_dir="$RUN_ROOT/fold$fold"
    mkdir -p "$fold_dir"
    CUDA_VISIBLE_DEVICES="${GPUS[$fold]}" "$PYTHON" train.py \
      "experiment=$EXPERIMENT" \
      "model.local_files_only=true" \
      "model.checkpoint_path=$MODEL_CHECKPOINT" \
      "$@" \
      "data.fold=$fold" \
      "trainer.devices=1" \
      "output_dir=$fold_dir" 2>&1 | tee "$fold_dir/train.log"
    checkpoint=$(
      "$PYTHON" -c \
        'import json,sys; s=json.load(open(sys.argv[1])); print(s["best_checkpoint"] or s["last_checkpoint"])' \
        "$fold_dir/run_summary.json"
    )
    if [[ ! -f "$checkpoint" ]]; then
      echo "Checkpoint not found for fold $fold: $checkpoint" >&2
      exit 1
    fi
    CUDA_VISIBLE_DEVICES="${GPUS[$fold]}" "$PYTHON" eval.py \
      "$@" \
      "checkpoint=$checkpoint" \
      "data.fold=$fold" \
      "eval.split=val" \
      "eval.device=cuda" \
      "eval.output_dir=$fold_dir/val" 2>&1 | tee "$fold_dir/eval.log"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if ((failed)); then
  echo "At least one fold failed" >&2
  exit 1
fi

metrics=()
for fold in 0 1 2 3 4; do
  metrics+=("$RUN_ROOT/fold$fold/val/metrics.json")
done
"$PYTHON" scripts/aggregate_cv.py "${metrics[@]}" --output "$RUN_ROOT/cv_metrics.json"

RUN_ROOT="$RUN_ROOT" "$PYTHON" - <<'PY'
import os
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(os.environ["RUN_ROOT"])
frames = []
embeddings = []
offset = 0
for fold in range(5):
    directory = root / f"fold{fold}" / "val"
    frame = pd.read_csv(directory / "oof.csv", dtype={"image_id": str})
    values = np.load(directory / "embeddings.npy")
    if len(frame) != len(values):
        raise ValueError(f"Fold {fold} OOF rows and embeddings differ")
    frame["fold"] = fold
    frame["global_embedding_index"] = np.arange(offset, offset + len(frame))
    offset += len(frame)
    frames.append(frame)
    embeddings.append(values)
oof = pd.concat(frames, ignore_index=True)
if oof.image_id.duplicated().any():
    raise ValueError("OOF image IDs are not unique")
matrix = np.concatenate(embeddings)
np.save(root / "oof_embeddings.npy", matrix.astype(np.float32))
oof.to_csv(root / "oof.csv", index=False)
print(f"OOF embeddings: {matrix.shape} -> {root / 'oof_embeddings.npy'}")
print(f"OOF metadata: {len(oof)} rows -> {root / 'oof.csv'}")
PY

echo "CV metrics: $RUN_ROOT/cv_metrics.json"
echo "Run root: $RUN_ROOT"
