#!/usr/bin/env bash
# Finish remaining folds (3,4) of a CV run; resume fold3 from its last.ckpt if present.
# usage: cv_finish_folds.sh <cuda_device> <run_root> <init_checkpoint>
set -uo pipefail

cd "$(dirname "$0")/.."

GPU="$1"
RUN_ROOT="$2"
CKPT="$3"

summary_ckpt() {
  .venv/bin/python -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["best_checkpoint"] or s["last_checkpoint"])' "$1"
}

for fold in 3 4; do
  FOLD_DIR="$RUN_ROOT/fold$fold"
  if [[ -f "$FOLD_DIR/checkpoints/last.ckpt" && ! -f "$FOLD_DIR/run_summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python train.py \
      experiment=current_best_tuned \
      resume="$FOLD_DIR/checkpoints/last.ckpt" \
      data.fold=$fold \
      trainer.devices=1 \
      output_dir="$FOLD_DIR"
  else
    CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python train.py \
      experiment=current_best_tuned \
      init_checkpoint="$CKPT" \
      data.fold=$fold \
      trainer.devices=1 \
      output_dir="$FOLD_DIR"
  fi
  FOLD_CKPT=$(summary_ckpt "$FOLD_DIR/run_summary.json")
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python eval.py \
    checkpoint="$FOLD_CKPT" \
    data.fold=$fold \
    eval.split=val \
    eval.device=cuda \
    eval.output_dir="$FOLD_DIR/val"
done

.venv/bin/python scripts/aggregate_cv.py \
  "$RUN_ROOT"/fold{0,1,2,3,4}/val/metrics.json \
  --output "$RUN_ROOT/cv_metrics.json"
