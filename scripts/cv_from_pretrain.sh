#!/usr/bin/env bash
# Run 5-fold CV from a pretrain checkpoint on one GPU.
# usage: cv_from_pretrain.sh <cuda_device> <init_checkpoint> <run_root> [folds...]
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="$1"
CKPT="$2"
RUN_ROOT="$3"
shift 3
FOLDS="${@:-0 1 2 3 4}"

mkdir -p "$RUN_ROOT"

summary_ckpt() {
  .venv/bin/python -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["best_checkpoint"] or s["last_checkpoint"])' "$1"
}

for fold in $FOLDS; do
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python train.py \
    experiment=current_best_tuned \
    init_checkpoint="$CKPT" \
    data.fold=$fold \
    trainer.devices=1 \
    output_dir="$RUN_ROOT/fold$fold"
  FOLD_CKPT=$(summary_ckpt "$RUN_ROOT/fold$fold/run_summary.json")
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python eval.py \
    checkpoint="$FOLD_CKPT" \
    data.fold=$fold \
    eval.split=val \
    eval.device=cuda \
    eval.output_dir="$RUN_ROOT/fold$fold/val"
done

.venv/bin/python scripts/aggregate_cv.py \
  $(for f in $FOLDS; do echo "$RUN_ROOT/fold$f/val/metrics.json"; done) \
  --output "$RUN_ROOT/cv_metrics.json"
