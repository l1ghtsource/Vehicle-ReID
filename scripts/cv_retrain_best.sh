#!/usr/bin/env bash
# Retrain one CV fold from the pretrain best checkpoint and keep only its best checkpoint.
# usage: cv_retrain_best.sh <cuda_device> <fold> <run_root> <pretrain_ckpt>
set -uo pipefail

cd "$(dirname "$0")/.."

GPU="$1"
FOLD="$2"
RUN_ROOT="$3"
CKPT="$4"
FOLD_DIR="$RUN_ROOT/fold$FOLD"

CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python train.py \
  experiment=current_best_tuned \
  init_checkpoint="$CKPT" \
  data.fold=$FOLD \
  trainer.devices=1 \
  output_dir="$FOLD_DIR"
RC=$?
[ $RC -ne 0 ] && exit $RC

BEST=$(.venv/bin/python -c 'import json; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$FOLD_DIR/run_summary.json")
BEST_ABS=$(realpath "$BEST")
find "$FOLD_DIR/checkpoints" -type f -name "*.ckpt" | while read -r f; do
  [ "$(realpath "$f")" != "$BEST_ABS" ] && rm -f "$f"
done
echo "fold$FOLD kept: $BEST"
