#!/usr/bin/env bash
# Full MAD-Cars pipeline: wait for the 4 download shards, wait for the stop-at-10
# CV pipeline (logs/madcars_experiments/finished.txt), then start the big llm2clip
# pretrain on madcars_full (69,955 cars x 20 views) with the PK 16x4 camera-diverse
# sampler recipe. Idempotent via markers in logs/madcars_full/.
set -uo pipefail

cd "$(dirname "$0")/.."
STATE=logs/madcars_full
mkdir -p "$STATE"

NAME=madcars_full
EXPERIMENT=current_best_tuned_pk4cd
GPU_CANDIDATES="0 1 3 2"

log() { echo "[$(date '+%F %T')] $*" >> "$STATE/pipeline.log"; }

shards_done() { grep -h "^DONE" logs/download_madcars_full_shard*.log 2>/dev/null | wc -l; }

wait_download() {
  while true; do
    n=$(shards_done)
    if [ "$n" -ge 4 ]; then
      log "download complete: $(grep -h '^DONE' logs/download_madcars_full_shard*.log | tr '\n' ' ')"
      return 0
    fi
    if ! pgrep -f "[d]ownload_madcars.py" > /dev/null; then
      log "ERROR: all download processes exited with only $n/4 shards DONE"
      return 1
    fi
    sleep 300
  done
}

pick_gpu() {
  local g used
  for g in $GPU_CANDIDATES; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo -1)
    if [ "$used" = "0" ]; then
      echo "$g"
      return 0
    fi
  done
  return 1
}

[ -f "$STATE/download.done" ] || { wait_download && touch "$STATE/download.done" || exit 1; }

# sanity: subsample csv present and non-trivial
.venv/bin/python - <<'EOF' || { log "ERROR: subsample20.csv sanity check failed"; exit 1; }
import pandas as pd
df = pd.read_csv("extra_data/madcars/meta/subsample20.csv")
assert len(df) > 1_300_000 and df.car_id.nunique() > 65_000, (len(df), df.car_id.nunique())
print(f"sanity ok: {len(df)} images, {df.car_id.nunique()} cars")
EOF

# wait for the screening pipeline (tilt CV) to release the GPUs
while [ ! -f logs/madcars_experiments/finished.txt ]; do
  sleep 300
done

while true; do
  gpu=$(pick_gpu) && break || { sleep 300; }
done
log "starting pretrain $NAME on GPU $gpu (experiment=$EXPERIMENT, datasets=[madcars_full])"

run_dir=$(ls -td "runs/pretrain/${NAME}/"*/ 2>/dev/null | head -1 | sed 's:/$::')
if [ -n "$run_dir" ] && [ -f "$run_dir/run_summary.json" ]; then
  log "$NAME already has a finished run: $run_dir"
  exit 0
fi

common_args=(
  "experiment=$EXPERIMENT" model=llm2clip "pretrain.datasets=[madcars_full]"
  "name=$NAME" trainer.devices=1 eval.device=cuda
  model.checkpoint_path=weights/llm2clip/LLM2CLIP-EVA02-L-14-336.pt
  model.local_files_only=true data.image_size='[336,336]'
  model.head.local_parts=0 eval.tta.scales='[1.0]' data.verify_files=true
)
if [ -n "$run_dir" ] && [ -f "$run_dir/checkpoints/last.ckpt" ]; then
  log "resuming crashed pretrain $run_dir on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python pretrain.py \
    "${common_args[@]}" resume="$run_dir/checkpoints/last.ckpt" output_dir="$run_dir" \
    > "$STATE/${NAME}_pretrain.log" 2>&1
else
  CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python pretrain.py \
    "${common_args[@]}" > "$STATE/${NAME}_pretrain.log" 2>&1
fi
rc=$?
log "pretrain exited rc=$rc (log: $STATE/${NAME}_pretrain.log)"
exit $rc
