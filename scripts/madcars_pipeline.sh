#!/usr/bin/env bash
# Full MAD-Cars experiment chain: wait download -> pretrain llm2clip on madcars ->
# 3-fold CV from the pretrain best checkpoint -> report.
# Idempotent: completed stages are skipped, safe to relaunch at any time.
# usage: madcars_pipeline.sh [gpu]
set -uo pipefail

cd "$(dirname "$0")/.."
GPU="${1:-2}"
PIPE=logs/madcars_pipeline
mkdir -p "$PIPE"

log() { echo "[$(date '+%F %T')] $*" >> "$PIPE/pipeline.log"; }

summary_ckpt() {
  .venv/bin/python -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["best_checkpoint"] or s["last_checkpoint"])' "$1"
}

# --- stage 1: wait for the 4 download shards ---------------------------------
if [ -f "$PIPE/pretrain.done" ] || [ -f "$PIPE/pretrain.running" ]; then
  log "stage1: already finished or superseded, skipping download wait"
else
  log "stage1: waiting for 4 download shards to finish"
  while true; do
    n=0
    for s in 0 1 2 3; do
      grep -q "^DONE" "logs/download_madcars_shard$s.log" 2>/dev/null && n=$((n + 1))
    done
    [ "$n" -eq 4 ] && break
    pgrep -f "download_madcars.py" > /dev/null || { log "stage1: WARNING no downloader alive, only $n/4 shards done"; }
    sleep 120
  done
  imgs=$(find extra_data/madcars/images -name "*.jpg" | wc -l)
  log "stage1: download complete, $imgs images on disk"
fi

# --- stage 2: pretrain llm2clip on madcars -----------------------------------
RUN_DIR=$(ls -td runs/pretrain/llm2clip_madcars/*/ 2>/dev/null | head -1 | sed 's:/$::')
if [ -f "$PIPE/pretrain.done" ]; then
  log "stage2: pretrain already done ($RUN_DIR)"
elif [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/run_summary.json" ]; then
  log "stage2: found finished pretrain run $RUN_DIR"
  touch "$PIPE/pretrain.done"
else
  if [ -n "$RUN_DIR" ]; then
    log "stage2: resuming interrupted pretrain run $RUN_DIR"
    CKPT_RESUME="$RUN_DIR/checkpoints/last.ckpt"
    touch "$PIPE/pretrain.running"
    if [ -f "$CKPT_RESUME" ]; then
      WAIT_GPU_FREE=1 POLL_SECS=60 FREE_STREAK=2 CUDA_VISIBLE_DEVICES="" \
        .venv/bin/python pretrain.py \
        experiment=current_best_tuned model=llm2clip 'pretrain.datasets=[madcars]' \
        name=llm2clip_madcars trainer.devices=1 eval.device=cuda \
        model.checkpoint_path=weights/llm2clip/LLM2CLIP-EVA02-L-14-336.pt \
        model.local_files_only=true data.image_size='[336,336]' \
        model.head.local_parts=0 eval.tta.scales='[1.0]' \
        resume="$CKPT_RESUME" output_dir="$RUN_DIR" \
        >> "$PIPE/pretrain.log" 2>&1
    else
      log "stage2: run dir exists but no last.ckpt; starting fresh run dir"
      RUN_DIR=""
    fi
    RC=$?
    rm -f "$PIPE/pretrain.running"
    if [ -n "$RUN_DIR" ]; then
      [ $RC -ne 0 ] && { log "stage2: pretrain resume FAILED rc=$RC, see $PIPE/pretrain.log"; exit 1; }
      touch "$PIPE/pretrain.done"
      log "stage2: pretrain resumed run finished"
    fi
  fi
  if [ ! -f "$PIPE/pretrain.done" ]; then
    log "stage2: starting pretrain llm2clip on madcars (GPU $GPU, waiting for it to be free)"
    touch "$PIPE/pretrain.running"
    STAMP=$(date +%Y%m%d_%H%M%S)
    WAIT_GPU_FREE=1 POLL_SECS=60 FREE_STREAK=2 \
      bash scripts/pretrain.sh "$GPU" llm2clip madcars \
      > "$PIPE/pretrain.log" 2>&1
    RC=$?
    rm -f "$PIPE/pretrain.running"
    [ $RC -ne 0 ] && { log "stage2: pretrain FAILED rc=$RC, see $PIPE/pretrain.log"; exit 1; }
    touch "$PIPE/pretrain.done"
    log "stage2: pretrain finished"
  fi
fi

RUN_DIR=$(ls -td runs/pretrain/llm2clip_madcars/*/ 2>/dev/null | head -1 | sed 's:/$::')
PRE_CKPT=$(summary_ckpt "$RUN_DIR/run_summary.json")
log "stage2: pretrain best checkpoint: $PRE_CKPT"

# --- stage 3: 3-fold CV from the pretrain checkpoint -------------------------
STAMP=$(date +%Y%m%d_%H%M%S)
CV_ROOT="runs/cv/llm2clip_from_madcars_$STAMP"
if [ -f "$PIPE/cv.done" ]; then
  CV_ROOT=$(cat "$PIPE/cv_root.txt")
  log "stage3: CV already done ($CV_ROOT)"
else
  log "stage3: starting 3-fold CV from $PRE_CKPT on GPU $GPU"
  touch "$PIPE/cv.running"
  bash scripts/cv_from_pretrain.sh "$GPU" "$PRE_CKPT" "$CV_ROOT" > "$PIPE/cv.log" 2>&1
  RC=$?
  rm -f "$PIPE/cv.running"
  [ $RC -ne 0 ] && { log "stage3: CV FAILED rc=$RC, see $PIPE/cv.log"; exit 1; }
  echo "$CV_ROOT" > "$PIPE/cv_root.txt"
  touch "$PIPE/cv.done"
  log "stage3: CV finished"
fi

# --- stage 4: report ----------------------------------------------------------
.venv/bin/python - <<EOF
import json
from pathlib import Path

cv = Path("$CV_ROOT")
rows = []
for fold in range(3):
    m = json.loads((cv / f"fold{fold}/val/metrics.json").read_text())["metrics"]
    rows.append((fold, m["mAP"], m["Rank-1"], m["mAP@10"]))

vric = {0: (0.8293, 0.8414), 1: (0.8094, 0.8149), 2: (0.8225, 0.8117)}
lines = [
    "# MAD-Cars pretrain: CV comparison report",
    "",
    f"Pretrain run: $RUN_DIR",
    f"CV run: $CV_ROOT",
    "",
    "| fold | madcars mAP | madcars Rank-1 | vric mAP | vric Rank-1 |",
    "|---|---|---|---|---|",
]
for fold, mAP, r1, m10 in rows:
    v = vric[fold]
    lines.append(f"| {fold} | {mAP:.4f} | {r1:.4f} | {v[0]:.4f} | {v[1]:.4f} |")
mm = sum(r[1] for r in rows) / len(rows)
mr = sum(r[2] for r in rows) / len(rows)
vm = sum(v[0] for v in vric.values()) / 3
vr = sum(v[1] for v in vric.values()) / 3
lines += [
    "",
    f"mean: madcars mAP {mm:.4f} / Rank-1 {mr:.4f} vs vric mAP {vm:.4f} / Rank-1 {vr:.4f}",
    f"delta mAP: {mm - vm:+.4f}",
    "",
    "Reference (no pretrain, full 5-fold): mAP 0.847, Rank-1 0.842",
]
(cv / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
EOF

log "stage4: report written to $CV_ROOT/REPORT.md"
echo "ALL DONE" > "$PIPE/finished.txt"
