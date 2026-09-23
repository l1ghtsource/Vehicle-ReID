#!/usr/bin/env bash
# Three parallel MAD-Cars pretrain experiments, each followed by 3-fold CV:
#   lane A (gpu 0): llm2clip on madcars, default recipe
#   lane B (gpu 1): llm2clip on madcars, tilt augmentation (experiment=current_best_tuned_tilt)
#   lane C (gpu 3): llm2clip on vric+madcars mix, default recipe
# Idempotent per lane: finished stages are skipped; crashed pretrains resume from last.ckpt.
# usage: madcars_experiments.sh
set -uo pipefail

cd "$(dirname "$0")/.."
STATE=logs/madcars_experiments
mkdir -p "$STATE"

log() { echo "[$(date '+%F %T')] $*" >> "$STATE/pipeline.log"; }

summary_ckpt() {
  .venv/bin/python -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["best_checkpoint"] or s["last_checkpoint"])' "$1"
}

# lane <name> <gpu> <experiment> <datasets>
lane() {
  local name="$1" gpu="$2" exp="$3" datasets="$4"
  local marker="$STATE/$name.done"
  [ -f "$marker" ] && { log "$name: already done, skipping"; return 0; }

  local run_dir
  run_dir=$(ls -td "runs/pretrain/${name}/"*/ 2>/dev/null | head -1 | sed 's:/$::')

  if [ -n "$run_dir" ] && [ -f "$run_dir/run_summary.json" ]; then
    log "$name: found finished pretrain $run_dir"
  else
    if [ -n "$run_dir" ] && [ -f "$run_dir/checkpoints/last.ckpt" ]; then
      log "$name: resuming crashed pretrain $run_dir"
      CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python pretrain.py \
        "experiment=$exp" model=llm2clip "pretrain.datasets=[$datasets]" \
        "name=$name" trainer.devices=1 eval.device=cuda \
        model.checkpoint_path=weights/llm2clip/LLM2CLIP-EVA02-L-14-336.pt \
        model.local_files_only=true data.image_size='[336,336]' \
        model.head.local_parts=0 eval.tta.scales='[1.0]' data.verify_files=false \
        resume="$run_dir/checkpoints/last.ckpt" output_dir="$run_dir" \
        >> "$STATE/${name}_pretrain.log" 2>&1
    else
      log "$name: starting pretrain on GPU $gpu (datasets=[$datasets] experiment=$exp)"
      WAIT_GPU_FREE=1 POLL_SECS=60 FREE_STREAK=2 NAME="$name" EXPERIMENT="$exp" \
        bash scripts/pretrain.sh "$gpu" llm2clip "$datasets" \
        data.verify_files=false \
        > "$STATE/${name}_pretrain.log" 2>&1
    fi
    RC=$?
    [ $RC -ne 0 ] && { log "$name: PRETRAIN FAILED rc=$RC (see $STATE/${name}_pretrain.log)"; return 1; }
  fi

  run_dir=$(ls -td "runs/pretrain/${name}/"*/ 2>/dev/null | head -1 | sed 's:/$::')
  local ckpt
  ckpt=$(summary_ckpt "$run_dir/run_summary.json")
  log "$name: pretrain best checkpoint: $ckpt"

  local cv_root="runs/cv/${name}_cv"
  if [ ! -f "$cv_root/cv_metrics.json" ]; then
    log "$name: starting 3-fold CV on GPU $gpu"
    bash scripts/cv_from_pretrain.sh "$gpu" "$ckpt" "$cv_root" 0 1 2 \
      > "$STATE/${name}_cv.log" 2>&1
    RC=$?
    [ $RC -ne 0 ] && { log "$name: CV FAILED rc=$RC (see $STATE/${name}_cv.log)"; return 1; }
  fi
  touch "$marker"
  log "$name: lane complete"
}

lane madcars_default 0 current_best_tuned madcars &
PID_A=$!
lane madcars_tilt 1 current_best_tuned_tilt madcars &
PID_B=$!
lane madcars_mix 3 current_best_tuned vric,madcars &
PID_C=$!
wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
wait "$PID_C"; RC_C=$?
log "lanes finished rc: default=$RC_A tilt=$RC_B mix=$RC_C"

# --- combined report ----------------------------------------------------------
.venv/bin/python - <<'EOF'
import json
from pathlib import Path

vric = {0: (0.8293, 0.8414), 1: (0.8094, 0.8149), 2: (0.8225, 0.8117)}
lines = [
    "# MAD-Cars experiments: 3-fold CV comparison",
    "",
    "Reference: vric pretrain (best ckpt) and no-pretrain baseline below.",
    "",
    "| lane | fold0 mAP | fold1 mAP | fold2 mAP | mean mAP | mean Rank-1 |",
    "|---|---|---|---|---|---|",
]
found = False
for name in ["madcars_default", "madcars_tilt", "madcars_mix"]:
    cv = Path(f"runs/cv/{name}_cv")
    if not cv.exists():
        continue
    maps, r1s = [], []
    ok = True
    for fold in range(3):
        p = cv / f"fold{fold}/val/metrics.json"
        if not p.is_file():
            ok = False
            break
        m = json.loads(p.read_text())["metrics"]
        maps.append(m["mAP"])
        r1s.append(m["Rank-1"])
    if not ok:
        continue
    found = True
    lines.append(
        f"| {name} | {maps[0]:.4f} | {maps[1]:.4f} | {maps[2]:.4f} "
        f"| {sum(maps)/3:.4f} | {sum(r1s)/3:.4f} |"
    )
vm = sum(v[0] for v in vric.values()) / 3
vr = sum(v[1] for v in vric.values()) / 3
lines += [
    "",
    f"Reference vric pretrain: mean mAP {vm:.4f}, mean Rank-1 {vr:.4f}",
    "Reference no pretrain (5-fold trial23): mAP 0.847, Rank-1 0.842",
]
Path("logs/madcars_experiments/REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
EOF

log "all lanes finished, report at logs/madcars_experiments/REPORT.md"
echo ALL DONE > "$STATE/finished.txt"
