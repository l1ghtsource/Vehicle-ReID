#!/usr/bin/env bash
# Stop the three MAD-Cars pretrains at TARGET_EPOCH epochs, then run 3-fold CV per
# lane on the lane's GPU, then write logs/madcars_experiments/REPORT.md.
# Replaces madcars_experiments.sh (which treats a killed pretrain as failure).
# Idempotent: markers in logs/madcars_experiments/*.stop10.* guard each step.
set -uo pipefail

cd "$(dirname "$0")/.."
STATE=logs/madcars_experiments
mkdir -p "$STATE"

TARGET_EPOCH=10

log() { echo "[$(date '+%F %T')] $*" >> "$STATE/pipeline.log"; }

lane_epoch() {
  tr '\r' '\n' < "$STATE/${1}_pretrain.log" 2>/dev/null | grep -oE "^Epoch [0-9]+" | tail -1 | awk '{print $2}'
}

stop_and_summarize() {
  local name="$1"
  local marker="$STATE/${name}.stop10.stopped"
  [ -f "$marker" ] && return 0
  local run_dir
  run_dir=$(ls -td "runs/pretrain/${name}/"*/ 2>/dev/null | head -1 | sed 's:/$::')
  [ -z "$run_dir" ] && { log "$name: no run dir yet, cannot stop"; return 1; }
  local ckpt="$run_dir/checkpoints/last.ckpt"
  [ -f "$ckpt" ] || { log "$name: no last.ckpt in $run_dir yet, cannot stop"; return 1; }

  if pgrep -f "name=${name} trainer.devices" > /dev/null; then
    log "$name: stopping pretrain at >= epoch $TARGET_EPOCH"
    pkill -f "name=${name} trainer.devices" || true
    sleep 5
    pkill -9 -f "name=${name} trainer.devices" || true
  fi

  .venv/bin/python - "$run_dir" "$ckpt" "$TARGET_EPOCH" <<'EOF'
import json, sys
from pathlib import Path
run_dir, ckpt, target = sys.argv[1], sys.argv[2], int(sys.argv[3])
summary = {
    "output_dir": run_dir,
    "stopped_early": True,
    "target_epoch": target,
    "best_checkpoint": ckpt,
    "last_checkpoint": ckpt,
}
Path(run_dir, "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
EOF
  touch "$marker"
  log "$name: run_summary.json written (ckpt=$ckpt)"
  return 0
}

start_cv() {
  local name="$1" gpu="$2"
  local marker="$STATE/${name}.stop10.cv_started"
  [ -f "$marker" ] && return 0
  local run_dir ckpt
  run_dir=$(ls -td "runs/pretrain/${name}/"*/ 2>/dev/null | head -1 | sed 's:/$::')
  [ -n "$run_dir" ] && [ -f "$run_dir/run_summary.json" ] || return 1
  ckpt=$(.venv/bin/python -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s["best_checkpoint"] or s["last_checkpoint"])' "$run_dir/run_summary.json")
  touch "$marker"
  log "$name: starting 3-fold CV on GPU $gpu (ckpt=$ckpt)"
  setsid nohup bash scripts/cv_from_pretrain.sh "$gpu" "$ckpt" "runs/cv/${name}_cv" 0 1 2 \
    > "$STATE/${name}_cv.log" 2>&1 < /dev/null &
  return 0
}

all_cv_done() {
  local name
  for name in madcars_default madcars_tilt madcars_mix; do
    [ -f "runs/cv/${name}_cv/cv_metrics.json" ] || return 1
  done
  return 0
}

write_report() {
  .venv/bin/python - <<'EOF'
import json
from pathlib import Path

vric = {0: (0.8293, 0.8414), 1: (0.8094, 0.8149), 2: (0.8225, 0.8117)}
lines = [
    "# MAD-Cars experiments (pretrains cut at 10 epochs): 3-fold CV comparison",
    "",
    "Reference: vric pretrain (27 epochs) and no-pretrain baseline below.",
    "",
    "| lane | fold0 mAP | fold1 mAP | fold2 mAP | mean mAP | mean Rank-1 |",
    "|---|---|---|---|---|---|",
]
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
}

while true; do
  for spec in "madcars_default 0" "madcars_tilt 1" "madcars_mix 3"; do
    set -- $spec
    name=$1; gpu=$2
    if [ -f "$STATE/${name}.stop10.stopped" ]; then
      start_cv "$name" "$gpu" || true
      continue
    fi
    ep=$(lane_epoch "$name")
    if [ -n "${ep:-}" ] && [ "$ep" -ge "$((TARGET_EPOCH + 1))" ]; then
      log "$name: epoch $ep reached (target $TARGET_EPOCH)"
      stop_and_summarize "$name" && start_cv "$name" "$gpu" || true
    elif ! pgrep -f "name=${name} trainer.devices" > /dev/null; then
      log "$name: pretrain process gone at epoch ${ep:-?} — using last.ckpt as-is"
      stop_and_summarize "$name" && start_cv "$name" "$gpu" || true
    fi
  done

  if all_cv_done && [ ! -f "$STATE/finished.txt" ]; then
    write_report
    echo ALL DONE > "$STATE/finished.txt"
    log "all lanes finished, report at $STATE/REPORT.md"
    break
  fi
  sleep 120
done
