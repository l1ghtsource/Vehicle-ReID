#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for fold in 0 1 2 3 4; do
  .venv/bin/python train.py "$@" "data.fold=$fold"
done
