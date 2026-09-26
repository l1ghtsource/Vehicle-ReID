#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python pretrain.py \
  experiment=current_best_tuned_madcars_full model=llm2clip 'pretrain.datasets=[madcars_full]' \
  name=madcars_full trainer.devices=1 eval.device=cuda 'checkpointing.keep_epochs=[7]' \
  model.checkpoint_path=weights/llm2clip/LLM2CLIP-EVA02-L-14-336.pt \
  model.local_files_only=true 'data.image_size=[336,336]' \
  model.head.local_parts=0 'eval.tta.scales=[1.0]' data.verify_files=true
