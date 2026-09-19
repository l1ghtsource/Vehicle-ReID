#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  echo "usage: $0 <cuda_device> <model> <datasets> [experiment] [hydra overrides...]" >&2
  echo "  cuda_device: cuda:2 or 2" >&2
  echo "  model: Hydra model group (dinov3_convnext_base, llm2clip, radio, ...)" >&2
  echo "  datasets: veri | vric | veri,vric | both | test_train_ssl_crop | test_train_ssl_full" >&2
  echo "  experiment defaults to current_best_tuned; override with a 4th token or EXPERIMENT=" >&2
}

if [[ $# -lt 3 ]]; then
  usage
  exit 1
fi

RAW_DEVICE="$1"
MODEL="$2"
DATASETS="$3"
shift 3

if [[ "$RAW_DEVICE" == cuda:* ]]; then
  GPU="${RAW_DEVICE#cuda:}"
elif [[ "$RAW_DEVICE" =~ ^[0-9]+$ ]]; then
  GPU="$RAW_DEVICE"
else
  echo "device must be cuda:N or N" >&2
  usage
  exit 1
fi

if [[ "$DATASETS" == both ]]; then
  DATASETS="veri,vric"
fi
IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"
for dataset in "${DATASET_LIST[@]}"; do
  case "$dataset" in
    veri|vric|test_train_ssl|test_train_ssl_crop|test_train_ssl_full) ;;
    *)
      echo "unknown dataset: $dataset" >&2
      usage
      exit 1
      ;;
  esac
done
ssl=0
for dataset in "${DATASET_LIST[@]}"; do
  case "$dataset" in
    test_train_ssl|test_train_ssl_crop|test_train_ssl_full) ssl=$((ssl + 1)) ;;
  esac
done
if [[ "$ssl" -gt 0 && ( "$ssl" -ne 1 || ${#DATASET_LIST[@]} -ne 1 ) ]]; then
  echo "test_train_ssl cannot mix with labeled extra datasets" >&2
  usage
  exit 1
fi
DATASETS=$(IFS=,; echo "${DATASET_LIST[*]}")
DATASET_TAG="${DATASETS//,/_}"

EXPERIMENT="${EXPERIMENT:-current_best_tuned}"
if [[ $# -gt 0 && "$1" != *=* ]]; then
  EXPERIMENT="$1"
  shift
fi

PYTHON="${PYTHON:-.venv/bin/python}"
NAME="${NAME:-${MODEL}_${DATASET_TAG}}"
export CUDA_VISIBLE_DEVICES="$GPU"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [[ -z "${MODEL_CHECKPOINT:-}" ]]; then
  case "$MODEL" in
    dinov3_convnext_base) MODEL_CHECKPOINT="weights/dinov3_base/model.safetensors" ;;
    dinov3_convnext_large) MODEL_CHECKPOINT="weights/dinov3_large/model.safetensors" ;;
    radio) MODEL_CHECKPOINT="weights/radio/model.safetensors" ;;
    llm2clip) MODEL_CHECKPOINT="weights/llm2clip/LLM2CLIP-EVA02-L-14-336.pt" ;;
  esac
fi

args=(
  "experiment=$EXPERIMENT"
  "model=$MODEL"
  "pretrain.datasets=[$DATASETS]"
  "name=$NAME"
  "trainer.devices=1"
  "eval.device=cuda"
)
if [[ -n "${MODEL_CHECKPOINT:-}" ]]; then
  args+=("model.checkpoint_path=$MODEL_CHECKPOINT" "model.local_files_only=true")
fi
case "$MODEL" in
  llm2clip)
    args+=(
      "data.image_size=[336,336]"
      "model.head.local_parts=0"
      "eval.tta.scales=[1.0]"
    )
    ;;
  radio|vit)
    args+=("model.head.local_parts=0")
    ;;
esac
if [[ "$ssl" -eq 1 ]]; then
  args+=("loss=dino" "data.sampler.kind=random")
fi

echo "pretrain $MODEL on $DATASETS with experiment=$EXPERIMENT (GPU $GPU)"
"$PYTHON" pretrain.py "${args[@]}" "$@"
