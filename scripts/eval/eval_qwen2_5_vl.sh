#!/usr/bin/env bash
# Evaluate Qwen2.5-VL-7B with GIFT keyframe selection.
# Edit the settings below, then run:  bash scripts/eval/eval_qwen2_5_vl.sh
set -euo pipefail

# ------------------------------- edit me -------------------------------------
MODEL_PATH="/path/to/Qwen2.5-VL-7B-Instruct"
SIGLIP_PATH="google/siglip-so400m-patch14-384"  # local dir, or HF hub id
TASKS="videomme"        # videomme | longvideobench_val_v | mlvu_dev | mvbench
FRAME_POOL=128          # dense candidate pool that GIFT scores
BUDGET=32               # frames kept and actually fed to the VLM
SELECTOR="GIFT"         # GIFT | AKS | BOLT | uniform
NUM_PROCESSES=8         # number of GPUs
OUTPUT_PATH="./logs/qwen2_5_vl_${SELECTOR}_${BUDGET}"
# ------------------------------------------------------------------------------

export GIFT_SIGLIP_PATH="${SIGLIP_PATH}"
export PYTHONPATH="$(pwd):$(pwd)/lmms-eval:${PYTHONPATH:-}"

python3 -m accelerate.commands.launch \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model=qwen2_5_vl \
    --model_args="pretrained=${MODEL_PATH},max_num_frames=${FRAME_POOL},use_flash_attention_2=True,frame_selector_name=${SELECTOR},select_frame_num=${BUDGET}" \
    --tasks="${TASKS}" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="qwen2_5_vl_${SELECTOR}" \
    --output_path="${OUTPUT_PATH}"
