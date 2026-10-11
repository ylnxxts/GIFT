#!/usr/bin/env bash
# Evaluate LLaVA-OneVision-7B with GIFT keyframe selection.
# Edit the settings below, then run:  bash scripts/eval/eval_llava_onevision.sh
set -euo pipefail

# ------------------------------- edit me -------------------------------------
MODEL_PATH="/path/to/llava-onevision-qwen2-7b-ov"
SIGLIP_PATH="google/siglip-so400m-patch14-384"  # local dir, or HF hub id
TASKS="videomme"        # videomme | longvideobench_val_v | mlvu_dev | mvbench
FRAME_POOL=128          # dense candidate pool that GIFT scores
BUDGET=32               # frames kept and actually fed to the VLM
SELECTOR="GIFT"         # GIFT | AKS | BOLT | uniform
NUM_PROCESSES=8         # number of GPUs
OUTPUT_PATH="./logs/llava_ov_${SELECTOR}_${BUDGET}"
# ------------------------------------------------------------------------------

export GIFT_SIGLIP_PATH="${SIGLIP_PATH}"
export PYTHONPATH="$(pwd):$(pwd)/lmms-eval:${PYTHONPATH:-}"

python3 -m accelerate.commands.launch \
    --num_processes="${NUM_PROCESSES}" \
    -m lmms_eval \
    --model=llava_onevision \
    --model_args="pretrained=${MODEL_PATH},conv_template=qwen_1_5,max_frames_num=${FRAME_POOL},frame_selector_name=${SELECTOR},select_frame_num=${BUDGET}" \
    --tasks="${TASKS}" \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix="llava_ov_${SELECTOR}" \
    --output_path="${OUTPUT_PATH}"
