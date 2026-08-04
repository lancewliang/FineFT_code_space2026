#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate finetf 2>/dev/null || true
export PYTHONPATH="${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

BASE_PATH=${BASE_PATH:-dataset/30min}
DATASET_NAME=${DATASET_NAME:-fu}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-30min_multi}
SAVE_PATH=${SAVE_PATH:-analysis_result/DiHFT/high_level_heurstic}

mkdir -p "log/analysis/pick_agent/DiHFT/${DATASET_NAME}/high_level_heurstic"

nohup python -u FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py \
    --base_path "${BASE_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --save_path "${SAVE_PATH}" \
    >"log/analysis/pick_agent/DiHFT/${DATASET_NAME}/high_level_heurstic/${EXPERIMENT_NAME}.log" 2>&1 &
