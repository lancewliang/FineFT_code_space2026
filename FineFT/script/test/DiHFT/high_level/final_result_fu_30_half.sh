#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

DATASET_NAME=${DATASET_NAME:-fu}
BASE_PATH=${BASE_PATH:-dataset/30min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-30min_multi}
MAX_HOLDING_NUMBER=${MAX_HOLDING_NUMBER:-2}

PARA_FILE="result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/high_level_agent_para.txt"

GAMMA_VAL=""
WINDOW_VAL=""
THRESH_VAL=""

if [ -f "${PARA_FILE}" ]; then
    PARA_NAME=$(cat "${PARA_FILE}" | head -n 1)
    GAMMA_VAL=$(echo "${PARA_NAME}" | sed -n 's/.*gamma_\([0-9.]*\).*/\1/p')
    WINDOW_VAL=$(echo "${PARA_NAME}" | sed -n 's/.*window_\([0-9]*\).*/\1/p')
    THRESH_VAL=$(echo "${PARA_NAME}" | sed -n 's/.*threshold_\([0-9.]*\).*/\1/p')
fi

WINDOW_LENGTH=${WINDOW_LENGTH:-${WINDOW_VAL:-64}}
GAMMA=${GAMMA:-${GAMMA_VAL:-0.9}}
RULE_BASE_THRESHOLD=${RULE_BASE_THRESHOLD:-${THRESH_VAL:-0.2}}

mkdir -p "log/DiHFT/${DATASET_NAME}/high_level/final_result/${EXPERIMENT_NAME}"

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate finetf 2>/dev/null || true
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

python -u FineFT/RL/DiHFT/high_level/vae_routing_final_result_macro_action.py \
    --base_path "${BASE_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --max_holding_number "${MAX_HOLDING_NUMBER}" \
    --initial_wallet_balance 10000 \
    --position_choices 5 \
    --label_number 4 \
    --transcation_cost 0.0004 \
    --short_estimated_rate 0 \
    --long_estimated_rate 0 \
    --window_length "${WINDOW_LENGTH}" \
    --gamma "${GAMMA}" \
    --rule_base_threshold "${RULE_BASE_THRESHOLD}" \
    >"log/DiHFT/${DATASET_NAME}/high_level/final_result/${EXPERIMENT_NAME}/final_result.log" 2>&1
