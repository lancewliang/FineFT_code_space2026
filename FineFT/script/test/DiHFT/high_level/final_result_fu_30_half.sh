#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

DATASET_NAME=${DATASET_NAME:-fu}
BASE_PATH=${BASE_PATH:-dataset/30min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-30min_multi}
MAX_HOLDING_NUMBER=${MAX_HOLDING_NUMBER:-2}
POSITION_CHOICES=${POSITION_CHOICES:-5}
ORDER_BOOK_DEPTH=${ORDER_BOOK_DEPTH:-5}
TRANSACTION_COST=${TRANSACTION_COST:-0.0004}

PARA_FILE="result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/high_level_agent_para.txt"
OPTUNA_CSV="result/DiHFT/high_level/${DATASET_NAME}/${EXPERIMENT_NAME}/vae_risk_aware_routing_optuna/optuna_results.csv"
SELECTION_MANIFEST="analysis_result/DiHFT/low_level/${DATASET_NAME}/${EXPERIMENT_NAME}/two_dimensional_selection/two_dimensional_selection_manifest.json"

LOG_DIR="log/DiHFT/${DATASET_NAME}/high_level/final_result/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate finetf 2>/dev/null || true
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

python -u FineFT/RL/DiHFT/high_level/vae_routing_final_result_macro_action.py \
    --base_path "${BASE_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --selection_manifest "${SELECTION_MANIFEST}" \
    --para_file "${PARA_FILE}" \
    --optuna_csv "${OPTUNA_CSV}" \
    --eval_stage test \
    --max_holding_number "${MAX_HOLDING_NUMBER}" \
    --initial_wallet_balance 10000 \
    --position_choices "${POSITION_CHOICES}" \
    --order_book_depth "${ORDER_BOOK_DEPTH}" \
    --transcation_cost "${TRANSACTION_COST}" \
    --short_estimated_rate 0 \
    --long_estimated_rate 0 \
    --allow_reverse_position \
    >"${LOG_DIR}/final_result.log" 2>&1
