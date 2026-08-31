#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

DATASET_NAME=${DATASET_NAME:-fu}
BASE_PATH=${BASE_PATH:-dataset/30min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-30min_multi}
MAX_HOLDING_NUMBER=${MAX_HOLDING_NUMBER:-2}

mkdir -p "log/DiHFT/fu/high_level/optuna/${EXPERIMENT_NAME}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

python -u FineFT/RL/DiHFT/high_level/vae_routing_optuna.py \
    --base_path "${BASE_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --max_holding_number "${MAX_HOLDING_NUMBER}" \
    --initial_wallet_balance 10000 \
    --position_choices 5 \
    --order_book_depth 5 \
    --selection_manifest "analysis_result/DiHFT/low_level/${DATASET_NAME}/${EXPERIMENT_NAME}/two_dimensional_selection/two_dimensional_selection_manifest.json" \
    --n_jobs "${N_JOBS:-10}" \
    --transcation_cost 0.0004 \
    --short_estimated_rate 0 \
    --long_estimated_rate 0 \
    --allow_reverse_position \
    >"log/DiHFT/fu/high_level/optuna/${EXPERIMENT_NAME}/optuna.log" 2>&1
