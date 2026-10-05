#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

DATASET_NAME=${DATASET_NAME:-fu}
BASE_PATH=${BASE_PATH:-dataset/5min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-5min_parallel}
MAX_HOLDING_NUMBER=${MAX_HOLDING_NUMBER:-1}
ENABLE_NON_MAIN_DEFENSE=${ENABLE_NON_MAIN_DEFENSE:-1}

mkdir -p "log/DiHFT/fu/high_level/optuna/${EXPERIMENT_NAME}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"
# CPU-only inference: batch-1 MLP forwards are latency-bound, GPUs stay idle
export CUDA_VISIBLE_DEVICES=""

DEFENSE_ARGS=()
# if [[ "${ENABLE_NON_MAIN_DEFENSE}" == "1" || "${ENABLE_NON_MAIN_DEFENSE}" == "true" ]]; then
#     DEFENSE_ARGS+=(--enable_non_main_contract_defense)
# fi

python -u FineFT/RL/DiHFT/high_level/vae_routing_optuna.py \
    --base_path "${BASE_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --max_holding_number "${MAX_HOLDING_NUMBER}" \
    --initial_wallet_balance 5000 \
    --position_choices 3 \
    --order_book_depth 5 \
    --selection_manifest "analysis_result/DiHFT/low_level/${DATASET_NAME}/${EXPERIMENT_NAME}/two_dimensional_selection/two_dimensional_selection_manifest.json" \
    --n_workers "${N_WORKERS:-45}" \
    --transcation_cost 0.0005 \
    --short_estimated_rate 0 \
    --long_estimated_rate 0 \
    --n_trials 80 \
    --rule_base_threshold_min 0.1 \
    --rule_base_threshold_max 0.3 \
    "${DEFENSE_ARGS[@]}" \
    >"log/DiHFT/fu/high_level/optuna/${EXPERIMENT_NAME}/optuna.log" 2>&1
