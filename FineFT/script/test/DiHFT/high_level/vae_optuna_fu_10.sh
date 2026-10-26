#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

DATASET_NAME=${DATASET_NAME:-fu}
BASE_PATH=${BASE_PATH:-dataset/10min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}
EXPERIMENT_MANIFEST=${EXPERIMENT_MANIFEST:?set EXPERIMENT_MANIFEST to the validated expert experiment manifest}
CALIBRATION_RUN_ID=${CALIBRATION_RUN_ID:?set CALIBRATION_RUN_ID to a new step-7 run identifier}
MAX_HOLDING_NUMBER=${MAX_HOLDING_NUMBER:-1}
POSITION_CHOICES=${POSITION_CHOICES:-3}
LEVERAGE_CHOICES=${LEVERAGE_CHOICES:-1}
INITIAL_LEVERAGE=${INITIAL_LEVERAGE:-1}
INITIAL_WALLET_BALANCE=${INITIAL_WALLET_BALANCE:-6000}
ORDER_BOOK_DEPTH=${ORDER_BOOK_DEPTH:-5}
TRANSACTION_COST=${TRANSACTION_COST:-0.0003}
ENABLE_NON_MAIN_DEFENSE=${ENABLE_NON_MAIN_DEFENSE:-1}
NON_MAIN_DEFENSE_MODE=${NON_MAIN_DEFENSE_MODE:-graceful_hold_exit}
ENABLE_MDD_PENALTY=${ENABLE_MDD_PENALTY:-1}
TARGET_MDD=${TARGET_MDD:-0.05}
MDD_PENALTY_WEIGHT=${MDD_PENALTY_WEIGHT:-4.0}
MIN_TRADES_GUARD=${MIN_TRADES_GUARD:-60}
ACTION_PERSISTENCE=${ACTION_PERSISTENCE:-6}
STOP_LOSS_RETURN_THRESHOLD=${STOP_LOSS_RETURN_THRESHOLD:-0.03}
STOP_LOSS_COOLDOWN_STEPS=${STOP_LOSS_COOLDOWN_STEPS:-12}
CIRCUIT_BREAKER_CONSECUTIVE_STOPS=${CIRCUIT_BREAKER_CONSECUTIVE_STOPS:-3}
CIRCUIT_BREAKER_COOLING_STEPS=${CIRCUIT_BREAKER_COOLING_STEPS:-24}
SELECTION_MANIFEST=${SELECTION_MANIFEST:-analysis_result/DiHFT/low_level/${DATASET_NAME}/${EXPERIMENT_NAME}/two_dimensional_selection/two_dimensional_selection_manifest.json}
N_TRIALS=${N_TRIALS:-400}
N_WORKERS=${N_WORKERS:-60}

LOG_DIR="log/DiHFT/fu/high_level/expert_hysteresis/${EXPERIMENT_NAME}/${CALIBRATION_RUN_ID}"
mkdir -p "${LOG_DIR}"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate finetf
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate finetf
else
    echo "conda is required; activate the finetf environment before step 7" >&2
    exit 1
fi
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"
# CPU-only inference: batch-1 MLP forwards are latency-bound, GPUs stay idle
export CUDA_VISIBLE_DEVICES=""

DEFENSE_ARGS=()
if [[ "${ENABLE_NON_MAIN_DEFENSE}" == "1" || "${ENABLE_NON_MAIN_DEFENSE}" == "true" ]]; then
    DEFENSE_ARGS+=(
        --enable_non_main_contract_defense
        --non_main_defense_mode "${NON_MAIN_DEFENSE_MODE}"
    )
fi
if [[ "${ENABLE_MDD_PENALTY}" == "1" || "${ENABLE_MDD_PENALTY}" == "true" ]]; then
    DEFENSE_ARGS+=(
        --enable_mdd_penalty
        --target_mdd "${TARGET_MDD}"
        --mdd_penalty_weight "${MDD_PENALTY_WEIGHT}"
        --min_trades_guard "${MIN_TRADES_GUARD}"
    )
else
    DEFENSE_ARGS+=(--no-enable_mdd_penalty)
fi

python -u FineFT/RL/DiHFT/high_level/vae_routing_optuna.py \
    --base_path "${BASE_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --max_holding_number "${MAX_HOLDING_NUMBER}" \
    --initial_wallet_balance "${INITIAL_WALLET_BALANCE}" \
    --position_choices "${POSITION_CHOICES}" \
    --leverage_choices "${LEVERAGE_CHOICES}" \
    --initial_leverage "${INITIAL_LEVERAGE}" \
    --order_book_depth "${ORDER_BOOK_DEPTH}" \
    --selection_manifest "${SELECTION_MANIFEST}" \
    --experiment_manifest "${EXPERIMENT_MANIFEST}" \
    --calibration_run_id "${CALIBRATION_RUN_ID}" \
    --gating_strategy expert_hysteresis \
    --n_workers "${N_WORKERS}" \
    --transcation_cost "${TRANSACTION_COST}" \
    --short_estimated_rate 0 \
    --long_estimated_rate 0 \
    --n_trials "${N_TRIALS}" \
    --action_persistence "${ACTION_PERSISTENCE}" \
    --stop_loss_return_threshold "${STOP_LOSS_RETURN_THRESHOLD}" \
    --stop_loss_cooldown_steps "${STOP_LOSS_COOLDOWN_STEPS}" \
    --circuit_breaker_consecutive_stops "${CIRCUIT_BREAKER_CONSECUTIVE_STOPS}" \
    --circuit_breaker_cooling_steps "${CIRCUIT_BREAKER_COOLING_STEPS}" \
    "${DEFENSE_ARGS[@]}" \
    >"${LOG_DIR}/optuna.log" 2>&1
