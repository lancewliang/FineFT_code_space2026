#!/usr/bin/env bash

# Step 10: final holdout replay. The frozen selected configuration is the
# sole source of scoring, routing, execution, fee, and risk settings.
set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
fi
conda activate finetf
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_NAME=${DATASET_NAME:-fu}
BASE_PATH=${BASE_PATH:-dataset/10min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}
FROZEN_CONFIG=${FROZEN_CONFIG:?Set FROZEN_CONFIG to the step-8 published frozen_config.json}

if [[ ! -f "$FROZEN_CONFIG" ]]; then
    echo "FROZEN_CONFIG does not exist: $FROZEN_CONFIG" >&2
    exit 2
fi

CONFIG_HASH=$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["config_hash"])' "$FROZEN_CONFIG")
RESULT_DIR="result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/expert_hysteresis/${CONFIG_HASH}/test/final"
LOG_DIR="log/DiHFT/${DATASET_NAME}/high_level/final_result/${EXPERIMENT_NAME}/${CONFIG_HASH}/test"
LOG_FILE="${LOG_DIR}/final_result.log"
mkdir -p "$LOG_DIR"

echo "Running final test replay for frozen expert_hysteresis configuration."
python -u FineFT/RL/DiHFT/high_level/vae_routing_final_result_macro_action.py \
    --frozen_config "$FROZEN_CONFIG" \
    --eval_stage test \
    --evaluation_purpose final \
    --result_path result/DiHFT/final_result \
    2>&1 | tee "$LOG_FILE"

AUDIT_OUTPUT_DIR="analysis_result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/expert_hysteresis/${CONFIG_HASH}/test/final/daily_trading_audit"
AUDIT_ARGS=(
    --result_dir "$RESULT_DIR"
    --market_data_dir "${BASE_PATH}/${DATASET_NAME}/test"
    --output_dir "$AUDIT_OUTPUT_DIR"
)
if [[ -n "${TRADING_DAY_MAPPING:-}" ]]; then
    AUDIT_ARGS+=(--trading_day_mapping "$TRADING_DAY_MAPPING")
fi
if [[ -n "${RUN_ID:-}" ]]; then
    AUDIT_ARGS+=(--run_id "$RUN_ID")
fi
python -u FineFT/analysis/diagnostics/daily_trading_audit.py "${AUDIT_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

echo "Completed: $RESULT_DIR"
