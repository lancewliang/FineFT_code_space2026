#!/usr/bin/env bash

# Step 9: replay the selected expert policy on the declared validation or test
# segment. All execution/risk/account settings come from FROZEN_CONFIG.
set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

# shellcheck source=FineFT/script/lib/require_finetf.sh
source "${ROOTPATH}/FineFT/script/lib/require_finetf.sh"
# shellcheck source=FineFT/script/lib/foreground_task.sh
source "${ROOTPATH}/FineFT/script/lib/foreground_task.sh"
require_finetf
install_task_signal_traps
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_NAME=${DATASET_NAME:-fu}
BASE_PATH=${BASE_PATH:-dataset/10min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}
EVAL_STAGE=${1:-${EVAL_STAGE:-valid}}
FROZEN_CONFIG=${FROZEN_CONFIG:?Set FROZEN_CONFIG to the step-8 published frozen_config.json}

if [[ ! -f "$FROZEN_CONFIG" ]]; then
    echo "FROZEN_CONFIG does not exist: $FROZEN_CONFIG" >&2
    exit 2
fi
if [[ "$EVAL_STAGE" != "valid" && "$EVAL_STAGE" != "test" ]]; then
    echo "EVAL_STAGE must be valid or test, got: $EVAL_STAGE" >&2
    exit 2
fi

CONFIG_HASH=$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["config_hash"])' "$FROZEN_CONFIG")
RESULT_DIR="result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/expert_hysteresis/${CONFIG_HASH}/${EVAL_STAGE}/main_sub"
LOG_DIR="log/DiHFT/${DATASET_NAME}/high_level/final_result_main_sub/${EXPERIMENT_NAME}/${CONFIG_HASH}/${EVAL_STAGE}"
LOG_FILE="${LOG_DIR}/final_result.log"
mkdir -p "$LOG_DIR"

echo "Running frozen expert_hysteresis main/sub evaluation: ${EVAL_STAGE}"
run_logged_foreground "${LOG_FILE}" \
    python -u FineFT/RL/DiHFT/high_level/vae_routing_final_result_macro_action.py \
    --frozen_config "$FROZEN_CONFIG" \
    --eval_stage "$EVAL_STAGE" \
    --evaluation_purpose main_sub \
    --result_path result/DiHFT/final_result

AUDIT_OUTPUT_DIR="analysis_result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/expert_hysteresis/${CONFIG_HASH}/${EVAL_STAGE}/main_sub/daily_trading_audit"
AUDIT_ARGS=(
    --result_dir "$RESULT_DIR"
    --market_data_dir "${BASE_PATH}/${DATASET_NAME}/${EVAL_STAGE}"
    --output_dir "$AUDIT_OUTPUT_DIR"
)
if [[ -n "${TRADING_DAY_MAPPING:-}" ]]; then
    AUDIT_ARGS+=(--trading_day_mapping "$TRADING_DAY_MAPPING")
fi
if [[ -n "${RUN_ID:-}" ]]; then
    AUDIT_ARGS+=(--run_id "$RUN_ID")
fi
run_logged_foreground "${LOG_FILE}" python -u FineFT/analysis/diagnostics/daily_trading_audit.py "${AUDIT_ARGS[@]}"

echo "Completed: $RESULT_DIR"
