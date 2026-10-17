#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate finetf 2>/dev/null || true
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

BASE_PATH=${BASE_PATH:-dataset/10min}
DATASET_NAME=${DATASET_NAME:-fu}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}
SAVE_PATH=${SAVE_PATH:-analysis_result/DiHFT/high_level_heurstic}
RESULT_PATH=${RESULT_PATH:-result/DiHFT/high_level}
SELECTION_METRIC=${SELECTION_METRIC:-tr}
EARLY_STOP=${EARLY_STOP:-0}
FOREGROUND=${FOREGROUND:-0}

AUDIT_OUTPUT_DIR=${AUDIT_OUTPUT_DIR:-${SAVE_PATH}/${DATASET_NAME}/${EXPERIMENT_NAME}/daily_trading_audit}
TRADING_DAY_MAPPING=${TRADING_DAY_MAPPING:-}
if [ -z "${TRADING_DAY_MAPPING}" ] && [ -f "${BASE_PATH}/${DATASET_NAME}/trading_day_mapping.json" ]; then
    TRADING_DAY_MAPPING="${BASE_PATH}/${DATASET_NAME}/trading_day_mapping.json"
fi
RUN_ID=${RUN_ID:-}
ENABLE_AUDIT=${ENABLE_AUDIT:-1}

mkdir -p "log/analysis/pick_agent/DiHFT/${DATASET_NAME}/high_level_heurstic"

CMD_ARGS=(
    --base_path "${BASE_PATH}"
    --dataset_name "${DATASET_NAME}"
    --experiment_name "${EXPERIMENT_NAME}"
    --save_path "${SAVE_PATH}"
    --result_path "${RESULT_PATH}"
    --selection_metric "${SELECTION_METRIC}"
    --early_stop "${EARLY_STOP}"
)

if [ -n "${OPTUNA_CSV:-}" ]; then
    CMD_ARGS+=(--optuna_csv "${OPTUNA_CSV}")
fi

AUDIT_CMD_ARGS=(
    --result_dir "result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}"
    --market_data_dir "${BASE_PATH}/${DATASET_NAME}/valid"
    --output_dir "${AUDIT_OUTPUT_DIR}"
)

if [ -n "${TRADING_DAY_MAPPING}" ]; then
    AUDIT_CMD_ARGS+=(--trading_day_mapping "${TRADING_DAY_MAPPING}")
fi

if [ -n "${RUN_ID}" ]; then
    AUDIT_CMD_ARGS+=(--run_id "${RUN_ID}")
fi

check_audit_ready() {
    local target_dir="result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}"
    if [ -f "${target_dir}/audit_metadata.json" ] && [ -f "${target_dir}/audit_trades.json" ]; then
        return 0
    fi
    if [ -d "${target_dir}/contracts" ]; then
        local total_c
        local meta_c
        total_c=$(find "${target_dir}/contracts" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        meta_c=$(find "${target_dir}/contracts" -mindepth 2 -maxdepth 2 -name "audit_metadata.json" 2>/dev/null | wc -l)
        if [ "${total_c}" -gt 0 ] && [ "${total_c}" -eq "${meta_c}" ]; then
            return 0
        fi
    fi
    return 1
}

run_pipeline() {
    python -u FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py \
        "${CMD_ARGS[@]}"

    python -u FineFT/analysis/diagnostics/trading_diagnostics.py \
        --result_dir "result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}" \
        --data_dir "${BASE_PATH}/${DATASET_NAME}/valid" \
        --output_dir "${SAVE_PATH}/${DATASET_NAME}/${EXPERIMENT_NAME}/diagnostics"

    if [ "${ENABLE_AUDIT}" = "1" ]; then
        if check_audit_ready; then
            python -u FineFT/analysis/diagnostics/daily_trading_audit.py \
                --skip-missing-input \
                "${AUDIT_CMD_ARGS[@]}"
        else
            echo "Notice: No audit artifacts found in result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}, skipping daily trading audit (applies to newly generated audit-grade backtests only)."
        fi
    fi
}

if [ "${FOREGROUND}" = "1" ]; then
    run_pipeline
else
    AUDIT_CHECK_CODE='
    target_dir="result/DiHFT/final_result/'"${DATASET_NAME}"'/'"${EXPERIMENT_NAME}"'"
    has_audit=0
    if [ -f "${target_dir}/audit_metadata.json" ] && [ -f "${target_dir}/audit_trades.json" ]; then
        has_audit=1
    elif [ -d "${target_dir}/contracts" ]; then
        total_c=$(find "${target_dir}/contracts" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        meta_c=$(find "${target_dir}/contracts" -mindepth 2 -maxdepth 2 -name "audit_metadata.json" 2>/dev/null | wc -l)
        if [ "${total_c}" -gt 0 ] && [ "${total_c}" -eq "${meta_c}" ]; then
            has_audit=1
        fi
    fi
    if [ "${has_audit}" = "1" ]; then
        python -u FineFT/analysis/diagnostics/daily_trading_audit.py --skip-missing-input '"${AUDIT_CMD_ARGS[*]}"'
    else
        echo "Notice: No audit artifacts found in ${target_dir}, skipping daily trading audit."
    fi
    '
    AUDIT_STEP=""
    if [ "${ENABLE_AUDIT}" = "1" ]; then
        AUDIT_STEP=' && '"${AUDIT_CHECK_CODE}"
    fi

    nohup bash -c '
        source "'$(conda info --base)'/etc/profile.d/conda.sh" 2>/dev/null || true
        conda activate finetf 2>/dev/null || true
        export PYTHONPATH="'${ROOTPATH}':'${ROOTPATH}'/FineFT${PYTHONPATH:+:${PYTHONPATH}}"
        python -u FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py '"${CMD_ARGS[*]}"' && \
        python -u FineFT/analysis/diagnostics/trading_diagnostics.py \
            --result_dir "result/DiHFT/final_result/'"${DATASET_NAME}"'/'"${EXPERIMENT_NAME}"'" \
            --data_dir "'"${BASE_PATH}"'/'"${DATASET_NAME}"'/valid" \
            --output_dir "'"${SAVE_PATH}"'/'"${DATASET_NAME}"'/'"${EXPERIMENT_NAME}"'/diagnostics"'"${AUDIT_STEP}"'
    ' >"log/analysis/pick_agent/DiHFT/${DATASET_NAME}/high_level_heurstic/${EXPERIMENT_NAME}.log" 2>&1 &
    PID=$!
    echo "Started DiHFT_high_level_heurstic with diagnostics and audit (PID: ${PID}) -> log/analysis/pick_agent/DiHFT/${DATASET_NAME}/high_level_heurstic/${EXPERIMENT_NAME}.log"
fi
