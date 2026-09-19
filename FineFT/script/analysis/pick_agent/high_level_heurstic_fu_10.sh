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

if [ "${FOREGROUND}" = "1" ]; then
    python -u FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py \
        "${CMD_ARGS[@]}"
else
    nohup python -u FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py \
        "${CMD_ARGS[@]}" \
        >"log/analysis/pick_agent/DiHFT/${DATASET_NAME}/high_level_heurstic/${EXPERIMENT_NAME}.log" 2>&1 &
    PID=$!
    echo "Started DiHFT_high_level_heurstic (PID: ${PID}) -> log/analysis/pick_agent/DiHFT/${DATASET_NAME}/high_level_heurstic/${EXPERIMENT_NAME}.log"
fi
