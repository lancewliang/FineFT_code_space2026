#!/usr/bin/env bash

# Step 8: publish the *already selected* calibration Trial.  Selection happens
# only in step 7; this entry point deliberately has no metric/Trial arguments.
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

BASE_PATH=${BASE_PATH:-dataset/10min}
DATASET_NAME=${DATASET_NAME:-fu}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}
SAVE_PATH=${SAVE_PATH:-analysis_result/DiHFT/high_level_heurstic}
MARKET_DATA_DIR=${MARKET_DATA_DIR:-"${BASE_PATH}/${DATASET_NAME}/valid"}
PUBLICATION_DIR=${PUBLICATION_DIR:-"${SAVE_PATH}/${DATASET_NAME}/${EXPERIMENT_NAME}/selected_trial_publication"}

SELECTION_EVIDENCE=${SELECTION_EVIDENCE:-"result/DiHFT/high_level/fu/10min_parallel/expert_hysteresis/fu-10min-expert-hysteresis-v1/valid/calibration/20261026/selection_evidence.json"}
if [[ ! -f "${SELECTION_EVIDENCE}" ]]; then
    echo "SELECTION_EVIDENCE does not exist: ${SELECTION_EVIDENCE}" >&2
    exit 2
fi
if [[ -n "${FROZEN_CONFIG:-}" && ! -f "${FROZEN_CONFIG}" ]]; then
    echo "FROZEN_CONFIG does not exist: ${FROZEN_CONFIG}" >&2
    exit 2
fi
if [[ -n "${BEST_REPLAY:-}" && ! -d "${BEST_REPLAY}" ]]; then
    echo "BEST_REPLAY does not exist: ${BEST_REPLAY}" >&2
    exit 2
fi

LOG_DIR="log/DiHFT/${DATASET_NAME}/high_level/expert_hysteresis/${EXPERIMENT_NAME}/valid/selected_trial_publication"
LOG_FILE="${LOG_DIR}/publication.log"
mkdir -p "${LOG_DIR}"

command=(
    python -u FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py
    --selection-evidence "$SELECTION_EVIDENCE"
    --market-data-dir "$MARKET_DATA_DIR"
    --output-dir "$PUBLICATION_DIR"
)

if [[ -n "${FROZEN_CONFIG:-}" ]]; then
    command+=(--frozen-config "$FROZEN_CONFIG")
fi
if [[ -n "${BEST_REPLAY:-}" ]]; then
    command+=(--best-replay "$BEST_REPLAY")
fi

echo "Step 8 publication log: ${LOG_FILE}"
run_logged_foreground "${LOG_FILE}" "${command[@]}"
