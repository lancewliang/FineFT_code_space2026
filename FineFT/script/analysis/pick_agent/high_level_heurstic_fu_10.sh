#!/usr/bin/env bash

# Step 8: publish the *already selected* calibration Trial.  Selection happens
# only in step 7; this entry point deliberately has no metric/Trial arguments.
set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

BASE_PATH=${BASE_PATH:-dataset/10min}
DATASET_NAME=${DATASET_NAME:-fu}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}
SAVE_PATH=${SAVE_PATH:-analysis_result/DiHFT/high_level_heurstic}
MARKET_DATA_DIR=${MARKET_DATA_DIR:-"${BASE_PATH}/${DATASET_NAME}/valid"}
PUBLICATION_DIR=${PUBLICATION_DIR:-"${SAVE_PATH}/${DATASET_NAME}/${EXPERIMENT_NAME}/selected_trial_publication"}

: "${SELECTION_EVIDENCE:?SELECTION_EVIDENCE must name step-7 selection_evidence.json}"

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

"${command[@]}"
