#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

DATASET_NAME=${DATASET_NAME:-fu}
BASE_PATH=${BASE_PATH:-dataset/5min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-5min_parallel}
MODEL_PATH=${MODEL_PATH:-result/DiHFT/low_level/${DATASET_NAME}/${EXPERIMENT_NAME}/weights_advantage_pretrain/epoch_75/trained_model.pkl}
BUFFER_PATH=${BUFFER_PATH:-result/DiHFT/low_level/${DATASET_NAME}/${EXPERIMENT_NAME}/weights_advantage_pretrain/buffer_diverse.pkl}
OUTPUT_DIR="analysis_result/DiHFT/agent_ood/${DATASET_NAME}/${EXPERIMENT_NAME}"
LOG_DIR="log/analysis/agent_ood/DiHFT/${DATASET_NAME}/${EXPERIMENT_NAME}"
EVAL_MODE=${EVAL_MODE:-env_rollout}
DEVICE=${DEVICE:-cpu}
MAX_SAMPLES=${MAX_SAMPLES:-2000}
ORDER_BOOK_DEPTH=${ORDER_BOOK_DEPTH:-5}

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate finetf 2>/dev/null || true
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

python -u FineFT/analysis/feature/low_level_agent_ood_analysis.py \
    --model_path "${MODEL_PATH}" \
    --buffer_path "${BUFFER_PATH}" \
    --data_dir "${BASE_PATH}/${DATASET_NAME}" \
    --feature_path "${BASE_PATH}/${DATASET_NAME}/rl_state_features.npy" \
    --split test \
    --symbol "${DATASET_NAME}" \
    --target_freq 5min \
    --eval_mode "${EVAL_MODE}" \
    --order_book_depth "${ORDER_BOOK_DEPTH}" \
    --device "${DEVICE}" \
    --max_samples_per_grid "${MAX_SAMPLES}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/agent_ood.log"
