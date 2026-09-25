#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

DATASET_NAME=${DATASET_NAME:-fu}
BASE_PATH=${BASE_PATH:-dataset/10min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}
VAE_PATH=${VAE_PATH:-result/DiHFT/vae_results}
SAMPLE_SIZE=${SAMPLE_SIZE:-5000}
OUTPUT_DIR="analysis_result/DiHFT/feature_ood/${DATASET_NAME}/${EXPERIMENT_NAME}"
LOG_DIR="log/analysis/feature/DiHFT/${DATASET_NAME}/${EXPERIMENT_NAME}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate finetf 2>/dev/null || true
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

python -u FineFT/analysis/feature/vae_feature_ood_analysis.py \
    --base_path "${BASE_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --vae_path "${VAE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --sample_size "${SAMPLE_SIZE}" \
    --per_contract \
    2>&1 | tee "${LOG_DIR}/feature_ood.log"
