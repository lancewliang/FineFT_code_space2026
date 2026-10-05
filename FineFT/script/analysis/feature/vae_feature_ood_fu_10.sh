#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

DATASET_NAME=${DATASET_NAME:-fu}
BASE_PATH=${BASE_PATH:-dataset/10min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}
VAE_PATH=${VAE_PATH:-result/DiHFT/vae_results}
SAMPLE_SIZE=${SAMPLE_SIZE:-5000}
OUTPUT_DIR=${OUTPUT_DIR:-analysis_result/DiHFT/feature_ood/${DATASET_NAME}/${EXPERIMENT_NAME}}
LOG_DIR=${LOG_DIR:-log/analysis/feature/DiHFT/${DATASET_NAME}/${EXPERIMENT_NAME}}
AXES=("slope" "volatility")
if [ -n "${AXIS:-}" ]; then
    AXES=("${AXIS}")
fi
extra_args=()
if [ -n "${DEVICE:-}" ]; then
    extra_args+=(--device "${DEVICE}")
fi
if [ -n "${Z_DIM:-}" ]; then
    extra_args+=(--z_dim "${Z_DIM}")
fi
if [ -n "${HIDDEN_DIMS:-}" ]; then
    extra_args+=(--hidden_dims ${HIDDEN_DIMS})
fi

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate finetf 2>/dev/null || true
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

for axis in "${AXES[@]}"; do
    axis_output_dir="${OUTPUT_DIR}/${axis}"
    axis_log_dir="${LOG_DIR}/${axis}"
    mkdir -p "${axis_output_dir}" "${axis_log_dir}"

    echo "Running VAE feature OOD analysis: dataset=${DATASET_NAME} experiment=${EXPERIMENT_NAME} axis=${axis}"
    python -u FineFT/analysis/feature/vae_feature_ood_analysis.py \
        --base_path "${BASE_PATH}" \
        --dataset_name "${DATASET_NAME}" \
        --experiment_name "${EXPERIMENT_NAME}" \
        --vae_path "${VAE_PATH}" \
        --axis "${axis}" \
        --output_dir "${axis_output_dir}" \
        --sample_size "${SAMPLE_SIZE}" \
        --per_contract \
        "${extra_args[@]}" \
        2>&1 | tee "${axis_log_dir}/feature_ood.log"
done
