#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

DATASET_NAME=${DATASET_NAME:-fu}
DATA_BASE_PATH=${DATA_BASE_PATH:-dataset/30min}
LABEL_COUNT=${LABEL_COUNT:-3}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-30min_multi}
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-2}
LABELING_METHODS=("slope" "volatility")
if [ -n "${LABELING_METHOD:-}" ]; then
    LABELING_METHODS=("${LABELING_METHOD}")
fi

if ! [[ "${MAX_PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_PARALLEL_JOBS must be a positive integer." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

pids=()
failed=0

wait_for_available_slot() {
    while ((${#pids[@]} >= MAX_PARALLEL_JOBS)); do
        if ! wait -n; then
            failed=1
        fi
        prune_finished_jobs
    done
}

prune_finished_jobs() {
    local active_pids=()
    local pid
    for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            active_pids+=("${pid}")
        fi
    done
    pids=("${active_pids[@]}")
}

for method in "${LABELING_METHODS[@]}"; do
    method_exp_name="${EXPERIMENT_NAME}/${method}"
    log_dir="log/DiHFT/${DATASET_NAME}/VAE/${EXPERIMENT_NAME}/${method}"
    mkdir -p "${log_dir}"

    for label_index in $(seq 0 $((LABEL_COUNT - 1))); do
        wait_for_available_slot
        nohup python -u FineFT/RL/DiHFT/VAE/main.py \
            --dataset_name "${DATASET_NAME}" \
            --data_base_path "${DATA_BASE_PATH}" \
            --label_index "${label_index}" \
            --total_label_number "${LABEL_COUNT}" \
            --experiment_name "${method_exp_name}" \
            --labeling_method "${method}" \
            --train \
            >"${log_dir}/train_label_${label_index}.log" 2>&1 &
        pids+=("$!")
    done
done

while ((${#pids[@]} > 0)); do
    if ! wait -n; then
        failed=1
    fi
    prune_finished_jobs
done

if ((failed != 0)); then
    echo "${DATASET_NAME} VAE (${LABELING_METHODS[*]}) labels 0 to $((LABEL_COUNT - 1)) finished with failures."
    exit 1
fi

echo "${DATASET_NAME} VAE (${LABELING_METHODS[*]}) labels 0 to $((LABEL_COUNT - 1)) finished."
