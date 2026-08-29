#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-1min}

mkdir -p "log_futures/al/low_level/train/1min/${EXPERIMENT_NAME}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

numactl --cpunodebind=0 --membind=0 python -u FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py \
    --base_path dataset/1min \
    --dataset_name al --experiment_name "${EXPERIMENT_NAME}" \
    --initial_wallet_balance 100000 --batch_size 4096 --update_times=30 \
    --max_holding_number 1 --short_estimated_rate 0 --long_estimated_rate 0 \
    --position_choices 3 --transcation_cost 0.0004 --n_step 12 --gamma 0.99 \
    --order_book_depth 5 --early_stop 20\
    >"log_futures/al/low_level/train/1min/${EXPERIMENT_NAME}/advantage-1min.log"
