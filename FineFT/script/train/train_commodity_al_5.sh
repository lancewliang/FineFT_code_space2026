#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-5min_nstep6_costw5}

mkdir -p "log_futures/al/low_level/train/${EXPERIMENT_NAME}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
export PYTHONPATH="${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

python -u FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py \
    --base_path dataset/5min \
    --dataset_name al --experiment_name "${EXPERIMENT_NAME}" \
    --initial_wallet_balance 10000 --batch_size 2048 --update_times=20 \
    --max_holding_number 0.2 --short_estimated_rate 0 --long_estimated_rate 0 \
    --position_choices 3 --transcation_cost 0.0005 --n_step 6 --gamma 0.97 \
    --order_book_depth 5 \
    >"log_futures/al/low_level/train/${EXPERIMENT_NAME}/advantage.log"
