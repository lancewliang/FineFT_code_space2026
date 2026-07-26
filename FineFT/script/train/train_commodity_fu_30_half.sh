#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-30min_multi}

mkdir -p "log_futures/fu/low_level/train/30min/${EXPERIMENT_NAME}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
export PYTHONPATH="${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

python -u FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py \
    --base_path dataset/30min \
    --dataset_name fu --experiment_name "${EXPERIMENT_NAME}" \
    --initial_wallet_balance 10000 --batch_size 512 --update_times=30 \
    --max_holding_number 2 --short_estimated_rate 0 --long_estimated_rate 0 \
    --position_choices 5 --transcation_cost 0.0004 --n_step 12 --gamma 0.99 \
    --order_book_depth 5 --early_stop 2 \
    --pretrain_epoch 100 --lr_init 0.0005 --epsilon_min 0.05 \
    --ada_min 2.0 --ada_step 8000000 \
    --allow_reverse_position \
    >"log_futures/fu/low_level/train/30min/${EXPERIMENT_NAME}/advantage-30min.log"
