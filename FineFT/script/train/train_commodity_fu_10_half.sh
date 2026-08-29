#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_multi}

mkdir -p "log/fu/low_level/train/10min/${EXPERIMENT_NAME}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

numactl --cpunodebind=0 --membind=0 python -u FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py \
    --base_path dataset/10min \
    --dataset_name fu --experiment_name "${EXPERIMENT_NAME}" \
    --initial_wallet_balance 10000 --batch_size 8192 --update_times=30 \
    --max_holding_number 2 --short_estimated_rate 0 --long_estimated_rate 0 \
    --position_choices 5 --transcation_cost 0.0004 --n_step 12 --gamma 0.9999 \
    --order_book_depth 5 --early_stop 2 \
    --pretrain_epoch 80 --lr_init 0.0005 --epsilon_min 0.05 \
    --ada_min 0.01 --ada_step 8000000 \
    --allow_reverse_position \
    >"log/fu/low_level/train/10min/${EXPERIMENT_NAME}/advantage-10min_multi.log"
