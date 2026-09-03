#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-30min_multi}

mkdir -p "log/fu/low_level/train/30min/${EXPERIMENT_NAME}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

numactl --cpunodebind=1 --membind=1 python -u FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py \
    --base_path dataset/30min \
    --dataset_name fu --experiment_name "${EXPERIMENT_NAME}" \
    --initial_wallet_balance 10000 --batch_size 10240 --update_times=30 \
    --max_holding_number 1 --short_estimated_rate 0 --long_estimated_rate 0 \
    --position_choices 3 --transcation_cost 0.001 --n_step 12 --gamma 0.9999 \
    --order_book_depth 5 --early_stop 2 --N 13 --rollout_steps 1024 \
    --num_sample 300 --pretrain_epoch 80 --lr_init 0.0001 --epsilon_min 0.05 \
    --ada_min 0.01 --ada_step 280000 --neighbor_size 2 \
    --allow_reverse_position \
    >"log/fu/low_level/train/30min/${EXPERIMENT_NAME}/advantage-30min.log"
