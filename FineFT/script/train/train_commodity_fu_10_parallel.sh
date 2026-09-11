#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}

mkdir -p "log/DiHFT/fu/low_level/train/10min/${EXPERIMENT_NAME}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

numactl --cpunodebind=0 --preferred=0 python -u FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py \
    --base_path dataset/10min \
    --dataset_name fu --experiment_name "${EXPERIMENT_NAME}" \
    --initial_wallet_balance 6000 --batch_size 102400 --update_times=500 \
    --max_holding_number 1 --short_estimated_rate 0 --long_estimated_rate 0 \
    --position_choices 3 --transcation_cost 0.0005 --n_step 12 --gamma 0.992 \
    --order_book_depth 5 --early_stop 2  --N 13 --buffer_size 1000000 \
    --pretrain_epoch 5 --num_epoch 12 --decay_epochs 10 --lr_init 0.0005 --ada_init 96.0 --epsilon_min 0.05 \
    --ada_min 0.1 --neighbor_size 2 --load_pretrain_model False \
    >"log/DiHFT/fu/low_level/train/10min/${EXPERIMENT_NAME}/advantage-10min-parallel.log"
