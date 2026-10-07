#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}

mkdir -p "log/DiHFT/fu/low_level/train/10min/${EXPERIMENT_NAME}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

python -u FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py \
    --base_path dataset/10min \
    --dataset_name fu --experiment_name "${EXPERIMENT_NAME}" \
    --result_path result/DiHFT/low_level \
    --initial_wallet_balance 10000 --batch_size 130000 --update_times=300 --diverse_num_workers 40 \
    --max_holding_number 1 --short_estimated_rate 0 --long_estimated_rate 0 \
    --position_choices 3 --transcation_cost 0.0003 --n_step 18 --gamma 0.992 \
    --order_book_depth 5 --early_stop 2  --N 11 --buffer_size 700000 \
    --pretrain_epoch 2 --curriculum_block_epochs 5 --num_epoch 62 --lr_init 0.0005 --lr_min 0.0001 --ada_init 96.0 --epsilon_min 0.05 \
    --ada_min 0.1 --neighbor_size 2 --load_pretrain_model False --action_persistence 3 --eval_dfs "0,6,12" \
    --turnover_base_rate "${TURNOVER_BASE_RATE:-0.004}" \
    --turnover_adverse_ratio "${TURNOVER_ADVERSE_RATIO:-6.0}" \
    >"log/DiHFT/fu/low_level/train/10min/${EXPERIMENT_NAME}/advantage-10min-parallel.log"
