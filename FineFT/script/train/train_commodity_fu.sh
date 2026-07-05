#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

mkdir -p log_futures/fu/low_level/train

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
export PYTHONPATH="${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

python -u FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py \
    --dataset_name fu --initial_wallet_balance 10000 --batch_size 2048 --update_times=20 \
    --max_holding_number 1 --short_estimated_rate 0 --long_estimated_rate 0 \
    --position_choices 3 --transcation_cost 0.0002 \
    --order_book_depth 5 \
    >log_futures/fu/low_level/train/advantage.log
