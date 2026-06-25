#!/usr/bin/env bash
set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
/home/lanceliang/miniconda3/bin/conda run -n finetf env PYTHONPATH="${ROOTPATH}/FineFT" nohup python -u FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py --dataset_name fu --max_holding_number 10 --order_book_depth 5 >log_futures/fu/low_level/train/advantage.log 2>&1 &
