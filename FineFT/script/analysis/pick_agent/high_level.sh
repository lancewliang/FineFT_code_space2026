nohup python analysis/pick_agent/DiHFT_high_level.py \
    --dataset_name BTCUSDT \
    >log/analysis/pick_agent/DiHFT_high/BTCUSDT.log 2>&1 &

nohup python analysis/pick_agent/DiHFT_high_level.py \
    --dataset_name BNBUSDT \
    >log/analysis/pick_agent/DiHFT_high/BNBUSDT.log 2>&1 &

nohup python analysis/pick_agent/DiHFT_high_level.py \
    --dataset_name ETHUSDT \
    >log/analysis/pick_agent/DiHFT_high/ETHUSDT.log 2>&1 &

nohup python analysis/pick_agent/DiHFT_high_level.py \
    --dataset_name DOTUSDT \
    >log/analysis/pick_agent/DiHFT_high/DOTUSDT.log 2>&1 &
