nohup python analysis/pick_agent/EarnHFT_high_level.py \
    --dataset_name BNBUSDT \
    >log/EarnHFT/BNBUSDT/high_level/pick.log 2>&1 &

nohup python analysis/pick_agent/EarnHFT_high_level.py \
    --dataset_name BTCUSDT \
    >log/EarnHFT/BTCUSDT/high_level/pick.log 2>&1 &

nohup python analysis/pick_agent/EarnHFT_high_level.py \
    --dataset_name DOTUSDT \
    >log/EarnHFT/DOTUSDT/high_level/pick.log 2>&1 &

nohup python analysis/pick_agent/EarnHFT_high_level.py \
    --dataset_name ETHUSDT --selected_rank 2 \
    >log/EarnHFT/ETHUSDT/high_level/pick.log 2>&1 &
