nohup python analysis/pick_agent/EarnHFT.py \
    --dataset_name BNBUSDT \
    >log/EarnHFT/BNBUSDT/low_level/pick.log 2>&1 &

nohup python analysis/pick_agent/EarnHFT.py \
    --dataset_name BTCUSDT \
    >log/EarnHFT/BTCUSDT/low_level/pick.log 2>&1 &

nohup python analysis/pick_agent/EarnHFT.py \
    --dataset_name DOTUSDT \
    >log/EarnHFT/DOTUSDT/low_level/pick.log 2>&1 &

nohup python analysis/pick_agent/EarnHFT.py \
    --dataset_name ETHUSDT \
    >log/EarnHFT/ETHUSDT/low_level/pick.log 2>&1 &
