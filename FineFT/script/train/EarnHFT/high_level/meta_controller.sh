CUDA_VISIBLE_DEVICES=0 nohup python RL/EarnHFT/high_level/dqn.py \
    --dataset_name BTCUSDT --max_holding_number 8 \
    >log/EarnHFT/BTCUSDT/high_level/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python RL/EarnHFT/high_level/dqn.py \
    --dataset_name BNBUSDT --max_holding_number 100 \
    >log/EarnHFT/BNBUSDT/high_level/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python RL/EarnHFT/high_level/dqn.py \
    --dataset_name ETHUSDT --max_holding_number 160 \
    >log/EarnHFT/ETHUSDT/high_level/train.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python RL/EarnHFT/high_level/dqn.py \
    --dataset_name DOTUSDT --max_holding_number 6000 \
    >log/EarnHFT/DOTUSDT/high_level/train.log 2>&1 &
