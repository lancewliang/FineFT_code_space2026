# BTCUSDT

CUDA_VISIBLE_DEVICES=0 nohup python RL/DiHFT/low_level/weight_advantage_pretrain.py \
    --dataset_name BTCUSDT --max_holding_number 8  \
    >log/DiHFT/BTCUSDT/low_level/train/advantage.log 2>&1 &
# ETHUSDT

CUDA_VISIBLE_DEVICES=1 nohup python RL/DiHFT/low_level/weight_advantage_pretrain.py \
    --dataset_name ETHUSDT --max_holding_number 160  \
    >log/DiHFT/ETHUSDT/low_level/train/advantage.log 2>&1 &

# BNBUSDT

CUDA_VISIBLE_DEVICES=2 nohup python RL/DiHFT/low_level/weight_advantage_pretrain.py \
    --dataset_name BNBUSDT --max_holding_number 100  \
    >log/DiHFT/BNBUSDT/low_level/train/advantage.log 2>&1 &
# DOTUSDT

CUDA_VISIBLE_DEVICES=3 nohup python RL/DiHFT/low_level/weight_advantage_pretrain.py \
    --dataset_name DOTUSDT --max_holding_number 6000 \
    >log/DiHFT/DOTUSDT/low_level/train/advantage.log 2>&1 &
