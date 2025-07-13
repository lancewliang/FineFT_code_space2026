# BNBUSDT
CUDA_VISIBLE_DEVICES=0 nohup python RL/base/dqn_train.py \
    --dataset_name BNBUSDT --max_holding_number 100 \
    >log/base/dqn/BNBUSDT/train.log 2>&1 &

# BTCUSDT
CUDA_VISIBLE_DEVICES=1 nohup python RL/base/dqn_train.py \
    --dataset_name BTCUSDT --max_holding_number 8 \
    >log/base/dqn/BTCUSDT/train.log 2>&1 &

# DOTUSDT
CUDA_VISIBLE_DEVICES=2 nohup python RL/base/dqn_train.py \
    --dataset_name DOTUSDT --max_holding_number 6000 \
    >log/base/dqn/DOTUSDT/train.log 2>&1 &

# ETHUSDT
CUDA_VISIBLE_DEVICES=3 nohup python RL/base/dqn_train.py \
    --dataset_name ETHUSDT --max_holding_number 160 \
    >log/base/dqn/ETHUSDT/train.log 2>&1 &
