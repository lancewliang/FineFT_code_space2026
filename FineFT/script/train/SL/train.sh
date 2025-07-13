CUDA_VISIBLE_DEVICES=0 nohup python RL/SL/train_adaboost.py \
    --dataset_name BNBUSDT \
    >log/SL/BNBUSDT_train.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python RL/SL/train_adaboost.py \
    --dataset_name BTCUSDT \
    >log/SL/BTCUSDT_train.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python RL/SL/train_adaboost.py \
    --dataset_name DOTUSDT \
    >log/SL/DOTUSDT_train.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python RL/SL/train_adaboost.py \
    --dataset_name ETHUSDT \
    >log/SL/ETHUSDT_train.log 2>&1 &
