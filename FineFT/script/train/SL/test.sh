CUDA_VISIBLE_DEVICES=0 nohup python RL/SL/test_adaboost.py \
    --dataset_name BNBUSDT \
    >log/SL/BNBUSDT_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python RL/SL/test_adaboost.py \
    --dataset_name BTCUSDT \
    >log/SL/BTCUSDT_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python RL/SL/test_adaboost.py \
    --dataset_name DOTUSDT \
    >log/SL/DOTUSDT_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python RL/SL/test_adaboost.py \
    --dataset_name ETHUSDT \
    >log/SL/ETHUSDT_test.log 2>&1 &
