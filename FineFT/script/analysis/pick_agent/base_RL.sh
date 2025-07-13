nohup python analysis/pick_agent/RL_base.py \
    --dataset_name BNBUSDT \
    >log/base/pick/BNBUSDT.log 2>&1 &

nohup python analysis/pick_agent/RL_base.py \
    --dataset_name BTCUSDT \
    >log/base/pick/BTCUSDT.log 2>&1 &

nohup python analysis/pick_agent/RL_base.py \
    --dataset_name DOTUSDT \
    >log/base/pick/DOTUSDT.log 2>&1 &

nohup python analysis/pick_agent/RL_base.py \
    --dataset_name ETHUSDT \
    >log/base/pick/ETHUSDT.log 2>&1 &
