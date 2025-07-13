nohup python analysis/pick_agent/rule_base.py \
    --dataset_name BNBUSDT \
    >log/base/pick/BNBUSDT_policy.log 2>&1 &

nohup python analysis/pick_agent/rule_base.py \
    --dataset_name BTCUSDT \
    >log/base/pick/BTCUSDT_policy.log 2>&1 &

nohup python analysis/pick_agent/rule_base.py \
    --dataset_name DOTUSDT \
    >log/base/pick/DOTUSDT_policy.log 2>&1 &

nohup python analysis/pick_agent/rule_base.py \
    --dataset_name ETHUSDT \
    >log/base/pick/ETHUSDT_policy.log 2>&1 &
