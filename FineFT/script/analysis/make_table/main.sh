nohup python analysis/calculate_metric/conclude_metric.py \
    --dataset_name BNBUSDT \
    >log/analysis/table/BNBUSDT.log 2>&1 &

nohup python analysis/calculate_metric/conclude_metric.py \
    --dataset_name BTCUSDT \
    >log/analysis/table/BTCUSDT.log 2>&1 &

nohup python analysis/calculate_metric/conclude_metric.py \
    --dataset_name DOTUSDT \
    >log/analysis/table/DOTUSDT.log 2>&1 &

nohup python analysis/calculate_metric/conclude_metric.py \
    --dataset_name ETHUSDT \
    >log/analysis/table/ETHUSDT.log 2>&1 &




nohup python analysis/calculate_metric/conclude_behavior_metric.py \
    --dataset_name BNBUSDT \
    >log/analysis/table/behavior_BNBUSDT.log 2>&1 &

nohup python analysis/calculate_metric/conclude_behavior_metric.py \
    --dataset_name BTCUSDT \
    >log/analysis/table/behavior_BTCUSDT.log 2>&1 &

nohup python analysis/calculate_metric/conclude_behavior_metric.py \
    --dataset_name DOTUSDT \
    >log/analysis/table/behavior_DOTUSDT.log 2>&1 &

nohup python analysis/calculate_metric/conclude_behavior_metric.py \
    --dataset_name ETHUSDT \
    >log/analysis/table/behavior_ETHUSDT.log 2>&1 &