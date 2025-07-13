nohup python analysis/calculate_metric/conclude_financial_metric_sl.py \
    --dataset_name BNBUSDT \
    >log/analysis/SL/BNBUSDT.log 2>&1 &

nohup python analysis/calculate_metric/conclude_financial_metric_sl.py \
    --dataset_name BTCUSDT \
    >log/analysis/SL/BTCUSDT.log 2>&1 &

nohup python analysis/calculate_metric/conclude_financial_metric_sl.py \
    --dataset_name DOTUSDT \
    >log/analysis/SL/DOTUSDT.log 2>&1 &

nohup python analysis/calculate_metric/conclude_financial_metric_sl.py \
    --dataset_name ETHUSDT \
    >log/analysis/SL/ETHUSDT.log 2>&1 &