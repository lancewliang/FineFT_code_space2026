nohup python analysis/plot/trade_cases.py \
    --dataset_name BNBUSDT \
    --selected_start 998 --selected_end 1003 \
    >log/analysis/plot/trade_BNBUSDT.log 2>&1 &

nohup python analysis/plot/trade_cases.py \
    --dataset_name BTCUSDT \
    --selected_start 980 --selected_end 1005 \
    >log/analysis/plot/trade_BTCUSDT.log 2>&1 &

nohup python analysis/plot/trade_cases.py \
    --dataset_name DOTUSDT \
    --selected_start 1050 --selected_end 1765 \
    >log/analysis/plot/trade_DOTUSDT.log 2>&1 &

nohup python analysis/plot/trade_cases.py \
    --dataset_name ETHUSDT \
    --selected_start 1015 --selected_end 1500 \
    >log/analysis/plot/trade_ETHUSDT.log 2>&1 &
