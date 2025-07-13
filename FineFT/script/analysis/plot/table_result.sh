nohup python analysis/plot/main_result_plot.py \
    --dataset_name BNBUSDT \
    >log/analysis/plot/table_BNBUSDT.log 2>&1 &

nohup python analysis/plot/main_result_plot.py \
    --dataset_name BTCUSDT \
    >log/analysis/plot/table_BTCUSDT.log 2>&1 &

nohup python analysis/plot/main_result_plot.py \
    --dataset_name DOTUSDT \
    >log/analysis/plot/table_DOTUSDT.log 2>&1 &

nohup python analysis/plot/main_result_plot.py \
    --dataset_name ETHUSDT \
    >log/analysis/plot/table_ETHUSDT.log 2>&1 &
