nohup python analysis/plot/DiHFT_single_agent_plot.py \
    --dataset_name BNBUSDT \
    >log/analysis/single_agent/DiHFT/BNBUSDT.log 2>&1 &

nohup python analysis/plot/DiHFT_single_agent_plot.py \
    --dataset_name BTCUSDT \
    >log/analysis/single_agent/DiHFT/BTCUSDT.log 2>&1 &

nohup python analysis/plot/DiHFT_single_agent_plot.py \
    --dataset_name DOTUSDT \
    >log/analysis/single_agent/DiHFT/DOTUSDT.log 2>&1 &

nohup python analysis/plot/DiHFT_single_agent_plot.py \
    --dataset_name ETHUSDT \
    >log/analysis/single_agent/DiHFT/ETHUSDT.log 2>&1 &
