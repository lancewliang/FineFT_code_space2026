nohup python analysis/pick_agent/FineFT_single_agent_with_different_position.py \
    --dataset_name BTCUSDT \
    >log/analysis/pick_agent/DiHFT/BTCUSDT.log 2>&1 &

nohup python analysis/pick_agent/FineFT_single_agent_with_different_position.py \
    --dataset_name BNBUSDT \
    >log/analysis/pick_agent/DiHFT/BNBUSDT.log 2>&1 &

nohup python analysis/pick_agent/FineFT_single_agent_with_different_position.py \
    --dataset_name ETHUSDT \
    >log/analysis/pick_agent/DiHFT/ETHUSDT.log 2>&1 &

nohup python analysis/pick_agent/FineFT_single_agent_with_different_position.py \
    --dataset_name DOTUSDT \
    >log/analysis/pick_agent/DiHFT/DOTUSDT.log 2>&1 &
