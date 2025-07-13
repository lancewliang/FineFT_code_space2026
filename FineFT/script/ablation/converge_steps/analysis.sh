nohup python analysis/ablation/converge_steps/collect_result.py \
    --dataset_name BNBUSDT \
    >log/ablation/convergence_BNBUSDT.log 2>&1 &

nohup python analysis/ablation/converge_steps/collect_result.py \
    --dataset_name BTCUSDT \
    >log/ablation/convergence_BTCUSDT.log 2>&1 &


nohup python analysis/ablation/converge_steps/collect_result.py \
    --dataset_name DOTUSDT \
    >log/ablation/convergence_DOTUSDT.log 2>&1 &


nohup python analysis/ablation/converge_steps/collect_result.py \
    --dataset_name ETHUSDT \
    >log/ablation/convergence_ETHUSDT.log 2>&1 &
