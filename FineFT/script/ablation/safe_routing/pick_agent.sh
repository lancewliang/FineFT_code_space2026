nohup python analysis/ablation/safe_routing/collect_wo_reject.py \
    --dataset_name BNBUSDT \
    >log/ablation/wo_safe_routing/BNBUSDT.log 2>&1 &

nohup python analysis/ablation/safe_routing/collect_wo_reject.py \
    --dataset_name BTCUSDT \
    >log/ablation/wo_safe_routing/BTCUSDT.log 2>&1 &


nohup python analysis/ablation/safe_routing/collect_wo_reject.py \
    --dataset_name DOTUSDT \
    >log/ablation/wo_safe_routing/DOTUSDT.log 2>&1 &


nohup python analysis/ablation/safe_routing/collect_wo_reject.py \
    --dataset_name ETHUSDT \
    >log/ablation/wo_safe_routing/ETHUSDT.log 2>&1 &
