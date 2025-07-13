nohup python analysis/ablation/safe_routing/routing_result.py \
    --dataset_name BNBUSDT --minum_value 85 \
    >log/ablation/routing_result/BNBUSDT.log 2>&1 &

nohup python analysis/ablation/safe_routing/routing_result.py \
    --dataset_name BTCUSDT --minum_value 80 \
    >log/ablation/routing_result/BTCUSDT.log 2>&1 &


nohup python analysis/ablation/safe_routing/routing_result.py \
    --dataset_name DOTUSDT --minum_value 95 \
    >log/ablation/routing_result/DOTUSDT.log 2>&1 &


nohup python analysis/ablation/safe_routing/routing_result.py \
    --dataset_name ETHUSDT --minum_value 55 \
    >log/ablation/routing_result/ETHUSDT.log 2>&1 &
