CUDA_VISIBLE_DEVICES=0 nohup python RL/DiHFT/ablation/safe_routing.py \
    --dataset_name BNBUSDT --max_holding_number 100 \
    --window_length 81 --gamma 0.943861129500238 \
    >log/DiHFT/BNBUSDT/high_level/ablation_safe_routing.log 2>&1 &




CUDA_VISIBLE_DEVICES=1 nohup python RL/DiHFT/ablation/safe_routing.py \
    --dataset_name BTCUSDT --max_holding_number 8 \
    --window_length 60 --gamma 0.99 \
    >log/DiHFT/BTCUSDT/high_level/ablation_safe_routing.log 2>&1 &


CUDA_VISIBLE_DEVICES=2 nohup python RL/DiHFT/ablation/safe_routing.py \
    --dataset_name DOTUSDT --max_holding_number 6000 \
    --window_length 70 --gamma 0.9630864171914365 \
    >log/DiHFT/DOTUSDT/high_level/ablation_safe_routing.log 2>&1 &




CUDA_VISIBLE_DEVICES=3 nohup python RL/DiHFT/ablation/safe_routing.py \
    --dataset_name ETHUSDT --max_holding_number 160 \
    --window_length 240 --gamma 0.99 \
    >log/DiHFT/ETHUSDT/high_level/ablation_safe_routing.log 2>&1 &