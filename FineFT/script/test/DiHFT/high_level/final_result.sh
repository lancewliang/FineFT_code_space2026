# BNBUSDT

CUDA_VISIBLE_DEVICES=2 nohup python RL/DiHFT/high_level/vae_routing_final_result_macro_action.py \
    --dataset_name BNBUSDT --max_holding_number 100 \
    --window_length 61 --gamma 0.9680916878597078 --rule_base_threshold 0.3375892142509265 \
    >log/DiHFT/BNBUSDT/high_level/final_result.log 2>&1 &



# DOTUSDT

CUDA_VISIBLE_DEVICES=3 nohup python RL/DiHFT/high_level/vae_routing_final_result_macro_action.py \
    --dataset_name DOTUSDT --max_holding_number 6000 \
    --window_length 67 --gamma 0.9583643145017051 --rule_base_threshold 0.7382268181198924 \
    >log/DiHFT/DOTUSDT/high_level/final_result.log 2>&1 &

