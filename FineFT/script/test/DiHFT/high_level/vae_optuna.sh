nohup python RL/DiHFT/high_level/vae_routing_optuna.py \
    --dataset_name BNBUSDT --max_holding_number 100 \
    >log/DiHFT/BNBUSDT/high_level/optuna/advantage.log 2>&1 & # --window_length_max 150  --window_length_min 50 \
# --gamma_min 0.92 --gamma_max 0.96 \
# --rule_base_threshold_min 0.2 --rule_base_threshold_max 0.5 \

# nohup python RL/DiHFT/high_level/vae_routing_optuna.py \
#     --dataset_name BTCUSDT --max_holding_number 8 \
#     >log/DiHFT/BTCUSDT/high_level/optuna/advantage.log 2>&1 & # --window_length_max 90  --window_length_min 30 \
# # --gamma_min 0.96 --gamma_max 1 \
# # --rule_base_threshold_min 0.1 --rule_base_threshold_max 0.6 \

nohup python RL/DiHFT/high_level/vae_routing_optuna.py \
    --dataset_name DOTUSDT --max_holding_number 6000 \
    >log/DiHFT/DOTUSDT/high_level/optuna/advantage.log 2>&1 &
# --window_length_max 100  --window_length_min 40 \
# --gamma_min 0.92 --gamma_max 0.98 \
# --rule_base_threshold_min 0.5 --rule_base_threshold_max 0.9 \

# nohup python RL/DiHFT/high_level/vae_routing_optuna.py \
#     --dataset_name ETHUSDT --max_holding_number 160 \
#     >log/DiHFT/ETHUSDT/high_level/optuna/advantage.log 2>&1 & # --window_length_max 360  --window_length_min 60 \
# # --gamma_min 0.99 --gamma_max 1 \
# # --rule_base_threshold_min 0.01 --rule_base_threshold_max 0.55 \
