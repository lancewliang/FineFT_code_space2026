run_grid_search() {
    local dataset_name=$1
    local max_holding_number=$2
    local window_length_list=(60 120 180 240 300 360)
    local gamma_list=(0.99 0.999 1)
    local rule_base_threshold_list=(0.0 0.05 0.1 0.15 0.2 0.3)
    local gpu_index_number=0

    log_dir="log/DiHFT/${dataset_name}/high_level/test_vae"
    mkdir -p "${log_dir}"
    pids=()

    for window_length in "${window_length_list[@]}"; do
        for gamma in "${gamma_list[@]}"; do
            for threshold in "${rule_base_threshold_list[@]}"; do
                local gpu_id=$((gpu_index_number % 4))
                CUDA_VISIBLE_DEVICES=${gpu_id} nohup python RL/DiHFT/high_level/vae_routing_util.py \
                    --dataset_name "${dataset_name}" \
                    --max_holding_number "${max_holding_number}" \
                    --window_length "${window_length}" \
                    --gamma "${gamma}" \
                    --rule_base_threshold "${threshold}" \
                    >"${log_dir}/w_${window_length}_gamma_${gamma}_threshold_${threshold}.log" 2>&1 &
                pids+=($!)
                ((gpu_index_number++))  # Increment GPU index to distribute across GPUs
            done
        done
    done
    echo "${dataset_name} ${max_holding_number} All processes initiate successfully."
    for pid in "${pids[@]}"; do
        wait "$pid" || echo "Process $pid failed!"
    done

    echo "${dataset_name} ${max_holding_number} All processes completed successfully."
}

# BNBUSDT
run_grid_search BNBUSDT 1000 
# BTCUSDT
run_grid_search BTCUSDT 8 
#DOTUSDT
run_grid_search DOTUSDT 6000 
# ETHUSDT
run_grid_search ETHUSDT 160 
