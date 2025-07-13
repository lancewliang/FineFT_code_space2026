run_grid_search() {
    local dataset_name=$1
    local max_holding_number=$2
    local long_term_list=(120 240 360 60 30 60 90)
    local mid_term_list=(60 120 240 30 15 15 45)
    local short_term_list=(30 60 120 15 6 3 15)

    log_dir="log/base/rule/macd/${dataset_name}"
    mkdir -p "${log_dir}"
    pids=()

    local length=${#long_term_list[@]}

    for ((i = 0; i < $length; i++)); do
        local long_term=${long_term_list[$i]}
        local mid_term=${mid_term_list[$i]}
        local short_term=${short_term_list[$i]}

        local gpu_id=$((gpu_index_number % 4))
        CUDA_VISIBLE_DEVICES=${gpu_id} nohup python RL/base/rule_macd_util.py \
            --dataset_name "${dataset_name}" \
            --max_holding_number "${max_holding_number}" \
            --short_term "${short_term}" \
            --mid_term "${mid_term}" \
            --long_term "${long_term}" \
            >"${log_dir}/short_${short_term}_mid_${mid_term}_long_${long_term}.log" 2>&1 &
        pids+=($!)
        ((gpu_index_number++)) # Increment GPU index to distribute across GPUs
    done

    echo "${dataset_name} ${max_holding_number} All processes initiated successfully."
    # Uncomment the following lines if you want to wait for all processes to complete
    # for pid in "${pids[@]}"; do
    #     wait "$pid" || echo "Process $pid failed!"
    # done

    echo "${dataset_name} ${max_holding_number} All processes completed successfully."
}


# BNBUSDT
run_grid_search BNBUSDT 1000
# BTCUSDT
run_grid_search BTCUSDT 8
# DOTUSDT
run_grid_search DOTUSDT 6000
# ETHUSDT
run_grid_search ETHUSDT 160