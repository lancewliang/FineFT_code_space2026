run_grid_search() {
    local dataset_name=$1
    local max_holding_number=$2
    local upper_bond_list=(0.2 0.4 0.6 0.8)
    local lower_bond_list=(-0.2 -0.4 -0.6 -0.8)

    log_dir="log/base/rule/iv/${dataset_name}"
    mkdir -p "${log_dir}"
    pids=()

    for upper_bond in "${upper_bond_list[@]}"; do
        for lower_bond in "${lower_bond_list[@]}"; do
            local gpu_id=$((gpu_index_number % 4))
            CUDA_VISIBLE_DEVICES=${gpu_id} nohup python RL/base/rule_imbalance_volume_util.py \
                --dataset_name "${dataset_name}" \
                --max_holding_number "${max_holding_number}" \
                --upper_bond "${upper_bond}" \
                --lower_bond "${lower_bond}" \
                >"${log_dir}/upper_${upper_bond}_lower_${lower_bond}.log" 2>&1 &
            pids+=($!)
            ((gpu_index_number++)) # Increment GPU index to distribute across GPUs
        done
    done
    echo "${dataset_name} ${max_holding_number} All processes initiate successfully."
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
