function run_ddqn_context {
    local dataset_name=$1
    local max_holding_number=$2
    local epoch_start=$3
    local epoch_end=$4
    local reject_quantile=$5
    local reject_threshold=$6

    # 检查并创建日志目录
    log_dir="log/DiHFT/${dataset_name}/high_level/test_reject_quantile_${reject_quantile}_reject_threshold_${reject_threshold}"
    mkdir -p "${log_dir}"

    # 保存PID的数组
    pids=()

    # 循环执行Epoch 1到100
    for epoch in $(seq $epoch_start $epoch_end); do

        local gpu_index=$(((epoch - 1) % 4))
        CUDA_VISIBLE_DEVICES=${gpu_index} nohup python RL/DiHFT/high_level/test_high_level_rejection.py \
            --dataset_name "${dataset_name}" \
            --max_holding_number "${max_holding_number}" \
            --epoch_number "${epoch}" \
            --reject_quantile "${reject_quantile}" \
            --risk_reject_rate_theshold "${reject_threshold}" \
            >"${log_dir}/epoch_${epoch}.log" 2>&1 &
        pids+=($!) # 将每个后台进程的PID添加到数组中
    done

    # 等待所有的PID
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "${dataset_name} ${max_holding_number}  All processes completed successfully."
}

# BTCUSDT

run_ddqn_context BTCUSDT 8 1 100 0 0.1

# ETHUSDT

run_ddqn_context ETHUSDT 160 1 100 0 0.1

# BNBUSDT

run_ddqn_context BNBUSDT 1000 1 100 0 0.1

#DOTUSDT

run_ddqn_context DOTUSDT 6000 1 100 0 0.1


run_ddqn_context BTCUSDT 8 1 100 0 0.2
run_ddqn_context ETHUSDT 160 1 100 0 0.2
run_ddqn_context BNBUSDT 1000 1 100 0 0.2
run_ddqn_context DOTUSDT 6000 1 100 0 0.2
