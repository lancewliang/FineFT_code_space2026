function run_ddqn_context {
    local dataset_name=$1
    local max_holding_number=$2
    local epoch_start=$3
    local epoch_end=$4

    # 检查并创建日志目录
    log_dir="log/DiHFT/${dataset_name}/high_level/test"
    mkdir -p "${log_dir}"

    # 保存PID的数组
    pids=()

    # 循环执行Epoch 1到100
    for epoch in $(seq $epoch_start $epoch_end); do

        local gpu_index=$(((epoch - 1) % 4))
        CUDA_VISIBLE_DEVICES=${gpu_index} nohup python RL/DiHFT/high_level/test_high_level.py \
            --dataset_name "${dataset_name}" \
            --max_holding_number "${max_holding_number}" \
            --epoch_number "${epoch}" \
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
run_ddqn_context BTCUSDT 8 1 100

# # ETHUSDT
# run_ddqn_context ETHUSDT 160 1 100

# # BNBUSDT
# run_ddqn_context BNBUSDT 1000 1 100

# #DOTUSDT
# run_ddqn_context DOTUSDT 6000 1 100
