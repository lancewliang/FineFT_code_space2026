function run_ddqn_context {
    local dataset_name=$1
    local max_holding_number=$2
    local epoch_start=$3
    local epoch_end=$4
    local label_types=("slope" "volatility")
    if [ -n "$LABEL_TYPE" ]; then
        label_types=("$LABEL_TYPE")
    fi

    for label_type in "${label_types[@]}"; do
        # 检查并创建日志目录
        log_dir="log/DiHFT/${dataset_name}/low_level/test/${label_type}"
        mkdir -p "${log_dir}"

        # 保存PID的数组
        pids=()

        # 循环执行Epoch 1到50
        for epoch in $(seq $epoch_start $epoch_end); do

            local gpu_index=$(((epoch - 1) % 4))
            CUDA_VISIBLE_DEVICES=${gpu_index} nohup python RL/DiHFT/low_level/test_agent_index.py \
                --dataset_name "${dataset_name}" \
                --max_holding_number "${max_holding_number}" \
                --epoch_num "${epoch}" \
                --label_type "${label_type}" \
                >"${log_dir}/epoch_${epoch}.log" 2>&1 &
            pids+=($!) # 将每个后台进程的PID添加到数组中
        done

        echo "${dataset_name} ${max_holding_number} label_type ${label_type} All processes completed successfully."
    done
}

function run_ddqn_average {
    local dataset_name=$1
    local max_holding_number=$2
    local epoch_start=$3
    local epoch_end=$4

    # 检查并创建日志目录
    log_dir="log/DiHFT/${dataset_name}/low_level/test_average"
    mkdir -p "${log_dir}"

    # 保存PID的数组
    pids=()

    # 循环执行Epoch 1到50
    for epoch in $(seq $epoch_start $epoch_end); do

        local gpu_index=$(((epoch - 1) % 4))
        CUDA_VISIBLE_DEVICES=${gpu_index} nohup python RL/DiHFT/low_level/test_agent_average.py \
            --dataset_name "${dataset_name}" \
            --max_holding_number "${max_holding_number}" \
            --epoch_num "${epoch}" \
            >"${log_dir}/bin_${bin_size}_epoch_${epoch}.log" 2>&1 &
        pids+=($!) # 将每个后台进程的PID添加到数组中
    done

    # 等待所有的PID
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "${dataset_name} ${max_holding_number} ${bin_size} All processes completed successfully."
}

# # # BNBUSDT
# # run_ddqn_average BNBUSDT 100 1 100
# run_ddqn_context BNBUSDT 100 1 50

# # # # BTCUSDT
# # run_ddqn_average BTCUSDT 8 1 100
# run_ddqn_context BTCUSDT 8 1 50

# # #DOTUSDT
# # run_ddqn_average DOTUSDT 6000 1 100
run_ddqn_context DOTUSDT 6000 1 50

# # # ETHUSDT
# # run_ddqn_average ETHUSDT 160 1 100
run_ddqn_context ETHUSDT 160 1 50

# BTCUSDT
# run_ddqn_context BTCUSDT 8 45 100

# ETHUSDT
# run_ddqn_context ETHUSDT 160 1 100

# # BNBUSDT
# run_ddqn_context BNBUSDT 100 1 100

# #DOTUSDT
# run_ddqn_context DOTUSDT 6000 1 100
