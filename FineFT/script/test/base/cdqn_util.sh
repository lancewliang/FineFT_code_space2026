function run_ddqn_test {
    local dataset_name=$1
    local max_holding_number=$2
    local epoch_start=$3
    local epoch_end=$4

    # 检查并创建日志目录
    log_dir="log/base/cdqn/${dataset_name}/low_level/test"
    mkdir -p "${log_dir}"

    # 保存PID的数组
    pids=()

    # 循环执行Epoch 1到50
    for epoch in $(seq $epoch_start $epoch_end); do

        local gpu_index=$(((epoch - 1) % 4))
        CUDA_VISIBLE_DEVICES=${gpu_index} nohup python RL/base/cdqn_test.py \
            --dataset_name "${dataset_name}" \
            --max_holding_number "${max_holding_number}" \
            --epoch_num "${epoch}" \
            >"${log_dir}/epoch_${epoch}.log" 2>&1 &
        pids+=($!) # 将每个后台进程的PID添加到数组中
    done

    # 等待所有的PID
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "${dataset_name} ${max_holding_number}  All processes completed successfully."
}

run_ddqn_test BNBUSDT 100 1 50
run_ddqn_test BTCUSDT 8 1 50
run_ddqn_test DOTUSDT 6000 1 50
run_ddqn_test ETHUSDT 160 1 50
