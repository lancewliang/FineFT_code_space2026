run_rl_test() {
    local dataset_name=$1
    local max_holding_number=$2
    local max_context=$3

    # 检查并创建日志目录
    log_dir="log/DiHFT/${dataset_name}/high_level/test"
    mkdir -p "${log_dir}"

    # 保存PID的数组
    pids=()

    # 循环执行context 0到(max_context-1)
    for context_number in $(seq 0 $((max_context - 1))); do
        local gpu_index=$((context_number % 4))
        CUDA_VISIBLE_DEVICES=${gpu_index} nohup python RL/DiHFT/high_level/test_single_agent.py \
            --dataset_name "${dataset_name}" \
            --max_holding_number "${max_holding_number}" \
            --context_index "${context_number}" \
            >"${log_dir}/context_number_${context_number}.log" 2>&1 &
        pids+=($!) # 将每个后台进程的PID添加到数组中
    done

    # 等待所有的PID


    echo "${dataset_name} ${max_holding_number} with context up to ${max_context} processes initiate successfully."
}




# BNBUSDT
run_rl_test BNBUSDT 1000 5
# BTCUSDT
run_rl_test BTCUSDT 8 5
#DOTUSDT
run_rl_test DOTUSDT 6000 5
# # ETHUSDT
run_rl_test ETHUSDT 160 5