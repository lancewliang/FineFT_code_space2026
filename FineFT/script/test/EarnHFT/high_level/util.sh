function run_ddqn_pes_risk_aware {
    local dataset_name=$1
    local max_holding_number=$2
    local total_epochs=50
    local epochs_per_batch=50

    # 检查并创建日志目录
    log_dir="log/EarnHFT/${dataset_name}/high_level/test"
    mkdir -p "${log_dir}"

    # 循环执行Epoch 1到total_epochs，每批次处理epochs_per_batch个Epoch
    for ((batch_start = 1; batch_start <= total_epochs; batch_start += epochs_per_batch)); do
        # 保存PID的数组
        pids=()

        # 内部循环，每批次运行epochs_per_batch个Epoch
        for ((epoch = batch_start; epoch < batch_start + epochs_per_batch && epoch <= total_epochs; epoch++)); do
            local gpu_index=$(((epoch - 1) % 4))
            CUDA_VISIBLE_DEVICES=${gpu_index} nohup python RL/EarnHFT/high_level/test_dqn.py \
                --dataset_name "${dataset_name}" \
                --max_holding_number "${max_holding_number}" \
                --epoch_num "${epoch}" \
                >"${log_dir}/epoch_${epoch}.log" 2>&1 &
            pids+=($!) # 将每个后台进程的PID添加到数组中
        done

        # 等待当前批次的所有PID
        for pid in "${pids[@]}"; do
            wait "$pid"
        done

        echo "Batch starting from epoch ${batch_start} completed."
    done

    echo "${dataset_name} ${max_holding_number} All processes completed successfully."
}
run_ddqn_pes_risk_aware BTCUSDT 8
run_ddqn_pes_risk_aware ETHUSDT 160
run_ddqn_pes_risk_aware BNBUSDT 100
run_ddqn_pes_risk_aware DOTUSDT 6000
