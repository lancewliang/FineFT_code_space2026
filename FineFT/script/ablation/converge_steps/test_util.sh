function run_test_ablation {
    local dataset_name=$1
    local max_holding_number=$2
    local start_epoch=$3
    local end_epoch=$4
    local batch_size=$5

    local settings=("EarnHFT_PES" "EarnHFT_random" "FineFT" "FineFT_wo_pretrain")
    local betas=(-80 -40 0 40 100)
    # 遍历设置并创建日志目录

    log_EarnHFT_PES="log/ablation/EarnHFT_PES"
    mkdir -p "${log_EarnHFT_PES}"

    log_EarnHFT_random="log/ablation/EarnHFT_random"
    mkdir -p "${log_EarnHFT_random}"

    log_FineFT="log/ablation/FineFT"
    mkdir -p "${log_FineFT}"

    log_FineFT_wo_pretrain="log/ablation/FineFT_wo_pretrain"
    mkdir -p "${log_FineFT_wo_pretrain}"
    #EarnHFT PES

    for beta in "${betas[@]}"; do

        for ((batch_start = start_epoch; batch_start <= end_epoch; batch_start += batch_size)); do
            # 保存PID的数组
            pids=()

            # 内部循环，每批次运行epochs_per_batch个Epoch
            for ((epoch = batch_start; epoch < batch_start + batch_size && epoch <= end_epoch; epoch++)); do
                local gpu_index=$(((epoch - 1) % 4))
                CUDA_VISIBLE_DEVICES=${gpu_index} nohup python RL/DiHFT/ablation/converge_steps_sun/EarnHFT_PES_test.py \
                    --dataset_name "${dataset_name}" \
                    --max_holding_number "${max_holding_number}" \
                    --epoch_number "${epoch}" \
                    --beta "${beta}" \
                    >"${log_EarnHFT_PES}/beta_${beta}_epoch_${epoch}.log" 2>&1 &
                pids+=($!) # 将每个后台进程的PID添加到数组中
            done

            # 等待当前批次的所有PID
            for pid in "${pids[@]}"; do
                wait "$pid"
            done

            echo "Batch starting from epoch ${batch_start} completed."
        done
    done

    #EarnHFT Random
    for ((batch_start = start_epoch; batch_start <= end_epoch; batch_start += batch_size)); do
        # 保存PID的数组
        pids=()

        # 内部循环，每批次运行epochs_per_batch个Epoch
        for ((epoch = batch_start; epoch < batch_start + batch_size && epoch <= end_epoch; epoch++)); do
            local gpu_index=$(((epoch - 1) % 4))
            CUDA_VISIBLE_DEVICES=${gpu_index} nohup python RL/DiHFT/ablation/converge_steps_sun/EarnHFT_random_test.py \
                --dataset_name "${dataset_name}" \
                --max_holding_number "${max_holding_number}" \
                --epoch_number "${epoch}" \
                >"${log_EarnHFT_random}/epoch_${epoch}.log" 2>&1 &
            pids+=($!) # 将每个后台进程的PID添加到数组中
        done

        # 等待当前批次的所有PID
        for pid in "${pids[@]}"; do
            wait "$pid"
        done

        echo "Batch starting from epoch ${batch_start} completed."
    done
    #FineFT wo pretrain
    for ((batch_start = start_epoch; batch_start <= end_epoch; batch_start += batch_size)); do
        # 保存PID的数组
        pids=()

        # 内部循环，每批次运行epochs_per_batch个Epoch
        for ((epoch = batch_start; epoch < batch_start + batch_size && epoch <= end_epoch; epoch++)); do
            local gpu_index=$(((epoch - 1) % 4))
            CUDA_VISIBLE_DEVICES=${gpu_index} nohup python RL/DiHFT/ablation/converge_steps_sun/FineFT_wo_pretrain_test.py \
                --dataset_name "${dataset_name}" \
                --max_holding_number "${max_holding_number}" \
                --epoch_num "${epoch}" \
                >"${log_FineFT_wo_pretrain}/epoch_${epoch}.log" 2>&1 &
            pids+=($!) # 将每个后台进程的PID添加到数组中
        done

        # 等待当前批次的所有PID
        for pid in "${pids[@]}"; do
            wait "$pid"
        done

        echo "Batch starting from epoch ${batch_start} completed."
    done

    # FineFT
    for ((batch_start = start_epoch; batch_start <= end_epoch; batch_start += batch_size)); do
        # 保存PID的数组
        pids=()

        # 内部循环，每批次运行epochs_per_batch个Epoch
        for ((epoch = batch_start; epoch < batch_start + batch_size && epoch <= end_epoch; epoch++)); do
            local gpu_index=$(((epoch - 1) % 4))
            CUDA_VISIBLE_DEVICES=${gpu_index} nohup python RL/DiHFT/ablation/converge_steps_sun/FineFT_test.py \
                --dataset_name "${dataset_name}" \
                --max_holding_number "${max_holding_number}" \
                --epoch_num "${epoch}" \
                >"${log_FineFT}/epoch_${epoch}.log" 2>&1 &
            pids+=($!) # 将每个后台进程的PID添加到数组中
        done

        # 等待当前批次的所有PID
        for pid in "${pids[@]}"; do
            wait "$pid"
        done

        echo "Batch starting from epoch ${batch_start} completed."
    done

}

run_test_ablation BNBUSDT 100 1 100 100

run_test_ablation BTCUSDT 8 1 100 100

run_test_ablation DOTUSDT 6000 1 100 100

run_test_ablation ETHUSDT 160 1 100 100
