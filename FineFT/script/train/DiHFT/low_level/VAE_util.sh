function run_rl_vae {
    local dataset_name=$1
    local label_index_total=$2

    # 检查并创建日志目录
    log_dir="log/DiHFT/${dataset_name}/VAE"
    mkdir -p "${log_dir}"

    # 循环执行Label Index 0到Label Index总数 - 1
    for label_index in $(seq 0 $((label_index_total - 1))); do

        local gpu_index=$(((label_index) % 4))
        CUDA_VISIBLE_DEVICES=${gpu_index} nohup python RL/DiHFT/VAE/main.py \
            --dataset_name "${dataset_name}" \
            --label_index "${label_index}" \
            >"${log_dir}/train_label_${label_index}.log" 2>&1 &
    done

    echo "${dataset_name} with labels 0 to $((label_index_total - 1)) processes started successfully."
}

run_rl_vae "BNBUSDT" 5

run_rl_vae "BTCUSDT" 5

run_rl_vae "DOTUSDT" 5

run_rl_vae "ETHUSDT" 5
