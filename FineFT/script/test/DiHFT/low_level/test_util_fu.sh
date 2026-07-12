function run_ddqn_context {
    local dataset_name=$1
    local max_holding_number=$2
    local epoch_start=$3
    local epoch_end=$4
    local base_path=$5
    local experiment_name=$6
    ROOTPATH=${ROOTPATH:-$(pwd)}
    cd "$ROOTPATH"
    # 检查并创建日志目录
    log_dir="log/DiHFT/${dataset_name}/low_level/test/${experiment_name}"
    mkdir -p "${log_dir}"
    export PYTHONPATH="${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"
    local max_parallel=4
    # 保存PID的数组
    pids=()

    # 循环执行Epoch 1到50
    for epoch in $(seq $epoch_start $epoch_end); do

        local gpu_index=$(((epoch - 1) % 4))
        nohup python FineFT/RL/DiHFT/low_level/test_agent_index.py \
            --base_path "${base_path}" \
            --dataset_name "${dataset_name}" --experiment_name "${experiment_name}" \
            --max_holding_number "${max_holding_number}" --initial_wallet_balance 10000 --order_book_depth 5 \
            --epoch_num "${epoch}" --position_choices 3 --transcation_cost 0.0005 --short_estimated_rate 0 --long_estimated_rate 0 \
            >"${log_dir}/epoch_${epoch}.log" 2>&1 &
        pids+=($!) # 将每个后台进程的PID添加到数组中

        echo "${dataset_name} ${experiment_name} ${max_holding_number}  epoch ${epoch} completed successfully."

        if ((${#pids[@]} >= max_parallel)); then
            for pid in "${pids[@]}"; do
                wait "$pid"
            done
            pids=()
        fi
    done

    # 等待最后一批不足 max_parallel 的进程
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "${dataset_name} ${experiment_name} ${max_holding_number}  All processes completed successfully."
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
        nohup python RL/DiHFT/low_level/test_agent_average.py \
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
# run_ddqn_context fu 1 1 2 dataset/5min 5min_nstep6_costw5
run_ddqn_context fu 1 30 100 dataset/5min 5min_nstep6_costw5

# # # ETHUSDT
# # run_ddqn_average ETHUSDT 160 1 100
run_ddqn_context fu 1 30 100 dataset/10min 10min_nstep6_costw5

# BTCUSDT
# run_ddqn_context BTCUSDT 8 45 100

# ETHUSDT
# run_ddqn_context ETHUSDT 160 1 100

# # BNBUSDT
# run_ddqn_context BNBUSDT 100 1 100

# #DOTUSDT
# run_ddqn_context DOTUSDT 6000 1 100
