#!/bin/bash
function run_test_agent_index {
    local dataset_name=$1
    local max_holding_number=$2
    local epoch_start=$3
    local epoch_end=$4
    local base_path=$5
    local experiment_name=$6
    local ensemble_number=${ENSEMBLE_NUMBER:-7}
    local label_types=("slope" "volatility")
    if [ -n "$LABEL_TYPE" ]; then
        label_types=("$LABEL_TYPE")
    fi
    local result_path=${RESULT_PATH:-result/DiHFT/low_level}
    local max_parallel=${MAX_PARALLEL:-1}
    ROOTPATH=${ROOTPATH:-$(pwd)}
    cd "$ROOTPATH"
    export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

    local failed=0
    # 保存PID的数组
    local pids=()

    # 循环执行 epoch，再循环 label_type
    for epoch in $(seq $epoch_start $epoch_end); do
        local gpu_index=$(((epoch - 1) % 4))
        for label_type in "${label_types[@]}"; do
            # 检查并创建日志目录
            log_dir="log/DiHFT/${dataset_name}/low_level/test/${experiment_name}/${label_type}"
            mkdir -p "${log_dir}"

            nohup python FineFT/RL/DiHFT/low_level/test_agent_index.py \
                --base_path "${base_path}" \
                --dataset_name "${dataset_name}" --experiment_name "${experiment_name}" \
                --result_path "${result_path}" \
                --max_holding_number "${max_holding_number}" --initial_wallet_balance 10000 --order_book_depth 5 \
                --epoch_num "${epoch}" --position_choices 5 --N "${ensemble_number}" --transcation_cost 0.0004 --short_estimated_rate 0 --long_estimated_rate 0 \
                --allow_reverse_position \
                --label_type "${label_type}" \
                --save_trading_detail_csv \
                >"${log_dir}/epoch_${epoch}.log" 2>&1 &
            pids+=($!) # 将每个后台进程的PID添加到数组中

            echo "${dataset_name} ${experiment_name} ${max_holding_number} label_type ${label_type} epoch ${epoch} started."

            if ((${#pids[@]} >= max_parallel)); then
                for pid in "${pids[@]}"; do
                    wait "$pid" || failed=1
                done
                pids=()
                if ((failed)); then
                    echo "Failed to generate one or more epoch detail CSVs for ${label_type}. See ${log_dir}." >&2
                    return 1
                fi
            fi
        done
    done

    # 等待最后一批不足 max_parallel 的进程
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if ((failed)); then
        echo "Failed to generate one or more epoch detail CSVs. See log dir." >&2
        return 1
    fi

    echo "${dataset_name} ${experiment_name} ${max_holding_number} testing completed successfully."
}

function run_ddqn_context {
    run_test_agent_index "$1" "$2" "$3" "$4" "$5" "$6" || return 1
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
#run_ddqn_context fu 1 30 100 dataset/5min 5min_nstep6_costw5

DATASET_NAME=${DATASET_NAME:-fu}
MAX_HOLDING_NUMBER=${MAX_HOLDING_NUMBER:-2}
EPOCH_START=${EPOCH_START:-50}
EPOCH_END=${EPOCH_END:-100}
BASE_PATH=${BASE_PATH:-dataset/30min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-30min_multi_persistence}

run_ddqn_context "${DATASET_NAME}" "${MAX_HOLDING_NUMBER}" "${EPOCH_START}" "${EPOCH_END}" "${BASE_PATH}" "${EXPERIMENT_NAME}"

# BTCUSDT
# run_ddqn_context BTCUSDT 8 45 100

# ETHUSDT
# run_ddqn_context ETHUSDT 160 1 100

# # BNBUSDT
# run_ddqn_context BNBUSDT 100 1 100

# #DOTUSDT
# run_ddqn_context DOTUSDT 6000 1 100
