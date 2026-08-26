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
    export PYTHONPATH="${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

    local failed=0

    for label_type in "${label_types[@]}"; do
        # 检查并创建日志目录
        log_dir="log/DiHFT/${dataset_name}/low_level/test/${experiment_name}/${label_type}"
        mkdir -p "${log_dir}"
        # 保存PID的数组
        pids=()

        for epoch in $(seq "$epoch_start" "$epoch_end"); do
            nohup python FineFT/RL/DiHFT/low_level/test_agent_index.py \
                --base_path "${base_path}" \
                --dataset_name "${dataset_name}" --experiment_name "${experiment_name}" \
                --result_path "${result_path}" \
                --max_holding_number "${max_holding_number}" --initial_wallet_balance 25000 --order_book_depth 5 \
                --epoch_num "${epoch}" --position_choices 11 --N "${ensemble_number}" --transcation_cost 0.0005 --short_estimated_rate 0 --long_estimated_rate 0 \
                --allow_reverse_position \
                --label_type "${label_type}" \
                --save_trading_detail_csv \
                >"${log_dir}/epoch_${epoch}.log" 2>&1 &
            pids+=($!)

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

        for pid in "${pids[@]}"; do
            wait "$pid" || failed=1
        done
        if ((failed)); then
            echo "Failed to generate one or more epoch detail CSVs for ${label_type}. See ${log_dir}." >&2
            return 1
        fi

        echo "${dataset_name} ${experiment_name} ${max_holding_number} label_type ${label_type} testing completed successfully."
    done
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

    for epoch in $(seq "$epoch_start" "$epoch_end"); do
        nohup python RL/DiHFT/low_level/test_agent_average.py \
            --dataset_name "${dataset_name}" \
            --max_holding_number "${max_holding_number}" \
            --epoch_num "${epoch}" \
            >"${log_dir}/bin_${bin_size}_epoch_${epoch}.log" 2>&1 &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "${dataset_name} ${max_holding_number} ${bin_size} All processes completed successfully."
}

DATASET_NAME=${DATASET_NAME:-fu}
MAX_HOLDING_NUMBER=${MAX_HOLDING_NUMBER:-5}
EPOCH_START=${EPOCH_START:-60}
EPOCH_END=${EPOCH_END:-125}
BASE_PATH=${BASE_PATH:-dataset/10min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_multi}

run_ddqn_context "${DATASET_NAME}" "${MAX_HOLDING_NUMBER}" "${EPOCH_START}" "${EPOCH_END}" "${BASE_PATH}" "${EXPERIMENT_NAME}"
