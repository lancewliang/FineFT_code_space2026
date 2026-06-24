source data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh

run_commodity_logged_step() {
    local log_dir=$1
    local symbol=$2
    local target_freq=$3
    local start_date=$4
    local end_date=$5
    local step_name=$6
    shift 6

    local step_log_dir="${log_dir}/steps"
    local step_log="${step_log_dir}/${symbol}_${target_freq}_${start_date}_${end_date}_${step_name}.log"
    mkdir -p "${step_log_dir}"

    echo "[commodity][${step_name}] start -> ${step_log}"
    local had_errexit=0
    case "$-" in
        *e*) had_errexit=1 ;;
    esac
    set +e
    ( set -e; "$@" ) >"${step_log}" 2>&1
    local status=$?
    if [ "${had_errexit}" -eq 1 ]; then
        set -e
    else
        set +e
    fi
    if [ "${status}" -eq 0 ]; then
        echo "[commodity][${step_name}] success -> ${step_log}"
    else
        echo "[commodity][${step_name}] failed(${status}) -> ${step_log}"
        return "${status}"
    fi
}

commodity_downscale_outputs_exist() {
    local root_path=$1
    local symbol=$2
    local target_freq=$3
    local date=$4
    local output_root="${root_path}/PREPROCESS_DATASET/commodity-futures"

    [ -f "${output_root}/BASE_FEATURE/${symbol}/${target_freq}/${date}.feather" ] \
        && [ -f "${output_root}/DOWNSCALE_ORDERBOOK_25/${symbol}/${target_freq}/${date}.feather" ]
}

commodity_cross_section_outputs_exist() {
    local root_path=$1
    local symbol=$2
    local target_freq=$3
    local date=$4
    local output_root="${root_path}/PREPROCESS_DATASET/commodity-futures/CROSS_SECTION"

    [ -f "${output_root}/KLINE_FEATURE/${symbol}/${target_freq}/${date}.feather" ] \
        && [ -f "${output_root}/QUOTES_FEATURE/${symbol}/${target_freq}/${date}.feather" ] \
        && [ -f "${output_root}/SNAPSHOT_FEATURE/${symbol}/${target_freq}/${date}.feather" ]
}

run_commodity_stitch_main_contract() {
    local root_path=$1
    local commodity_name=${2:-燃料油}
    local start_date=$3
    local end_date=$4
    local symbol=${5:-fu}
    local output_dir="${root_path}/PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/${symbol}"

    mkdir -p "${output_dir}"
    PYTHONPATH="${root_path}/data_preprocess" python -m operator_futures.commodity.stitch_main_contract \
        --raw_root "${root_path}/data/原始下载" \
        --commodity_name "${commodity_name}" \
        --start_date "${start_date}" \
        --end_date "${end_date}" \
        --symbol "${symbol}" \
        --output_dir "${output_dir}"
}

run_commodity_downscale_continuous_by_trading_day() {
    local root_path=$1
    local continuous_dir=$2
    local start_date=$3
    local end_date=$4
    local target_freq=$5
    local symbol=${6:-fu}
    local output_root="${root_path}/PREPROCESS_DATASET/commodity-futures"

    PYTHONPATH="${root_path}/data_preprocess" python -m operator_futures.commodity.downscale_continuous_by_trading_day \
        --input_dir "${continuous_dir}" \
        --start_date "${start_date}" \
        --end_date "${end_date}" \
        --output_root "${output_root}" \
        --target_freq "${target_freq}" \
        --symbol "${symbol}" \
        --depth 5
}

run_commodity_cross_section_process() {
    local start_date=$1
    local end_date=$2
    local max_processes=$3
    local target_freq=$4
    local symbol=$5
    local root_path=$6

    local current_date
    current_date=$(date -I -d "$start_date")
    local process_count=0
    while [ "$current_date" != "$end_date" ]; do
        if ! commodity_downscale_outputs_exist "$root_path" "$symbol" "$target_freq" "$current_date"; then
            echo "Skipping commodity cross-section date with missing downscale outputs: date=${current_date}"
            current_date=$(date -I -d "$current_date + 1 day")
            continue
        fi
        local log_dir="log_futures/downscale/cross_section/${target_freq}/${symbol}"
        mkdir -p "$log_dir"
        PYTHONPATH="${root_path}/data_preprocess" nohup python -u data_preprocess/operator_futures/cross_section/create_feature.py \
            --symbols "$symbol" \
            --target_freq "$target_freq" \
            --date "$current_date" \
            --root_path "$root_path" \
            --data_path "PREPROCESS_DATASET/commodity-futures/" \
            --save_path "PREPROCESS_DATASET/commodity-futures/CROSS_SECTION" \
            --market_type commodity_futures \
            --orderbook_depth 5 \
            >"$log_dir/$current_date.log" 2>&1 &
        local pid=$!
        let process_count=process_count+1
        if [ "$process_count" -eq "$max_processes" ]; then
            wait "$pid" || return $?
            process_count=0
        fi
        current_date=$(date -I -d "$current_date + 1 day")
    done
    wait || return $?
}

run_commodity_scale_save() {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local root_path=$5

    PYTHONPATH="${root_path}/data_preprocess" python -u data_preprocess/operator_futures/scale_describe_save/scale_save.py \
        --symbols "$symbol" \
        --target_freq "$target_freq" \
        --start_date "$start_date" \
        --end_date "$end_date" \
        --root_path "$root_path" \
        --data_path "PREPROCESS_DATASET/commodity-futures/IC_RESULT" \
        --save_path "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/" \
        --market_type commodity_futures \
        --orderbook_depth 5 \
        --ic_choice ic
}

run_commodity_merge_process() {
    local start_date=$1
    local end_date=$2
    local max_processes=$3
    local target_freq=$4
    local symbol=$5
    local root_path=$6

    local current_date
    current_date=$(date -I -d "$start_date")
    local process_count=0
    while [ "$current_date" != "$end_date" ]; do
        if ! commodity_downscale_outputs_exist "$root_path" "$symbol" "$target_freq" "$current_date" \
            || ! commodity_cross_section_outputs_exist "$root_path" "$symbol" "$target_freq" "$current_date"; then
            echo "Skipping commodity merge date with missing feature outputs: date=${current_date}"
            current_date=$(date -I -d "$current_date + 1 day")
            continue
        fi
        local log_dir="log_futures/merge/${target_freq}/${symbol}"
        mkdir -p "$log_dir"
        PYTHONPATH="${root_path}/data_preprocess" nohup python -u data_preprocess/operator_futures/merge_concat/merge.py \
            --symbols "$symbol" \
            --target_freq "$target_freq" \
            --date "$current_date" \
            --root_path "$root_path" \
            --data_path "PREPROCESS_DATASET/commodity-futures/" \
            --save_path "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT" \
            >"$log_dir/$current_date.log" 2>&1 &
        local pid=$!
        let process_count=process_count+1
        if [ "$process_count" -eq "$max_processes" ]; then
            wait "$pid" || return $?
            process_count=0
        fi
        current_date=$(date -I -d "$current_date + 1 day")
    done
    wait || return $?
}

run_commodity_concat_process() {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local root_path=$5

    PYTHONPATH="${root_path}/data_preprocess" python -u data_preprocess/operator_futures/merge_concat/concat.py \
        --symbols "$symbol" \
        --target_freq "$target_freq" \
        --start_date "$start_date" \
        --end_date "$end_date" \
        --root_path "$root_path" \
        --data_path "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT" \
        --save_path "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT"
}

run_commodity_time_feature() {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local root_path=$5

    PYTHONPATH="${root_path}/data_preprocess" python -u data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py \
        --symbols "$symbol" \
        --target_freq "$target_freq" \
        --start_date "$start_date" \
        --end_date "$end_date" \
        --root_path "$root_path" \
        --data_path "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/CONCAT_FEATURE/" \
        --save_path "PREPROCESS_DATASET/commodity-futures/TIME_FEATURE/" \
        --orderbook_depth 5
}

run_commodity_merge_and_clean() {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local root_path=$5

    PYTHONPATH="${root_path}/data_preprocess" python -u data_preprocess/operator_futures/merge_all/merge_clean.py \
        --symbols "$symbol" \
        --target_freq "$target_freq" \
        --start_date "$start_date" \
        --end_date "$end_date" \
        --root_path "$root_path" \
        --data_path_1 "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/CONCAT_FEATURE" \
        --data_path_2 "PREPROCESS_DATASET/commodity-futures/TIME_FEATURE" \
        --save_path "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE"
}

run_commodity_ic_correlation() {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local root_path=$5

    PYTHONPATH="${root_path}/data_preprocess" python -u data_preprocess/operator_futures/feature_selection/ic_correlation.py \
        --symbols "$symbol" \
        --target_freq "$target_freq" \
        --start_date "$start_date" \
        --end_date "$end_date" \
        --root_path "$root_path" \
        --data_path "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/" \
        --save_path "PREPROCESS_DATASET/commodity-futures/IC_RESULT/" \
        --market_type commodity_futures \
        --orderbook_depth 5
}

run_commodity_full_process() {
    local root_path=$1
    local start_date=$2
    local end_date=$3
    local target_freq=${4:-5min}
    local symbol=${5:-fu}
    local commodity_name=${6:-燃料油}
    local max_processes=${7:-4}

    local log_dir="${LOG_DIR:-${root_path}/log_futures/ticker_result/commodity}"

    # run_commodity_logged_step \
    #     "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
    #     "stitch_main_contract" \
    #     run_commodity_stitch_main_contract "$root_path" "$commodity_name" "$start_date" "$end_date" "$symbol"
    local continuous_dir="${root_path}/PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/${symbol}"
    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "downscale_continuous_by_trading_day" \
        run_commodity_downscale_continuous_by_trading_day "$root_path" "$continuous_dir" "$start_date" "$end_date" "$target_freq" "$symbol"
    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "cross_section" \
        run_commodity_cross_section_process "$start_date" "$end_date" "$max_processes" "$target_freq" "$symbol" "$root_path"
    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "merge" \
        run_commodity_merge_process "$start_date" "$end_date" "$max_processes" "$target_freq" "$symbol" "$root_path"
    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "concat" \
        run_commodity_concat_process "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"
    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "time_feature" \
        run_commodity_time_feature "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"
    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "merge_clean" \
        run_commodity_merge_and_clean "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"
    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "ic_correlation" \
        run_commodity_ic_correlation "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"
    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "scale_save" \
        run_commodity_scale_save "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"
}
