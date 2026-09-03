source data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh

COMMODITY_FU_FEATURE_BLACKLIST=(
    open
    high
    low
    open_interest
    vwap
    awap
    twap
    open_buy
    open_sell
    high_buy
    high_sell
    low_buy
    low_sell
    close_buy
    close_sell
    vwap_buy
    vwap_sell
    awap_buy
    awap_sell
    twap_buy
    twap_sell
    tradeval
    volume_buy
    volume_sell
    buy_volume
    sell_volume
    tradeval_buy
    tradeval_sell
    midprice
    wap_1
    wap_2
    buy_wap
    sell_wap
    buy_volume_oe
    sell_volume_oe
)

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
    local contract=${5:-}
    local output_root="${root_path}/PREPROCESS_DATASET/commodity-futures"
    local symbol_path="${symbol}"
    if [ -n "${contract}" ]; then
        symbol_path="${symbol}/${contract}"
    fi

    [ -f "${output_root}/BASE_FEATURE/${symbol_path}/${target_freq}/${date}.feather" ] \
        && [ -f "${output_root}/DOWNSCALE_ORDERBOOK_25/${symbol_path}/${target_freq}/${date}.feather" ]
}

commodity_cross_section_outputs_exist() {
    local root_path=$1
    local symbol=$2
    local target_freq=$3
    local date=$4
    local contract=${5:-}
    local output_root="${root_path}/PREPROCESS_DATASET/commodity-futures/CROSS_SECTION"
    local symbol_path="${symbol}"
    if [ -n "${contract}" ]; then
        symbol_path="${symbol}/${contract}"
    fi

    [ -f "${output_root}/KLINE_FEATURE/${symbol_path}/${target_freq}/${date}.feather" ] \
        && [ -f "${output_root}/QUOTES_FEATURE/${symbol_path}/${target_freq}/${date}.feather" ] \
        && [ -f "${output_root}/SNAPSHOT_FEATURE/${symbol_path}/${target_freq}/${date}.feather" ]
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
    local summary_path=$2
    local target_freq=$3
    local symbol=${4:-fu}
    local contract=${5:-}
    local output_root="${root_path}/PREPROCESS_DATASET/commodity-futures"
    local contract_args=()
    if [ -n "${contract}" ]; then
        contract_args=(--contract "${contract}")
    fi

    PYTHONPATH="${root_path}/data_preprocess" python -m operator_futures.commodity.downscale_continuous_by_trading_day \
        --summary "${summary_path}" \
        --output_root "${output_root}" \
        --target_freq "${target_freq}" \
        --symbol "${symbol}" \
        --depth 5 --max_workers 7 \
        "${contract_args[@]}"
}

run_commodity_summary_contracts() {
    local summary_path=$1
    local root_path="${ROOTPATH:-$(pwd)}"
    local pythonpath="${root_path}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}"
    PYTHONPATH="${pythonpath}" python - "${summary_path}" <<'PY'
import sys
from pathlib import Path

from operator_futures.commodity.main_contract import load_main_contract_summary

summary = load_main_contract_summary(Path(sys.argv[1]))
for item in summary.contracts:
    print(item.contract)
PY
}

run_commodity_cross_section_process() {
    local start_date=$1
    local end_date=$2
    local max_processes=$3
    local target_freq=$4
    local symbol=$5
    local root_path=$6
    local contract=${7:-}
    local contract_args=()
    local log_symbol="${symbol}"
    if [ -n "${contract}" ]; then
        contract_args=(--contract "${contract}")
        log_symbol="${symbol}/${contract}"
    fi

    local current_date
    current_date=$(date -I -d "$start_date")
    local process_count=0
    while [ "$current_date" != "$end_date" ]; do
        if ! commodity_downscale_outputs_exist "$root_path" "$symbol" "$target_freq" "$current_date" "$contract"; then
            echo "Skipping commodity cross-section date with missing downscale outputs: symbol=${symbol} contract=${contract} date=${current_date}"
            current_date=$(date -I -d "$current_date + 1 day")
            continue
        fi
        local log_dir="log_futures/downscale/cross_section/${target_freq}/${log_symbol}"
        mkdir -p "$log_dir"
        PYTHONPATH="${root_path}/data_preprocess" nohup python -u data_preprocess/operator_futures/cross_section/create_feature.py \
            --symbols "$symbol" \
            "${contract_args[@]}" \
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

BASE_TIME_FEATURE_COLUMNS=(
    trading_minute_progress
    morning_session
    afternoon_session
    night_session
    is_opening_30m
    is_closing_30m
    is_session_first_bar
    is_session_last_bar
    contract_month_sin
    contract_month_cos
    contract_life_remaining_ratio
)

CROSS_MONTH_FEATURE_COLUMNS=(
    cm_contract_role_main
    cm_contract_role_sub
    cm_contract_role_other
    cm_current_main_log_price_ratio
    cm_current_main_relative_price_spread
    cm_current_main_volume_share_current
    cm_current_main_open_interest_share_current
    cm_current_sub_log_price_ratio
    cm_current_sub_relative_price_spread
    cm_current_sub_volume_share_current
    cm_current_sub_open_interest_share_current
    cm_main_sub_log_price_ratio
    cm_main_sub_relative_price_spread
    cm_main_sub_volume_share_sub
    cm_main_sub_open_interest_share_sub
    cm_m1_m2_open_interest_share_m2
    cm_m2_m3_open_interest_share_m3
    cm_main_sub_log_price_spread_velocity_10m
    cm_open_interest_shift_speed_10m
    cm_m1_m2_log_price_spread_velocity_10m
    cm_m2_m3_log_price_spread_velocity_10m
    cm_m1_m2_m3_butterfly_spread_velocity_10m
)

PRICE_LIMIT_RATIO_FEATURE_COLUMNS=(
    limit_up_single_sided_ratio
    limit_down_single_sided_ratio
    limit_up_ask_depth_ratio_5
    limit_down_bid_depth_ratio_5
    limit_depth_imbalance_ratio_5
    prev_day_limit_up_single_sided_ratio
    prev_day_limit_down_single_sided_ratio
    prev_2_day_limit_up_single_sided_ratio
    prev_2_day_limit_down_single_sided_ratio
)

MIXED_FREQUENCY_FEATURE_COLUMNS=(
    prev_day_return
    prev_day_range_pct
    prev_day_body_pct
    prev_day_upper_shadow_pct
    prev_day_lower_shadow_pct
    prev_day_close_position
    prev_day_body_to_range
    prev_day_upper_shadow_to_range
    prev_day_lower_shadow_to_range
    prev_day_vwap_deviation_pct
    prev_day_twap_deviation_pct
    prev_day_trade_up_ratio
    prev_day_trade_down_ratio
    prev_day_trade_imbalance
    prev_day_open_interest_change
    prev_day_turnover_rate
    prev_week_return
    prev_week_range_pct
    prev_week_body_pct
    prev_week_close_position
    prev_week_body_to_range
    prev_week_upper_shadow_to_range
    prev_week_lower_shadow_to_range
    prev_week_vwap_deviation_pct
    prev_week_twap_deviation_pct
    prev_week_trade_up_ratio
    prev_week_trade_down_ratio
    prev_week_trade_imbalance
    prev_week_open_interest_change
    prev_week_turnover_rate
)

run_commodity_scale_save() {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local root_path=$5

    PYTHONPATH="${root_path}/data_preprocess" python -u data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py \
        --symbols "$symbol" \
        --target_freq "$target_freq" \
        --start_date "$start_date" \
        --end_date "$end_date" \
        --root_path "$root_path" \
        --data_path "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST" \
        --save_path "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/" \
        --market_type commodity_futures \
        --orderbook_depth 5 \
        --feature_list_path "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/${target_freq}/${symbol}/train/state_features.npy" \
        --passthrough_features "${BASE_TIME_FEATURE_COLUMNS[@]}"
}

run_commodity_merge_process() {
    local start_date=$1
    local end_date=$2
    local max_processes=$3
    local target_freq=$4
    local symbol=$5
    local root_path=$6
    local contract=${7:-}
    local contract_args=()
    local log_symbol="${symbol}"
    if [ -n "${contract}" ]; then
        contract_args=(--contract "${contract}")
        log_symbol="${symbol}/${contract}"
    fi

    local current_date
    current_date=$(date -I -d "$start_date")
    local process_count=0
    while [ "$current_date" != "$end_date" ]; do
        if ! commodity_downscale_outputs_exist "$root_path" "$symbol" "$target_freq" "$current_date" "$contract" \
            || ! commodity_cross_section_outputs_exist "$root_path" "$symbol" "$target_freq" "$current_date" "$contract"; then
            echo "Skipping commodity merge date with missing feature outputs: symbol=${symbol} contract=${contract} date=${current_date}"
            current_date=$(date -I -d "$current_date + 1 day")
            continue
        fi
        local log_dir="log_futures/merge/${target_freq}/${log_symbol}"
        mkdir -p "$log_dir"
        PYTHONPATH="${root_path}/data_preprocess" nohup python -u data_preprocess/operator_futures/merge_concat/merge.py \
            --symbols "$symbol" \
            "${contract_args[@]}" \
            --target_freq "$target_freq" \
            --date "$current_date" \
            --root_path "$root_path" \
            --data_path "PREPROCESS_DATASET/commodity-futures/" \
            --save_path "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT" \
            --require_mixed_frequency_feature \
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

run_commodity_cross_month_feature_process() {
    local start_date=$1
    local end_date=$2
    local max_processes=$3
    local target_freq=$4
    local symbol=$5
    local root_path=$6
    local summary_path=$7
    local contract=$8

    local current_date
    current_date=$(date -I -d "$start_date")
    local process_count=0
    while [ "$current_date" != "$end_date" ]; do
        if ! commodity_downscale_outputs_exist "$root_path" "$symbol" "$target_freq" "$current_date" "$contract"; then
            echo "Skipping commodity cross-month feature date with missing downscale outputs: symbol=${symbol} contract=${contract} date=${current_date}"
            current_date=$(date -I -d "$current_date + 1 day")
            continue
        fi
        local log_dir="log_futures/cross_month_feature/${target_freq}/${symbol}/${contract}"
        mkdir -p "$log_dir"
        PYTHONPATH="${root_path}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}" nohup python -u -m operator_futures.commodity.cross_month_feature \
            --symbol "$symbol" \
            --contract "$contract" \
            --target_freq "$target_freq" \
            --date "$current_date" \
            --root_path "$root_path" \
            --summary_path "$summary_path" \
            --data_path "PREPROCESS_DATASET/commodity-futures" \
            --save_path "PREPROCESS_DATASET/commodity-futures/CROSS_MONTH_FEATURE" \
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

run_commodity_daily_base_feature_process() {
    local start_date=$1
    local end_date=$2
    local target_freq=$3
    local symbol=$4
    local root_path=$5
    local contract=$6

    local log_dir="log_futures/daily_base_feature/${target_freq}/${symbol}/${contract}"
    mkdir -p "$log_dir"
    PYTHONPATH="${root_path}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}" python -u -m operator_futures.commodity.daily_base_feature \
        --symbol "$symbol" \
        --contract "$contract" \
        --target_freq "$target_freq" \
        --start_date "$start_date" \
        --end_date "$end_date" \
        --root_path "$root_path" \
        --data_path "PREPROCESS_DATASET/commodity-futures" \
        --save_path "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_BASE" \
        >"$log_dir/${start_date}-${end_date}.log" 2>&1
}

run_commodity_weekly_base_feature_process() {
    local start_date=$1
    local end_date=$2
    local target_freq=$3
    local symbol=$4
    local root_path=$5
    local contract=$6

    local log_dir="log_futures/weekly_base_feature/${target_freq}/${symbol}/${contract}"
    mkdir -p "$log_dir"
    PYTHONPATH="${root_path}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}" python -u -m operator_futures.commodity.weekly_base_feature \
        --symbol "$symbol" \
        --contract "$contract" \
        --target_freq "$target_freq" \
        --start_date "$start_date" \
        --end_date "$end_date" \
        --root_path "$root_path" \
        --data_path "PREPROCESS_DATASET/commodity-futures" \
        --save_path "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_BASE" \
        >"$log_dir/${start_date}-${end_date}.log" 2>&1
}

run_commodity_daily_mixed_frequency_feature_process() {
    local start_date=$1
    local end_date=$2
    local target_freq=$4
    local symbol=$5
    local root_path=$6
    local contract=$7

    local log_dir="log_futures/daily_mixed_frequency_feature/${target_freq}/${symbol}/${contract}"
    mkdir -p "$log_dir"
    PYTHONPATH="${root_path}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}" python -u -m operator_futures.commodity.daily_mixed_frequency_feature \
        --symbol "$symbol" \
        --contract "$contract" \
        --target_freq "$target_freq" \
        --start_date "$start_date" \
        --end_date "$end_date" \
        --root_path "$root_path" \
        --data_path "PREPROCESS_DATASET/commodity-futures" \
        --base_path "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_BASE" \
        --save_path "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE" \
        >"$log_dir/${start_date}-${end_date}.log" 2>&1
}

run_commodity_weekly_mixed_frequency_feature_process() {
    local start_date=$1
    local end_date=$2
    local target_freq=$4
    local symbol=$5
    local root_path=$6
    local contract=$7

    local log_dir="log_futures/weekly_mixed_frequency_feature/${target_freq}/${symbol}/${contract}"
    mkdir -p "$log_dir"
    PYTHONPATH="${root_path}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}" python -u -m operator_futures.commodity.weekly_mixed_frequency_feature \
        --symbol "$symbol" \
        --contract "$contract" \
        --target_freq "$target_freq" \
        --start_date "$start_date" \
        --end_date "$end_date" \
        --root_path "$root_path" \
        --data_path "PREPROCESS_DATASET/commodity-futures" \
        --base_path "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_BASE" \
        --save_path "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE" \
        >"$log_dir/${start_date}-${end_date}.log" 2>&1
}

run_commodity_mixed_frequency_feature_process() {
    local start_date=$1
    local end_date=$2
    local max_processes=$3
    local target_freq=$4
    local symbol=$5
    local root_path=$6
    local contract=$7

    local current_date
    current_date=$(date -I -d "$start_date")
    local process_count=0
    while [ "$current_date" != "$end_date" ]; do
        local log_dir="log_futures/mixed_frequency_feature/${target_freq}/${symbol}/${contract}"
        mkdir -p "$log_dir"
        PYTHONPATH="${root_path}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}" nohup python -u -m operator_futures.commodity.mixed_frequency_feature \
            --symbol "$symbol" \
            --contract "$contract" \
            --target_freq "$target_freq" \
            --date "$current_date" \
            --start_date "$start_date" \
            --end_date "$end_date" \
            --root_path "$root_path" \
            --feature_path "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE" \
            --save_path "PREPROCESS_DATASET/commodity-futures/MIXED_FREQUENCY_FEATURE" \
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
    local contract=${6:-}
    local contract_args=()
    if [ -n "${contract}" ]; then
        contract_args=(--contract "${contract}")
    fi

    PYTHONPATH="${root_path}/data_preprocess" python -u data_preprocess/operator_futures/merge_concat/concat.py \
        --symbols "$symbol" \
        "${contract_args[@]}" \
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
    local contract=${6:-}
    local contract_args=()
    if [ -n "${contract}" ]; then
        contract_args=(--contract "${contract}")
    fi

    PYTHONPATH="${root_path}/data_preprocess" python -u data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py \
        --symbols "$symbol" \
        "${contract_args[@]}" \
        --target_freq "$target_freq" \
        --start_date "$start_date" \
        --end_date "$end_date" \
        --root_path "$root_path" \
        --data_path "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/CONCAT_FEATURE/" \
        --save_path "PREPROCESS_DATASET/commodity-futures/TIME_FEATURE/" \
        --orderbook_depth 5 \
        --windows "2,6,12,16,24,48,96,192"
}

run_commodity_merge_and_clean() {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local root_path=$5
    local contract=${6:-}
    local contract_args=()
    if [ -n "${contract}" ]; then
        contract_args=(--contract "${contract}")
    fi

    PYTHONPATH="${root_path}/data_preprocess" python -u data_preprocess/operator_futures/merge_all/merge_clean.py \
        --symbols "$symbol" \
        "${contract_args[@]}" \
        --target_freq "$target_freq" \
        --start_date "$start_date" \
        --end_date "$end_date" \
        --root_path "$root_path" \
        --data_path_1 "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/CONCAT_FEATURE" \
        --data_path_2 "PREPROCESS_DATASET/commodity-futures/TIME_FEATURE" \
        --save_path "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE"
}

run_commodity_dataset_split() {
    local summary_path=$1
    local target_freq=$2
    local start_date=$3
    local end_date=$4
    local symbol=$5
    local root_path=$6

    PYTHONPATH="${root_path}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}" python -u -m operator_futures.dataset_split.dataset_split \
        --summary_path "${summary_path}" \
        --input_root "${root_path}/PREPROCESS_DATASET/commodity-futures/ALL_FEATURE" \
        --output_root "${root_path}/PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/${target_freq}" \
        --symbol "${symbol}" \
        --target_freq "${target_freq}" \
        --start_date "${start_date}" \
        --end_date "${end_date}" \
        --train_ratio 5 \
        --valid_ratio 3 \
        --test_ratio 2
}

run_commodity_feature_selection() {
    local stage=$1
    local split_root=$2
    local target_freq=$3
    local symbol=$4
    local root_path=$5
    local regime_bins=${6:-${REGIME_BINS:-3}}
    local feature_blacklist_args=()
    if [ "${#COMMODITY_FU_FEATURE_BLACKLIST[@]}" -gt 0 ]; then
        feature_blacklist_args=(--feature_blacklist "${COMMODITY_FU_FEATURE_BLACKLIST[@]}")
    fi

    local target_regime_bins_args=()
    if [ -n "${TARGET_REGIME_BINS:-}" ]; then
        target_regime_bins_args=(--target_regime_bins ${TARGET_REGIME_BINS})
    fi

    PYTHONPATH="${root_path}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}" python -u -m operator_futures.feature_selection.muti_contract \
        --root_path "${root_path}" \
        --split_path "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST" \
        --save_path "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION" \
        --symbol "${symbol}" --windows_list 1 2 6 12 24 48 96 \
        --target_freq "${target_freq}" \
        --stage "${stage}" \
        --orderbook_depth 5 \
        --regime_bins "${regime_bins}" \
        "${target_regime_bins_args[@]}" \
        --mandatory_state_features "${BASE_TIME_FEATURE_COLUMNS[@]}" "${CROSS_MONTH_FEATURE_COLUMNS[@]}" "${PRICE_LIMIT_RATIO_FEATURE_COLUMNS[@]}" \
        "${feature_blacklist_args[@]}"
}

run_commodity_maintenance_margin_dict() {
    local root_path=$1
    local symbol=${2:-fu}
    local output_root=${3:-dataset}

    PYTHONPATH="${root_path}/data_preprocess" python -u -m operator_futures.commodity.build_maintenance_margin_dict \
        --symbol "${symbol}" \
        --output_root "${root_path}/${output_root}"
}

run_commodity_full_process() {
    local root_path=$1
    local start_date=$2
    local end_date=$3
    local target_freq=${4:-5min}
    local symbol=${5:-fu}
    local commodity_name=${6:-燃料油}
    local max_processes=${7:-4}
    local regime_bins=${8:-${REGIME_BINS:-3}}

    local log_dir="${LOG_DIR:-${root_path}/log_futures/ticker_result/commodity}"

    # run_commodity_logged_step \
    #     "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
    #     "stitch_main_contract" \
    #     run_commodity_stitch_main_contract "$root_path" "$commodity_name" "$start_date" "$end_date" "$symbol"
    local summary_path="${root_path}/PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/${symbol}/main_contract_summary.json"
    # run_commodity_logged_step \
    #     "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
    #     "downscale_continuous_by_trading_day" \
    #     run_commodity_downscale_continuous_by_trading_day "$root_path" "$summary_path" "$target_freq" "$symbol"

    local contract
    # while IFS= read -r contract; do
    #     [ -n "$contract" ] || continue
    #     run_commodity_logged_step \
    #         "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
    #         "cross_section" \
    #         run_commodity_cross_section_process "$start_date" "$end_date" "$max_processes" "$target_freq" "$symbol" "$root_path" "$contract"
    # done < <(run_commodity_summary_contracts "$summary_path")

    # while IFS= read -r contract; do
    #     [ -n "$contract" ] || continue
    #     run_commodity_logged_step \
    #         "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
    #         "daily_base_feature" \
    #         run_commodity_daily_base_feature_process "$start_date" "$end_date" "$target_freq" "$symbol" "$root_path" "$contract"
    #     run_commodity_logged_step \
    #         "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
    #         "weekly_base_feature" \
    #         run_commodity_weekly_base_feature_process "$start_date" "$end_date" "$target_freq" "$symbol" "$root_path" "$contract"
    #     run_commodity_logged_step \
    #         "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
    #         "cross_month_feature" \
    #         run_commodity_cross_month_feature_process "$start_date" "$end_date" "$max_processes" "$target_freq" "$symbol" "$root_path" "$summary_path" "$contract"
    #     run_commodity_logged_step \
    #         "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
    #         "daily_mixed_frequency_feature" \
    #         run_commodity_daily_mixed_frequency_feature_process "$start_date" "$end_date" "$max_processes" "$target_freq" "$symbol" "$root_path" "$contract"
    #     run_commodity_logged_step \
    #         "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
    #         "weekly_mixed_frequency_feature" \
    #         run_commodity_weekly_mixed_frequency_feature_process "$start_date" "$end_date" "$max_processes" "$target_freq" "$symbol" "$root_path" "$contract"
    #     run_commodity_logged_step \
    #         "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
    #         "mixed_frequency_feature" \
    #         run_commodity_mixed_frequency_feature_process "$start_date" "$end_date" "$max_processes" "$target_freq" "$symbol" "$root_path" "$contract"
    #     run_commodity_logged_step \
    #         "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
    #         "merge" \
    #         run_commodity_merge_process "$start_date" "$end_date" "$max_processes" "$target_freq" "$symbol" "$root_path" "$contract"
    #     run_commodity_logged_step \
    #         "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
    #         "concat" \
    #         run_commodity_concat_process "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path" "$contract"
    #     run_commodity_logged_step \
    #         "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
    #         "time_feature" \
    #         run_commodity_time_feature "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path" "$contract"
    #     run_commodity_logged_step \
    #         "$log_dir" "${symbol}_${contract}" "$target_freq" "$start_date" "$end_date" \
    #         "merge_clean" \
    #         run_commodity_merge_and_clean "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path" "$contract"
    # done < <(run_commodity_summary_contracts "$summary_path")

    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "dataset_split" \
        run_commodity_dataset_split "$summary_path" "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"

    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "feature_selection_train" \
        run_commodity_feature_selection "train" "${root_path}/PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/${target_freq}" "$target_freq" "$symbol" "$root_path" "$regime_bins"

    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "feature_selection_valid" \
        run_commodity_feature_selection "valid" "${root_path}/PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/${target_freq}" "$target_freq" "$symbol" "$root_path" "$regime_bins"

    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "scale_save" \
        run_commodity_scale_save "$target_freq" "$start_date" "$end_date" "$symbol" "$root_path"

    run_commodity_logged_step \
        "$log_dir" "$symbol" "$target_freq" "$start_date" "$end_date" \
        "maintenance_margin_dict" \
        run_commodity_maintenance_margin_dict "$root_path" "$symbol" "dataset"
}
