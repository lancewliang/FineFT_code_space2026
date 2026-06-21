run_commodity_downscale_single_day() {
    local input_file=$1
    local output_dir=$2
    local target_freq=$3
    local symbol=$4
    local root_path=$5

    PYTHONPATH="${root_path}/data_preprocess" python -m operator_futures.commodity.downscale_single_day \
        --input "${input_file}" \
        --output_dir "${output_dir}" \
        --symbol "${symbol}" \
        --target_freq "${target_freq}"
}

run_commodity_smoke_fu() {
    local root_path=$1
    local target_freq=${2:-5min}
    local output_dir="${root_path}/PREPROCESS_DATASET/commodity-futures/fu/${target_freq}/sample"

    mkdir -p "${output_dir}"
    run_commodity_downscale_single_day \
        "${root_path}/docs/上海商品交易所/fu2302.csv" \
        "${output_dir}" \
        "${target_freq}" \
        "fu" \
        "${root_path}"
}

# Full fu dataset entry point.
# Raw files are expected under:
#   data/原始下载/燃料油/YYYY/MM/YYYYMMDD/CONTRACT.csv
# The current executable entry provides the verified single-day commodity
# downscale step. The following downstream pipeline stages use the shared
# future_upgraded scripts with commodity-aware flags added in Python:
#   cross-section -> merge/concat -> time feature -> feature selection -> scale/save.
