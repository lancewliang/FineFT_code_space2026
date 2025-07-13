source script/future_upgraded/1_downscale/derticker.sh
source script/future_upgraded/1_downscale/orderbook.sh
source script/future_upgraded/2_base_feature/base_feature.sh
source script/future_upgraded/3_cross_section/cross_section.sh
source script/future_upgraded/4_merge_concat/merge_concat.sh
source script/future_upgraded/5_time_feature/time_feature.sh
source script/future_upgraded/6_merge_all/merge.sh
source script/future_upgraded/7_feature_selection/ic_correlation.sh
source script/future_upgraded/8_scale_save/scale.sh


function run_all_process_from_base {
    local base_freq=$1
    local target_freq=$2
    local start_date=$3
    local end_date=$4
    local symbol=$5
    local max_processes_1=$6
    local max_processes_2=$7
    local ROOTPATH=$8

    run_derivative_ticker_downscaling_with_base $start_date $end_date $max_processes_1 $target_freq $base_freq $symbol $ROOTPATH
    run_orderbook_downscaling_with_base $start_date $end_date $max_processes_2 $target_freq $base_freq $symbol $ROOTPATH
    run_downscale_process $start_date $end_date $max_processes_1 $target_freq $symbol $ROOTPATH
    run_cross_section_process $start_date $end_date $max_processes_1 $target_freq $symbol $ROOTPATH
    run_merge_process $start_date $end_date $max_processes_1 $target_freq $symbol $ROOTPATH
    run_concat_process $target_freq $start_date $end_date $symbol $ROOTPATH
    run_feature_creation_multiprocessing $target_freq $start_date $end_date $symbol $ROOTPATH
    run_merge_and_clean $target_freq $start_date $end_date $symbol $ROOTPATH
    run_ic_correlation $target_freq $start_date $end_date $symbol $ROOTPATH
    run_scale_save $target_freq $start_date $end_date $symbol $ROOTPATH
}

function run_to_cross_section_from_base{
    local base_freq=$1
    local target_freq=$2
    local start_date=$3
    local end_date=$4
    local symbol=$5
    local max_processes_1=$6
    local max_processes_2=$7
    local ROOTPATH=$8
    run_derivative_ticker_downscaling_with_base $start_date $end_date $max_processes_1 $target_freq $base_freq $symbol $ROOTPATH
    run_orderbook_downscaling_with_base $start_date $end_date $max_processes_2 $target_freq $base_freq $symbol $ROOTPATH
    run_downscale_process $start_date $end_date $max_processes_1 $target_freq $symbol $ROOTPATH
    run_cross_section_process $start_date $end_date $max_processes_1 $target_freq $symbol $ROOTPATH
}