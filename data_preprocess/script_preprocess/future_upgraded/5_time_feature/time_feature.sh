function run_feature_creation_multiprocessing {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local ROOTPATH=$5
    local logdir="log_futures/time_operator/${target_freq}/${symbol}"

    # Check if the log directory exists, create if not
    if [ ! -d "$logdir" ]; then
        mkdir -p "$logdir"
    fi

    # Execute the Python script in the background with multiprocessing
    nohup python -u operator_futures/time_operator/create_feature_multi_processing.py \
        --symbols $symbol --target_freq $target_freq --start_date $start_date --end_date $end_date --root_path $ROOTPATH \
        >"$logdir/${start_date}_${end_date}.log" 2>&1 &
    local pid=$!
    echo "All processes of creating time-related features have been initiated."
    wait $pid

    echo "All processes of creating time-related features have been completed."
}
