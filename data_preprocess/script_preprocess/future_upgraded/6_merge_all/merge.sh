function run_merge_and_clean {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local ROOTPATH=$5
    local logdir="log_futures/merge_all/${target_freq}/${symbol}"

    # Check if the log directory exists, create if not
    if [ ! -d "$logdir" ]; then
        mkdir -p "$logdir"
    fi

    # Execute the Python script in the background
    nohup python -u operator_futures/merge_all/merge_clean.py \
        --symbols $symbol --target_freq $target_freq --start_date $start_date --end_date $end_date --root_path $ROOTPATH \
        >"$logdir/${start_date}_${end_date}.log" 2>&1 &
    local pid=$!
    echo "All processes of merging time-related features have been initiated."
    wait $pid

    echo "All processes of merging time-related features have been completed."
}
