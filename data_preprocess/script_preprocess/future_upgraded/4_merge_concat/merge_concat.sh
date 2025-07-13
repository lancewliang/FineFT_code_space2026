run_merge_process() {
    local START_DATE=$1
    local END_DATE=$2
    local MAX_PROCESSES=$3
    local target_freq=$4
    local symbol=$5
    local ROOTPATH=$6

    local current_date=$(date -I -d "$START_DATE")
    local process_count=0
    if [ $(date -d "$START_DATE" +%s) -gt $(date -d "$END_DATE" +%s) ]; then
        echo "Error: START_DATE ($START_DATE) is later than END_DATE ($END_DATE). Exiting."
        return 1
    fi
    while [ "$current_date" != "$END_DATE" ]; do
        echo "Starting process for $current_date"
        local log_dir="log_futures/merge/$target_freq/$symbol"
        if [ ! -d "$log_dir" ]; then
            mkdir -p "$log_dir"
        fi
        nohup python -u operator_futures/merge_concat/merge.py \
            --symbols $symbol --target_freq $target_freq --date $current_date --root_path $ROOTPATH \
            >"$log_dir/$current_date.log" 2>&1 &
        local pid=$! # Capture the PID of the current process

        let process_count=process_count+1
        if [ $process_count -eq $MAX_PROCESSES ]; then
            local last_pid=$pid # Record the PID of the last process
            wait $last_pid      # Wait for the last process to complete
            process_count=0     # Reset process counter to start a new batch of processes
        fi

        current_date=$(date -I -d "$current_date + 1 day")
    done

    # Wait for all remaining processes to complete
    echo "All processes of initial merge have been initiated."
    wait
    echo "All processes of initial merge have been completed."
}

# Example call (commented out):
# run_merge_process "2023-01-01" "2023-12-31" 100 "10s" "BTCUSDT"
run_concat_process() {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local ROOTPATH=$5

    local logdir="log_futures/concat/$target_freq/$symbol"
    if [ ! -d "$logdir" ]; then
        mkdir -p "$logdir"
    fi

    nohup python -u operator_futures/merge_concat/concat.py \
        --symbols $symbol --target_freq $target_freq --start_date $start_date --end_date $end_date --root_path $ROOTPATH \
        >"$logdir/${start_date}_${end_date}.log" 2>&1 &
    local pid=$!
    echo "All processes of concate have been initiated."
    wait $pid

    echo "All processes of concate have been completed."
}

# Example call (commented out):
# run_concat_process "10s" "2023-01-01" "2023-12-31" "BTCUSDT"
