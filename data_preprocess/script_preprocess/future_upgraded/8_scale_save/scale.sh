function run_scale_save {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local ROOTPATH=$5
    local logdir="log_futures/scale/${target_freq}/${symbol}"

    # Check if the log directory exists, if not, create it
    if [ ! -d "$logdir" ]; then
        mkdir -p "$logdir"
    fi

    # Execute the Python script in the background
    nohup python -u operator_futures/scale_describe_save/scale_save.py \
        --symbols $symbol --target_freq $target_freq --start_date $start_date --end_date $end_date --root_path $ROOTPATH \
        >"$logdir/${start_date}_${end_date}.log" 2>&1
    local pid=$!
    echo "All processes of scaling have been initiated."
    wait $pid

    echo "All processes of scaling have been completed."
}

# To call the function with the specified arguments
# run_scale_save "10s" "2023-01-01" "2023-02-01" "BTCUSDT"
