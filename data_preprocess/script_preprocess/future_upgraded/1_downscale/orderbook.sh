function run_orderbook_downscaling {
    local START_DATE=$1
    local END_DATE=$2
    local MAX_PROCESSES=$3
    local target_freq=$4
    local symbols=$5
    local ROOTPATH=$6

    local current_date=$(date -I -d "$START_DATE")
    local process_count=0
    if [ $(date -d "$START_DATE" +%s) -gt $(date -d "$END_DATE" +%s) ]; then
        echo "Error: START_DATE ($START_DATE) is later than END_DATE ($END_DATE). Exiting."
        return 1
    fi
    while [ "$current_date" != "$END_DATE" ]; do
        echo "Starting process for $current_date"
        local log_dir="log_futures/downscale/orderbook/$target_freq/$symbols"
        if [ ! -d "$log_dir" ]; then
            mkdir -p "$log_dir"
        fi
        nohup python -u operator_futures/orderbook_25/down_scale_single_shot.py \
            --symbols $symbols --target_freq $target_freq --date $current_date --root_path $ROOTPATH \
            >"$log_dir/$current_date.log" 2>&1 &
        local pid=$!

        let process_count=process_count+1
        if [ $process_count -eq $MAX_PROCESSES ]; then
            local last_pid=$pid
            wait $last_pid
            process_count=0
        fi

        current_date=$(date -I -d "$current_date + 1 day")
    done
    wait
    echo "All processes of down scaling orderbook snapshot have been initiated."
}

function run_orderbook_downscaling_with_base {
    local START_DATE=$1
    local END_DATE=$2
    local MAX_PROCESSES=$3
    local target_freq=$4
    local base_freq=$5
    local symbols=$6
    local ROOTPATH=$7

    local current_date=$(date -I -d "$START_DATE")
    local process_count=0

    while [ "$current_date" != "$END_DATE" ]; do
        echo "Starting process for $current_date"
        local log_dir="log_futures/downscale/orderbook/${target_freq}/${symbols}"
        if [ ! -d "$log_dir" ]; then
            mkdir -p "$log_dir"
        fi
        nohup python -u operator_futures/orderbook_25/down_scale_single_shot_base_other.py \
            --symbols $symbols --target_freq $target_freq --base_freq $base_freq --date $current_date --root_path $ROOTPATH \
            >"$log_dir/$current_date.log" 2>&1 &
        local pid=$! 

        let process_count=process_count+1
        if [ $process_count -eq $MAX_PROCESSES ]; then
            local last_pid=$pid 
            wait $last_pid      
            process_count=0     
        fi

        current_date=$(date -I -d "$current_date + 1 day")
    done

    # 等待所有剩余的进程完成
    wait
    echo "All processes of down scaling orderbook snapshot based on other frequency have been initiated."
}

