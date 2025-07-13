run_downscale_process() {
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
        local log_dir="log_futures/downscale/base_feature/$target_freq/$symbol"
        if [ ! -d "$log_dir" ]; then
            mkdir -p "$log_dir"
        fi
        nohup python -u operator_futures/features_related/base_feature.py \
            --symbols $symbol --target_freq $target_freq --date $current_date --root_path $ROOTPATH \
            >"$log_dir/$current_date.log" 2>&1 &
        local pid=$! # 获取当前启动进程的PID

        let process_count=process_count+1
        if [ $process_count -eq $MAX_PROCESSES ]; then
            local last_pid=$pid # 记录最后一个进程的PID
            wait $last_pid      # 等待最后一个进程完成
            process_count=0     # 重置进程计数器以便开始新的一批进程
        fi

        current_date=$(date -I -d "$current_date + 1 day")
    done

    # 等待所有剩余的进程完成
    
    echo "All processes of creating base features have been initiated."
    wait
    echo "All processes of creating base features have been completed."
}
