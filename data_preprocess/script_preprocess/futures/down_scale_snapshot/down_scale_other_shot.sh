START_DATE="2023-01-01"
END_DATE="2023-12-31"
MAX_PROCESSES=100
target_freq="1min"
base_freq="10s"

current_date=$(date -I -d "$START_DATE")
process_count=0

while [ "$current_date" != "$END_DATE" ]; do
    echo "Starting process for $current_date"
    log_dir="log_futures/downscale/orderbook/$target_freq/BTCUSDT"
    if [ ! -d "$log_dir" ]; then
        mkdir -p "$log_dir"
    fi
    nohup python -u operator_futures/orderbook_25/down_scale_single_shot_base_other.py \
        --symbols BTCUSDT --target_freq $target_freq --base_freq $base_freq --date $current_date \
        >"$log_dir/$current_date.log" 2>&1 &
    pid=$! # 获取当前启动进程的PID

    let process_count=process_count+1
    if [ $process_count -eq $MAX_PROCESSES ]; then
        last_pid=$pid   # 记录最后一个进程的PID
        wait $last_pid  # 等待最后一个进程完成
        process_count=0 # 重置进程计数器以便开始新的一批进程
    fi

    current_date=$(date -I -d "$current_date + 1 day")
done

# 等待所有剩余的进程完成
echo "All processes have been initiated."
