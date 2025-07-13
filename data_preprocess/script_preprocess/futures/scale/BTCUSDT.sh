target_freq="10s"
start_date=2023-01-01
end_date=2023-02-01
logdir="log_futures/scale/10s/BTCUSDT"
if [ ! -d "$logdir" ]; then
    mkdir -p "$logdir"
fi

nohup python -u operator_futures/scale_describe_save/scale_save.py \
    --symbols BTCUSDT --target_freq $target_freq --start_date $start_date --end_date $end_date \
    >"$logdir/"$start_date"_"$end_date".log" 2>&1 &