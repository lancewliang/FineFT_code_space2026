target_freq="10s"
start_date=2023-01-01
end_date=2023-12-31
logdir="log_futures/concat/10s/BTCUSDT"
if [ ! -d "$logdir" ]; then
    mkdir -p "$logdir"
fi

nohup python -u operator_futures/merge_concat/concat.py \
    --symbols BTCUSDT --target_freq $target_freq --start_date $start_date --end_date $end_date \
    >"$logdir/$start_date_$end_date.log" 2>&1 &
