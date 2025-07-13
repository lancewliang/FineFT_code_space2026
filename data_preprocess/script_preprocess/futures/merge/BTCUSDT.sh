logdir="log_futures/merge/10s/BTCUSDT"
if [ ! -d "$logdir" ]; then
    mkdir -p "$logdir" 
fi
nohup bash script/futures/merge/merge_single_shot.sh \
    >"$logdir"/all.log 2>&1 &

