logdir="log_futures/downscale/base_feature/10s/BTCUSDT"
if [ ! -d "$logdir" ]; then
    mkdir -p "$logdir"
fi
nohup bash script/futures/base_feature/down_scale_single_shot.sh \
    >"$logdir"/all.log 2>&1 &
