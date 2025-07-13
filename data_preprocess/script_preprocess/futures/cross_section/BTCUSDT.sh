logdir="log_futures/downscale/cross_section/10s/BTCUSDT"
if [ ! -d "$logdir" ]; then
    mkdir -p "$logdir"
fi
nohup bash script/futures/cross_section/make_feature_single_shot.sh \
    >"$logdir"/all.log 2>&1 &
