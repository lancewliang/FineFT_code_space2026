logdir="log_futures/downscale/derivative_ticker/10s/BTCUSDT"
if [ ! -d "$logdir" ]; then
    mkdir -p "$logdir" 
fi
nohup bash script/futures/down_scale_derivative_ticker/down_scale_single_shot.sh \
    >"$logdir"/all.log 2>&1 &

logdir="log_futures/downscale/derivative_ticker/1min/BTCUSDT"
if [ ! -d "$logdir" ]; then
    mkdir -p "$logdir" 
fi
nohup bash script/futures/down_scale_derivative_ticker/down_scale_other_shot.sh \
    >"$logdir"/all.log 2>&1 &