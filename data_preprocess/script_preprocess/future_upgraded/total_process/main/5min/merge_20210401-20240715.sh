nohup bash script/future_upgraded/total_process/BNBUSDT/5min/merge_time_20210401-20240715.sh \
    >log_futures/ticker_record/BNBUSDT_merge.log 2>&1 &

nohup bash script/future_upgraded/total_process/BTCUSDT/5min/merge_time_20210401-20240715.sh \
    >log_futures/ticker_record/BTCUSDT_merge.log 2>&1 &

nohup bash script/future_upgraded/total_process/DOTUSDT/5min/merge_time_20210401-20240715.sh \
    >log_futures/ticker_record/DOTUSDT_merge.log 2>&1 &

nohup bash script/future_upgraded/total_process/ETHUSDT/5min/merge_time_20210401-20240715.sh \
    >log_futures/ticker_record/ETHUSDT_merge.log 2>&1 &
