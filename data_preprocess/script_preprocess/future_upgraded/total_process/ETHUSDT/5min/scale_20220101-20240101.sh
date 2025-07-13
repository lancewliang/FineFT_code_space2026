source script/future_upgraded/total_process/util_process.sh

target_freq="5min"
start_date='2022-01-01'
end_date='2024-01-01'
symbol='ETHUSDT'
max_processes_1=100
max_processes_2=20
root_path="."

run_scale_save_df $target_freq $start_date $end_date $symbol $max_processes_1 $max_processes_2 $root_path \
    >log_futures/ticker_result/ETHUSDT_scale_5min_20220101-20240101.log 2>&1
