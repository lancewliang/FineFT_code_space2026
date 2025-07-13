source script/future_upgraded/total_process/util_process.sh

target_freq="5min"
start_date='2022-04-01'
end_date='2023-07-15'
symbol='BTCUSDT'
max_processes_1=100
max_processes_2=20
root_path="."

run_cross_section $target_freq $start_date $end_date $symbol $max_processes_1 $max_processes_2 $root_path \
    >log_futures/ticker_result/BTCUSDT_5min_20210401-20240101.log 2>&1