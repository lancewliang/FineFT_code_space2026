source script/future_upgraded/total_process/util_process.sh

target_freq="5min"
start_date='2024-04-30'
end_date='2024-07-16'
symbol='BTCUSDT'
max_processes_1=100
max_processes_2=5
root_path="."

run_from_start_to_cross_section $target_freq $start_date $end_date $symbol $max_processes_1 $max_processes_2 $root_path \
    >log_futures/ticker_result/BTCUSDT_5min_20240430-20240716.log 2>&1
# ps aux | grep operator_futures | awk '{print $2}' | xargs kill
