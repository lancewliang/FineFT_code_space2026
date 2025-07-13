source script/future_upgraded/total_process/util_process.sh

target_freq="10s"
start_date='2023-10-01'
end_date='2024-04-01'
symbol='ETHUSDT'
max_processes_1=100
max_processes_2=20
root_path="."

run_all_process $target_freq $start_date $end_date $symbol $max_processes_1 $max_processes_2 $root_path >log_futures/ticker_result/ETHUSDT.log 2>&1


# ps aux | grep operator_futures | awk '{print $2}' | xargs kill