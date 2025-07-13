source script/future_upgraded/total_process/util_process.sh
target_freq="5min"
max_processes_1=100
max_processes_2=20
root_path="."


start_date='2022-06-18'
end_date='2022-06-21'
symbol='BTCUSDT'

run_from_start_to_cross_section $target_freq $start_date $end_date $symbol $max_processes_1 $max_processes_2 $root_path >log_futures/ticker_result/5min/BTCUSDT_June.log 2>&1


# start_date='2023-07-05'
# end_date='2023-07-31'
# symbol='BTCUSDT'

# run_from_start_to_cross_section $target_freq $start_date $end_date $symbol $max_processes_1 $max_processes_2 $root_path >log_futures/ticker_result/5min/BTCUSDT.log 2>&1
