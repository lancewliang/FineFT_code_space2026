source script/future_upgraded/total_process/util_process.sh

target_freq="5min"
max_processes_1=100
max_processes_2=20
root_path="."
start_date='2023-07-05'
end_date='2023-08-01'
symbol='BTCUSDT'
run_from_start_to_cross_section $target_freq $start_date $end_date $symbol $max_processes_1 $max_processes_2 $root_path >log_futures/ticker_result/BTCUSDT.log 2>&1

# start_date='2023-07-25'
# end_date='2023-08-05'

# run_from_start_to_cross_section $target_freq $start_date $end_date $symbol $max_processes_1 $max_processes_2 $root_path >log_futures/ticker_result/BTCUSDT.log 2>&1



# start_date='2023-01-01'
# end_date='2023-07-01'
# symbol='BNBUSDT'

# run_from_start_to_cross_section $target_freq $start_date $end_date $symbol $max_processes_1 $max_processes_2 $root_path >log_futures/ticker_result/BNBUSDT.log 2>&1

# start_date='2022-10-01'
# end_date='2023-04-01'
# start_date='2023-03-30'
# end_date='2023-04-02'
# symbol='ETHUSDT'

# run_from_start_to_cross_section $target_freq $start_date $end_date $symbol $max_processes_1 $max_processes_2 $root_path >log_futures/ticker_result/ETHUSDT.log 2>&1

# start_date='2023-01-01'
# end_date='2023-07-01'
# symbol='DOTUSDT'

# run_from_start_to_cross_section $target_freq $start_date $end_date $symbol $max_processes_1 $max_processes_2 $root_path >log_futures/ticker_result/DOTUSDT.log 2>&1
