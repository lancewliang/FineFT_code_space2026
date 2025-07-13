source script/future_upgraded/total_process/util_process.sh

target_freq="5min"
start_date='2022-06-23'
end_date='2022-06-30'
symbol='BTCUSDT'
max_processes_1=100
max_processes_2=10
root_path="."

run_from_start_to_cross_section $target_freq $start_date $end_date $symbol $max_processes_1 $max_processes_2 $root_path \
    >log_futures/ticker_result/BTCUSDT_5min_20210420-20210710.log 2>&1





start_date='2022-12-05'
end_date='2022-12-10'


run_from_start_to_cross_section $target_freq $start_date $end_date $symbol $max_processes_1 $max_processes_2 $root_path \
    >log_futures/ticker_result/BTCUSDT_5min_20221205-20221210.log 2>&1


start_date='2023-07-29'
end_date='2023-08-03'


run_from_start_to_cross_section $target_freq $start_date $end_date $symbol $max_processes_1 $max_processes_2 $root_path \
    >log_futures/ticker_result/BTCUSDT_5min_20230729-20230803.log 2>&1

# ps aux | grep operator_futures | awk '{print $2}' | xargs kill
