target_freq="5min"
# start_date=2022-05-01
# end_date=2024-05-01

nohup python ./HFT_DATAPREPROCESS/over_view/create_overview.py \
    --symbols BTCUSDT --start_date 2021-04-01 --end_date 2024-05-01 --target_freq $target_freq \
    >log_futures/overview/BTCUSDT.log 2>&1 &

nohup python ./HFT_DATAPREPROCESS/over_view/create_overview.py \
    --symbols BNBUSDT --start_date 2021-04-01 --end_date 2024-05-01 --target_freq $target_freq \
    >log_futures/overview/BNBUSDT.log 2>&1 &

nohup python ./HFT_DATAPREPROCESS/over_view/create_overview.py \
    --symbols DOTUSDT --start_date 2021-04-01 --end_date 2024-05-01 --target_freq $target_freq \
    >log_futures/overview/DOTUSDT.log 2>&1 &

nohup python ./HFT_DATAPREPROCESS/over_view/create_overview.py \
    --symbols ETHUSDT --start_date 2021-04-01 --end_date 2024-05-01 --target_freq $target_freq \
    >log_futures/overview/ETHUSDT.log 2>&1 &
