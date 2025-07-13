nohup python operator_futures/delete/delete.py \
    --symbols BTCUSDT --target_freq 5min \
    >log_futures/delete/5min/BTCUSDT.log 2>&1 &

nohup python operator_futures/delete/delete.py \
    --symbols BNBUSDT --target_freq 5min \
    >log_futures/delete/5min/BNBUSDT.log 2>&1 &

nohup python operator_futures/delete/delete.py \
    --symbols ETHUSDT --target_freq 5min \
    >log_futures/delete/5min/ETHUSDT.log 2>&1 &

nohup python operator_futures/delete/delete.py \
    --symbols DOTUSDT --target_freq 5min \
    >log_futures/delete/5min/DOTUSDT.log 2>&1 &
