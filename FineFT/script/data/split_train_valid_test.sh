nohup python datahandler/preprocess_data.py --trading_pair BTCUSDT \
    >log/data/BTCUSDT/train_valid_test_split.log 2>&1 &

nohup python datahandler/preprocess_data.py --trading_pair BNBUSDT \
    >log/data/BNBUSDT/train_valid_test_split.log 2>&1 &

nohup python datahandler/preprocess_data.py --trading_pair ETHUSDT \
    >log/data/ETHUSDT/train_valid_test_split.log 2>&1 &

nohup python datahandler/preprocess_data.py --trading_pair DOTUSDT \
    >log/data/DOTUSDT/train_valid_test_split.log 2>&1 &