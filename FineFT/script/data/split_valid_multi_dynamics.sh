nohup python datahandler/slice_model.py --data_path dataset/BTCUSDT/valid.feather \
    >log/data/BTCUSDT/valid_split.log 2>&1 &

nohup python datahandler/slice_model.py --data_path dataset/BNBUSDT/valid.feather \
    >log/data/BNBUSDT/valid_split.log 2>&1 &

nohup python datahandler/slice_model.py --data_path dataset/ETHUSDT/valid.feather \
    >log/data/ETHUSDT/valid_split.log 2>&1 &

nohup python datahandler/slice_model.py --data_path dataset/DOTUSDT/valid.feather \
    >log/data/DOTUSDT/valid_split.log 2>&1 &