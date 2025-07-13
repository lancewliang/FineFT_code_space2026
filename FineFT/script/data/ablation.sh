nohup python datahandler/ablation_data_slice.py --trading_pair BTCUSDT \
    >log/data/BTCUSDT/ablation_train_valid_split.log 2>&1 &

nohup python datahandler/ablation_data_slice.py --trading_pair BNBUSDT \
    >log/data/BNBUSDT/ablation_train_valid_split.log 2>&1 &

nohup python datahandler/ablation_data_slice.py --trading_pair ETHUSDT \
    >log/data/ETHUSDT/ablation_train_valid_split.log 2>&1 &

nohup python datahandler/ablation_data_slice.py --trading_pair DOTUSDT \
    >log/data/DOTUSDT/ablation_train_valid_split.log 2>&1 &



nohup python datahandler/slice_model.py --data_path dataset/ablation/BTCUSDT/valid.feather \
    >log/data/BTCUSDT/ablation_valid_split.log 2>&1 &

nohup python datahandler/slice_model.py --data_path dataset/ablation/BNBUSDT/valid.feather \
    >log/data/BNBUSDT/ablation_valid_split.log 2>&1 &

nohup python datahandler/slice_model.py --data_path dataset/ablation/ETHUSDT/valid.feather \
    >log/data/ETHUSDT/ablation_valid_split.log 2>&1 &

nohup python datahandler/slice_model.py --data_path dataset/ablation/DOTUSDT/valid.feather \
    >log/data/DOTUSDT/ablation_valid_split.log 2>&1 &