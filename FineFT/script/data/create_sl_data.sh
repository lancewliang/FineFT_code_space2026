nohup python datahandler/create_data_adaboost.py \
    --dataset_name BNBUSDT \
    >log/data/BNBUSDT/create_data_adaboost.log 2>&1 &

nohup python datahandler/create_data_adaboost.py \
    --dataset_name BTCUSDT \
    >log/data/BTCUSDT/create_data_adaboost.log 2>&1 &

nohup python datahandler/create_data_adaboost.py \
    --dataset_name ETHUSDT \
    >log/data/ETHUSDT/create_data_adaboost.log 2>&1 &

nohup python datahandler/create_data_adaboost.py \
    --dataset_name DOTUSDT \
    >log/data/DOTUSDT/create_data_adaboost.log 2>&1 &
