nohup python datahandler/vae_data_creation.py --dataset_name BTCUSDT \
    >log/data/BTCUSDT/vae_data.log 2>&1 &

nohup python datahandler/vae_data_creation.py --dataset_name BNBUSDT \
    >log/data/BNBUSDT/vae_data.log 2>&1 &

nohup python datahandler/vae_data_creation.py --dataset_name ETHUSDT \
    >log/data/ETHUSDT/vae_data.log 2>&1 &

nohup python datahandler/vae_data_creation.py --dataset_name DOTUSDT \
    >log/data/DOTUSDT/vae_data.log 2>&1 &
