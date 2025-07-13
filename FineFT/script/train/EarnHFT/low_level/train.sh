# BTCUSDT
CUDA_VISIBLE_DEVICES=0 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name BTCUSDT --max_holding_number 8 --beta 10 \
    >log/EarnHFT/BTCUSDT/low_level/beta_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name BTCUSDT --max_holding_number 8 --beta 100 \
    >log/EarnHFT/BTCUSDT/low_level/beta_100.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name BTCUSDT --max_holding_number 8 --beta -20 \
    >log/EarnHFT/BTCUSDT/low_level/beta_-20.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name BTCUSDT --max_holding_number 8 --beta -80 \
    >log/EarnHFT/BTCUSDT/low_level/beta_-80.log 2>&1 &
# ETHUSDT
CUDA_VISIBLE_DEVICES=0 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name ETHUSDT --max_holding_number 160 --beta 10 \
    >log/EarnHFT/ETHUSDT/low_level/beta_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name ETHUSDT --max_holding_number 160 --beta 100 \
    >log/EarnHFT/ETHUSDT/low_level/beta_100.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name ETHUSDT --max_holding_number 160 --beta -20 \
    >log/EarnHFT/ETHUSDT/low_level/beta_-20.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name ETHUSDT --max_holding_number 160 --beta -80 \
    >log/EarnHFT/ETHUSDT/low_level/beta_-80.log 2>&1 &
# BNBUSDT
CUDA_VISIBLE_DEVICES=0 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name BNBUSDT --max_holding_number 100 --beta 10 \
    >log/EarnHFT/BNBUSDT/low_level/beta_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name BNBUSDT --max_holding_number 100 --beta 100 \
    >log/EarnHFT/BNBUSDT/low_level/beta_100.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name BNBUSDT --max_holding_number 100 --beta -20 \
    >log/EarnHFT/BNBUSDT/low_level/beta_-20.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name BNBUSDT --max_holding_number 100 --beta -80 \
    >log/EarnHFT/BNBUSDT/low_level/beta_-80.log 2>&1 &
# DOTUSDT
CUDA_VISIBLE_DEVICES=0 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name DOTUSDT --max_holding_number 6000 --beta 10 \
    >log/EarnHFT/DOTUSDT/low_level/beta_10.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name DOTUSDT --max_holding_number 6000 --beta 100 \
    >log/EarnHFT/DOTUSDT/low_level/beta_100.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name DOTUSDT --max_holding_number 6000 --beta -20 \
    >log/EarnHFT/DOTUSDT/low_level/beta_-20.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python RL/EarnHFT/low_level/ddqn_pes_risk_aware.py \
    --dataset_name DOTUSDT --max_holding_number 6000 --beta -80 \
    >log/EarnHFT/DOTUSDT/low_level/beta_-80.log 2>&1 &
