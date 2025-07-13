function run_train_ablation {
    local dataset_name=$1
    local max_holding_number=$2

    # 检查并创建日志目录
    local settings=("EarnHFT_PES" "EarnHFT_random" "FineFT" "FineFT_wo_pretrain")

    # 遍历设置并创建日志目录

    log_EarnHFT_PES="log/ablation/EarnHFT_PES"
    mkdir -p "${log_EarnHFT_PES}"

    log_EarnHFT_random="log/ablation/EarnHFT_random"
    mkdir -p "${log_EarnHFT_random}"

    log_FineFT="log/ablation/FineFT"
    mkdir -p "${log_FineFT}"

    log_FineFT_wo_pretrain="log/ablation/FineFT_wo_pretrain"
    mkdir -p "${log_FineFT_wo_pretrain}"

    CUDA_VISIBLE_DEVICES=0 nohup python RL/DiHFT/ablation/converge_steps_sun/EarnHFT_PES.py \
        --dataset_name "${dataset_name}" \
        --max_holding_number "${max_holding_number}" \
        --beta -80 \
        >"${log_EarnHFT_PES}/${dataset_name}_beta_-80.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=1 nohup python RL/DiHFT/ablation/converge_steps_sun/EarnHFT_PES.py \
        --dataset_name "${dataset_name}" \
        --max_holding_number "${max_holding_number}" \
        --beta -40 \
        >"${log_EarnHFT_PES}/${dataset_name}_beta_-40.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 nohup python RL/DiHFT/ablation/converge_steps_sun/EarnHFT_PES.py \
        --dataset_name "${dataset_name}" \
        --max_holding_number "${max_holding_number}" \
        --beta 0 \
        >"${log_EarnHFT_PES}/${dataset_name}_beta_0.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=3 nohup python RL/DiHFT/ablation/converge_steps_sun/EarnHFT_PES.py \
        --dataset_name "${dataset_name}" \
        --max_holding_number "${max_holding_number}" \
        --beta 40 \
        >"${log_EarnHFT_PES}/${dataset_name}_beta_40.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=0 nohup python RL/DiHFT/ablation/converge_steps_sun/EarnHFT_PES.py \
        --dataset_name "${dataset_name}" \
        --max_holding_number "${max_holding_number}" \
        --beta 100 \
        >"${log_EarnHFT_PES}/${dataset_name}_beta_100.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=1 nohup python RL/DiHFT/ablation/converge_steps_sun/EarnHFT_random.py \
        --dataset_name "${dataset_name}" \
        --max_holding_number "${max_holding_number}" \
        >"${log_EarnHFT_random}/${dataset_name}.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=2 nohup python RL/DiHFT/ablation/converge_steps_sun/FineFT_without_pretrain.py \
        --dataset_name "${dataset_name}" \
        --max_holding_number "${max_holding_number}" \
        >"${log_FineFT_wo_pretrain}/${dataset_name}.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=3 nohup python RL/DiHFT/ablation/converge_steps_sun/FineFT.py \
        --dataset_name "${dataset_name}" \
        --max_holding_number "${max_holding_number}" \
        >"${log_FineFT}/${dataset_name}.log" 2>&1 &
}
run_train_ablation BNBUSDT 100
run_train_ablation BTCUSDT 8
run_train_ablation DOTUSDT 6000
run_train_ablation ETHUSDT 160
