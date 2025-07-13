# BTCUSDT
function run_ddqn_pes_risk_aware {
    local dataset_name=$1
    local max_holding_number=$2
    local beta=$3

    # 检查并创建日志目录
    log_dir="log/EarnHFT/${dataset_name}/low_level/test"
    mkdir -p "${log_dir}"

    # 保存PID的数组
    pids=()

    # 循环执行Epoch 1到50
    for epoch in {1..100}; do
        local gpu_index=$(((epoch - 1) % 4))
        CUDA_VISIBLE_DEVICES=${gpu_index} nohup python RL/EarnHFT/low_level/test_ddqn.py \
            --dataset_name "${dataset_name}" \
            --max_holding_number "${max_holding_number}" \
            --beta "${beta}" \
            --epoch_number "${epoch}" \
            >"${log_dir}/beta_${beta}_epoch_${epoch}.log" 2>&1 &
        pids+=($!) # 将每个后台进程的PID添加到数组中
    done

    # 等待所有的PID
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo "${dataset_name} ${max_holding_number} ${beta} All processes completed successfully."
}
#BTCUSDT
run_ddqn_pes_risk_aware BTCUSDT 8 10

run_ddqn_pes_risk_aware BTCUSDT 8 100

run_ddqn_pes_risk_aware BTCUSDT 8 -20

run_ddqn_pes_risk_aware BTCUSDT 8 -80

#ETHUSDT
run_ddqn_pes_risk_aware ETHUSDT 160 10

run_ddqn_pes_risk_aware ETHUSDT 160 100

run_ddqn_pes_risk_aware ETHUSDT 160 -20

run_ddqn_pes_risk_aware ETHUSDT 160 -80
# BNBUSDT
run_ddqn_pes_risk_aware BNBUSDT 100 10

run_ddqn_pes_risk_aware BNBUSDT 100 100

run_ddqn_pes_risk_aware BNBUSDT 100 -20

run_ddqn_pes_risk_aware BNBUSDT 100 -80
#DOTUSDT
run_ddqn_pes_risk_aware DOTUSDT 6000 10

run_ddqn_pes_risk_aware DOTUSDT 6000 100

run_ddqn_pes_risk_aware DOTUSDT 6000 -20

run_ddqn_pes_risk_aware DOTUSDT 6000 -80
