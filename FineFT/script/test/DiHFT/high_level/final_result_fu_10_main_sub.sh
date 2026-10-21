#!/usr/bin/env bash

set -euo pipefail

# ==============================================================================
# fu 30min 双主力动态时间同步回测与诊断评估脚本 (Graceful Downgrade & Dual-Main Only)
#
# 特性：
#   1. 仅允许在当前时点的主力与次主力合约上开仓；
#   2. 合约降级为非主力后，执行平滑降级（graceful_hold_exit）：禁止开新仓与反手，允许持有或策略平仓；
#   3. 单合约本金 6,000 元，双主力组合总本金 12,000 元；
#   4. 输出时间同步组合动作时间线 (portfolio_action_timeline.csv)、TR、MDD、Sharpe 报告；
#   5. 默认评测 valid 集，支持传入参数切换（如: bash ... test）。
# ==============================================================================

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

# 激活 conda finetf 环境
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate finetf 2>/dev/null || true
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate finetf 2>/dev/null || true
fi

export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_NAME=${DATASET_NAME:-fu}
BASE_PATH=${BASE_PATH:-dataset/10min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}
EVAL_STAGE=${1:-${EVAL_STAGE:-valid}}
INITIAL_WALLET_BALANCE=${INITIAL_WALLET_BALANCE:-6000}
MAX_HOLDING_NUMBER=${MAX_HOLDING_NUMBER:-1}
POSITION_CHOICES=${POSITION_CHOICES:-3}
ORDER_BOOK_DEPTH=${ORDER_BOOK_DEPTH:-5}
TRANSACTION_COST=${TRANSACTION_COST:-0.0003}
ACTION_PERSISTENCE=${ACTION_PERSISTENCE:-6}
STOP_LOSS_RETURN_THRESHOLD=${STOP_LOSS_RETURN_THRESHOLD:-0.03}
STOP_LOSS_COOLDOWN_STEPS=${STOP_LOSS_COOLDOWN_STEPS:-12}
CIRCUIT_BREAKER_CONSECUTIVE_STOPS=${CIRCUIT_BREAKER_CONSECUTIVE_STOPS:-3}
CIRCUIT_BREAKER_COOLING_STEPS=${CIRCUIT_BREAKER_COOLING_STEPS:-24}
NON_MAIN_DEFENSE_MODE=${NON_MAIN_DEFENSE_MODE:-graceful_hold_exit}
OUTPUT_SUB_DIR=${OUTPUT_SUB_DIR:-main_sub}
RESULT_DIR="result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/${OUTPUT_SUB_DIR}"

PARA_FILE=${PARA_FILE:-result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/high_level_agent_para.txt}
OPTUNA_CSV=${OPTUNA_CSV:-result/DiHFT/high_level/${DATASET_NAME}/${EXPERIMENT_NAME}/vae_risk_aware_routing_optuna/optuna_results.csv}
SELECTION_MANIFEST=${SELECTION_MANIFEST:-analysis_result/DiHFT/low_level/${DATASET_NAME}/${EXPERIMENT_NAME}/two_dimensional_selection/two_dimensional_selection_manifest.json}

LOG_DIR="log/DiHFT/${DATASET_NAME}/high_level/final_result_main_sub/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/final_result_${EVAL_STAGE}.log"

echo "================================================================================"
echo "🚀 启动 fu 10min 最优门控超参双主力平滑降级回测与诊断"
echo "  评测阶段 (EVAL_STAGE): ${EVAL_STAGE}"
echo "  单合约初始资金: ¥${INITIAL_WALLET_BALANCE} (两主力组合总本金: ¥$((INITIAL_WALLET_BALANCE * 2)))"
echo "  最优门控配置: ${PARA_FILE}"
echo "  非主力防御模式: ${NON_MAIN_DEFENSE_MODE}"
echo "  产物输出目录: ${RESULT_DIR}"
echo "  日志位置: ${LOG_FILE}"
echo "================================================================================"

python -u FineFT/RL/DiHFT/high_level/vae_routing_final_result_macro_action.py \
    --base_path "${BASE_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --output_sub_dir "${OUTPUT_SUB_DIR}" \
    --selection_manifest "${SELECTION_MANIFEST}" \
    --para_file "${PARA_FILE}" \
    --optuna_csv "${OPTUNA_CSV}" \
    --eval_stage "${EVAL_STAGE}" \
    --max_holding_number "${MAX_HOLDING_NUMBER}" \
    --initial_wallet_balance "${INITIAL_WALLET_BALANCE}" \
    --position_choices "${POSITION_CHOICES}" \
    --order_book_depth "${ORDER_BOOK_DEPTH}" \
    --transcation_cost "${TRANSACTION_COST}" \
    --short_estimated_rate 0 \
    --long_estimated_rate 0 \
    --enable_non_main_contract_defense \
    --non_main_defense_mode "${NON_MAIN_DEFENSE_MODE}" \
    --enable_delivery_month_defense --delivery_defense_days "${DELIVERY_DEFENSE_DAYS}" \
    --action_persistence "${ACTION_PERSISTENCE}" \
    --stop_loss_return_threshold "${STOP_LOSS_RETURN_THRESHOLD}" \
    --stop_loss_cooldown_steps "${STOP_LOSS_COOLDOWN_STEPS}" \
    --circuit_breaker_consecutive_stops "${CIRCUIT_BREAKER_CONSECUTIVE_STOPS}" \
    --circuit_breaker_cooling_steps "${CIRCUIT_BREAKER_COOLING_STEPS}" \
    2>&1 | tee "${LOG_FILE}"

# 审计级日度检查 (Daily Trading Audit)
ENABLE_AUDIT=${ENABLE_AUDIT:-1}
AUDIT_OUTPUT_DIR=${AUDIT_OUTPUT_DIR:-analysis_result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/${OUTPUT_SUB_DIR}/daily_trading_audit_${EVAL_STAGE}}
TRADING_DAY_MAPPING=${TRADING_DAY_MAPPING:-}
if [ -z "${TRADING_DAY_MAPPING}" ] && [ -f "${BASE_PATH}/${DATASET_NAME}/trading_day_mapping.json" ]; then
    TRADING_DAY_MAPPING="${BASE_PATH}/${DATASET_NAME}/trading_day_mapping.json"
fi
RUN_ID=${RUN_ID:-}

if [ "${ENABLE_AUDIT}" = "1" ]; then
    AUDIT_CMD_ARGS=(
        --result_dir "${RESULT_DIR}"
        --market_data_dir "${BASE_PATH}/${DATASET_NAME}/${EVAL_STAGE}"
        --output_dir "${AUDIT_OUTPUT_DIR}"
    )
    if [ -n "${TRADING_DAY_MAPPING}" ]; then
        AUDIT_CMD_ARGS+=(--trading_day_mapping "${TRADING_DAY_MAPPING}")
    fi
    if [ -n "${RUN_ID}" ]; then
        AUDIT_CMD_ARGS+=(--run_id "${RUN_ID}")
    fi
    if [ -f "${RESULT_DIR}/audit_metadata.json" ] || \
       ls "${RESULT_DIR}/contracts/*/audit_metadata.json" >/dev/null 2>&1; then
        echo "▶ 执行日度交易审计 (Daily Trading Audit)..."
        python -u FineFT/analysis/diagnostics/daily_trading_audit.py \
            --skip-missing-input \
            "${AUDIT_CMD_ARGS[@]}" \
            2>&1 | tee -a "${LOG_FILE}"
    fi
fi

echo "================================================================================"
echo "✔ 双主力动态时间同步回测与诊断评估已完成！"
echo "  产物目录: ${RESULT_DIR}/"
echo "  核心汇总表: ${RESULT_DIR}/summary_metrics.csv"
echo "  动作时间线: ${RESULT_DIR}/portfolio_action_timeline.csv"
echo "  完整报告: ${RESULT_DIR}/portfolio_summary.md"
echo "================================================================================"
