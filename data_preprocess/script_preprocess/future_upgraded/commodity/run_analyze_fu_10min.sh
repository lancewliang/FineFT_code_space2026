#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# fu (燃料油) 10min 多预测期限特征审计结果分析 (Analyze Audit Results) 执行脚本
#
# 该脚本执行以下步骤：
#   1. 检查并激活 conda finetf 环境，设置 PYTHONPATH
#   2. 解析审计产物目录及目标分析期限 (超短 1h, 短 2h, 次日 next_td, 中 next_5td, 长 next_15td)
#   3. 调用 analyze_audit_results.py 运行全量洞察分析与稳健特征筛选
#   4. 输出汇总统计与洞察报告文件清单
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

# 1. 激活 conda finetf 运行环境
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate finetf
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate finetf
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/data_preprocess:${REPO_ROOT}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

# 2. 解析参数
DEFAULT_AUDIT_DIR="${REPO_ROOT}/analysis_result/DiHFT/feature_horizon_audit/fu/10min"
AUDIT_DIR="${1:-${DEFAULT_AUDIT_DIR}}"

DEFAULT_OUTPUT_DIR="${AUDIT_DIR}/insights"
OUTPUT_DIR="${2:-${DEFAULT_OUTPUT_DIR}}"

DEFAULT_HORIZONS="1h,2h,next_td,next_5td,next_15td"
HORIZONS="${3:-${DEFAULT_HORIZONS}}"

if [[ ! -d "${AUDIT_DIR}" ]]; then
    echo "❌ 错误: 审计产物目录不存在: ${AUDIT_DIR}" >&2
    echo "请确认该目录已完成特征审计运行并生成了相关 CSV/JSON 产物。" >&2
    exit 1
fi

# 3. 日志目录初始化
LOG_DIR="${REPO_ROOT}/log/feature_audit/fu/10min"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/analyze_fu_10min_${TIMESTAMP}.log"

echo "================================================================================"
echo "🚀 启动 fu 10min 特征审计结果分析与洞察提取"
echo "  工程根目录:   ${REPO_ROOT}"
echo "  审计产物目录: ${AUDIT_DIR}"
echo "  分析产物目录: ${OUTPUT_DIR}"
echo "  目标分析周期: ${HORIZONS}"
echo "  运行环境:     $(which python 2>/dev/null || echo '未检测到 python')"
echo "  执行时间:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "  执行日志:     ${LOG_FILE}"
echo "================================================================================"

start_time=$(date +%s)

# 4. 执行分析脚本
if ! python -m data_preprocess.operator_futures.feature_audit.analyze_audit_results \
    --audit-dir "${AUDIT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --horizons "${HORIZONS}" \
    --top-k 15 \
    --min-sign-consistency 0.80 \
    --min-coverage 0.50 2>&1 | tee -a "${LOG_FILE}"; then
    echo "" >&2
    echo "❌ 审计分析执行失败，详情请查看日志: ${LOG_FILE}" >&2
    exit 1
fi

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "================================================================================"
echo "🎉 fu 10min 特征审计分析完成！(总耗时: ${duration}s)"
echo "--------------------------------------------------------------------------------"
echo "产物目录: ${OUTPUT_DIR}"
echo "核心生成文件:"
ls -lh "${OUTPUT_DIR}" | awk 'NR>1 {print "  - "$9" ("$5")"}'
echo ""
echo "执行日志已保存至: ${LOG_FILE}"
echo "================================================================================"
