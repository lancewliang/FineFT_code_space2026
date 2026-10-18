#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# fu (燃料油) 30min 多预测期限特征审计 (Multi-Horizon Feature Audit) 执行脚本
#
# 该脚本执行以下步骤：
#   1. 检查并激活 conda finetf 环境，设置 PYTHONPATH
#   2. 静态前置校验 (validate-inputs)：校验 YAML 配置及关联的 Manifest 文件
#   3. 正式执行特征审计 (run)：执行多合约、多期限特征评估与联合区块 Bootstrap
#   4. 检查产物输出目录与 _SUCCESS 标记
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

# 2. 解析配置文件参数（默认使用 configs/audit_fu_30min_template.yaml）
DEFAULT_CONFIG="${REPO_ROOT}/data_preprocess/operator_futures/feature_audit/configs/audit_fu_30min_template.yaml"
CONFIG_FILE="${1:-${DEFAULT_CONFIG}}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "❌ 错误: 配置文件不存在: ${CONFIG_FILE}" >&2
    exit 1
fi

# 3. 日志目录初始化
LOG_DIR="${REPO_ROOT}/log/feature_audit/fu/30min"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/audit_fu_30min_${TIMESTAMP}.log"

echo "================================================================================"
echo "🚀 启动 fu 30min 多预测期限特征审计"
echo "  工程根目录: ${REPO_ROOT}"
echo "  配置文件:   ${CONFIG_FILE}"
echo "  运行环境:   $(which python 2>/dev/null || echo '未检测到 python')"
echo "  执行时间:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "  执行日志:   ${LOG_FILE}"
echo "================================================================================"

start_time=$(date +%s)

# 4. 步骤一：前置静态校验 (validate-inputs)
echo ""
echo "--------------------------------------------------------------------------------"
echo "▶ [步骤 1/2] 正在校验配置文件与相关 Manifest 清单..."
echo "--------------------------------------------------------------------------------"
if ! python -m data_preprocess.operator_futures.feature_audit.cli validate-inputs --config "${CONFIG_FILE}" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "" >&2
    echo "❌ [步骤 1/2] 校验未通过，请检查上述错误信息后重试。" >&2
    exit 1
fi
echo "✔ [步骤 1/2] 配置文件与依赖清单校验通过！"

# 5. 步骤二：正式执行审计计算 (run)
echo ""
echo "--------------------------------------------------------------------------------"
echo "▶ [步骤 2/2] 正在执行全量特征审计（多合约、7 期限、双目标、联合区块 Bootstrap）..."
echo "--------------------------------------------------------------------------------"
if ! python -m data_preprocess.operator_futures.feature_audit.cli run --config "${CONFIG_FILE}" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "" >&2
    echo "❌ [步骤 2/2] 审计运行异常中断，请查看日志: ${LOG_FILE}" >&2
    exit 1
fi

end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo ""
echo "================================================================================"
echo "🎉 fu 30min 特征审计运行完成！(总耗时: ${minutes}m ${seconds}s)"
echo "--------------------------------------------------------------------------------"

# 提取并验证结果目录
RESULT_ROOT=$(grep '^result_root:' "${CONFIG_FILE}" | awk '{print $2}' | tr -d '"' | tr -d "'")
if [[ -n "${RESULT_ROOT}" && -d "${REPO_ROOT}/${RESULT_ROOT}" ]]; then
    FULL_RESULT_DIR="${REPO_ROOT}/${RESULT_ROOT}"
    echo "产物目录: ${FULL_RESULT_DIR}"
    if [[ -f "${FULL_RESULT_DIR}/_SUCCESS" ]]; then
        echo "状态标记: ✔ _SUCCESS 标记已生成"
    else
        echo "状态标记: ⚠️ 未检测到 _SUCCESS 标记文件"
    fi
    echo "核心产物文件列表:"
    ls -lh "${FULL_RESULT_DIR}" | awk 'NR>1 {print "  - "$9" ("$5")"}'
fi

echo "执行日志已保存至: ${LOG_FILE}"
echo "================================================================================"
