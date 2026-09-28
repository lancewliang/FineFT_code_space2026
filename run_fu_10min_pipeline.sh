#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# fu (燃料油) 10min 强化学习全流程流水线执行脚本
#
# 对应 fu.readme.md 中的推荐串联运行顺序：
#   1. main_10min_fu.sh
#   2. commodity_data_handler_10min_fu.sh
#   3. train_commodity_fu_10_parallel.sh
#   4. test_util_fu_10.sh
#   5. low_level_fu_10.sh
#   6. VAE_util_fu_10.sh
#   7. vae_optuna_fu_10.sh
#   8. high_level_heurstic_fu_10.sh
#   9. final_result_fu_10_p.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOTPATH="${ROOTPATH:-${SCRIPT_DIR}}"
cd "${ROOTPATH}"

# 激活 conda finetf 环境
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate finetf
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate finetf
fi

export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT:${ROOTPATH}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}"

# 9 个核心脚本定义 (格式: 步骤名称|脚本相对路径|主要日志路径)
declare -a PIPELINE_STEPS=(
    "数据预处理|data_preprocess/script_preprocess/future_upgraded/commodity/main_10min_fu.sh|log_futures/ticker_result/commodity/fu_10min_2023-01-01_2026-03-01.log"
    "FineFT 数据准备|FineFT/script/data/commodity_data_handler_10min_fu.sh|终端标准输出"
    "低层 Agent 并行训练|FineFT/script/train/train_commodity_fu_10_parallel.sh|log/DiHFT/fu/low_level/train/10min/10min_parallel/advantage-10min-parallel.log"
    "低层 Agent 并行测试|FineFT/script/test/DiHFT/low_level/test_util_fu_10.sh|log/DiHFT/fu/low_level/test/10min_parallel/{slope,volatility}/epoch_*.log"
    "低层 Agent 二维联合筛选与分析|FineFT/script/analysis/pick_agent/low_level_fu_10.sh|log/analysis/pick_agent/DiHFT/fu/10min_parallel.log"
    "VAE 并行训练与评估 (多进程并行)|FineFT/script/train/DiHFT/low_level/VAE_util_fu_10.sh|log/DiHFT/fu/VAE/10min_parallel/{slope,volatility}/train_label_*.log"
    "高层 VAE 路由 Optuna 优化与分析|FineFT/script/test/DiHFT/high_level/vae_optuna_fu_10.sh|log/DiHFT/fu/high_level/optuna/10min_parallel/optuna.log"
    "高层启发式路由策略筛选与分析|FineFT/script/analysis/pick_agent/high_level_heurstic_fu_10.sh|log/analysis/pick_agent/DiHFT/fu/high_level_heurstic/10min_parallel.log"
    "高层路由测试集最终回测与评估|FineFT/script/test/DiHFT/high_level/final_result_fu_10_p.sh|log/DiHFT/fu/high_level/final_result/10min_parallel/final_result.log"
)

TOTAL_STEPS="${#PIPELINE_STEPS[@]}"

format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    if ((hours > 0)); then
        printf "%dh %dm %ds" "$hours" "$minutes" "$seconds"
    elif ((minutes > 0)); then
        printf "%dm %ds" "$minutes" "$seconds"
    else
        printf "%ds" "$seconds"
    fi
}

show_help() {
    cat <<EOF
fu (燃料油) 10min 全流程流水线执行脚本

用法:
  ./run_fu_10min_pipeline.sh [选项] [步骤编号/列表/区间]

参数支持以下格式:
  (无参数)                     顺序执行全部 1 至 ${TOTAL_STEPS} 步
  1,2,6 或 -s 1,2,6            执行指定的离散步骤 (逗号分隔，如: 1,2,6)
  1 2 6                        执行指定的离散步骤 (空格分隔)
  3                            从第 3 步开始执行至最后 (3 至 ${TOTAL_STEPS})
  3 5                          执行指定闭区间 (第 3 步至第 5 步)
  1-3,6                        区间与离散混合指定

选项:
  -s, --steps <LIST>   指定执行的步骤列表 (例如: -s 1,2,6 或 -s 1-3,6)
  -n, --dry-run        预检模式：仅打印执行步骤清单及调用命令，不实际运行
  -l, --list           列出全部 ${TOTAL_STEPS} 个步骤及其对应的脚本与日志路径
  -h, --help           显示此帮助信息

示例:
  ./run_fu_10min_pipeline.sh 1,2,6            # 仅依次运行步骤 1, 2, 6
  ./run_fu_10min_pipeline.sh 1 2 6            # 仅依次运行步骤 1, 2, 6
  ./run_fu_10min_pipeline.sh 3                # 从第 3 步开始运行至最后
  ./run_fu_10min_pipeline.sh 3 5              # 仅运行第 3 步至第 5 步
  ./run_fu_10min_pipeline.sh -n 1,2,6         # 预检预览步骤 1, 2, 6
EOF
}

list_steps() {
    echo "================================================================================"
    echo "fu 10min 全流程流水线步骤清单 (共 ${TOTAL_STEPS} 步):"
    echo "================================================================================"
    for ((i = 1; i <= TOTAL_STEPS; i++)); do
        local idx=$((i - 1))
        IFS='|' read -r step_title script_rel_path log_rel_path <<< "${PIPELINE_STEPS[idx]}"
        printf "[步骤 %d] %s
" "$i" "$step_title"
        printf "         脚本: %s
" "$script_rel_path"
        printf "         日志: %s
" "$log_rel_path"
    done
    echo "================================================================================"
}

trap 'echo -e "

[Pipeline] 收到中断信号，正在退出..."; exit 130' INT
trap 'echo -e "

[Pipeline] 收到终止信号，正在退出..."; exit 143' TERM

DRY_RUN=0
STEPS_ARG=""
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -l|--list)
            list_steps
            exit 0
            ;;
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        -s|--steps)
            if [[ $# -lt 2 ]]; then
                echo "错误: $1 选项需要参数 (例如: --steps 1,2,6)" >&2
                exit 1
            fi
            STEPS_ARG="$2"
            shift 2
            ;;
        --steps=*)
            STEPS_ARG="${1#*=}"
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

SELECTED_STEPS=()

add_selected_step() {
    local s=$1
    if ! [[ "$s" =~ ^[1-9]$ ]] || (( s < 1 || s > TOTAL_STEPS )); then
        echo "错误: 步骤编号必须为 1 到 ${TOTAL_STEPS} 之间的整数 (输入: ${s})" >&2
        exit 1
    fi
    for existing in "${SELECTED_STEPS[@]}"; do
        if [[ "$existing" -eq "$s" ]]; then
            return 0
        fi
    done
    SELECTED_STEPS+=("$s")
}

parse_step_spec() {
    local raw_spec=$1
    local part
    local -a tokens=()
    IFS=',' read -ra tokens <<< "$raw_spec"
    for part in "${tokens[@]}"; do
        part=$(echo "$part" | tr -d ' ')
        [ -n "$part" ] || continue
        if [[ "$part" =~ ^([1-9])-([1-9])$ ]]; then
            local start_range="${BASH_REMATCH[1]}"
            local end_range="${BASH_REMATCH[2]}"
            if (( start_range > end_range )); then
                echo "错误: 步骤范围无效 (${part}): 起始 (${start_range}) 大于结束 (${end_range})" >&2
                exit 1
            fi
            for ((k = start_range; k <= end_range; k++)); do
                add_selected_step "$k"
            done
        elif [[ "$part" =~ ^[1-9]$ ]]; then
            add_selected_step "$part"
        else
            echo "错误: 无法解析步骤规格 '${part}' (有效格式如: 1,2,6 或 1-3,6)" >&2
            exit 1
        fi
    done
}

if [[ -n "${STEPS_ARG}" ]]; then
    parse_step_spec "${STEPS_ARG}"
elif [[ ${#POSITIONAL[@]} -eq 0 ]]; then
    for ((k = 1; k <= TOTAL_STEPS; k++)); do
        SELECTED_STEPS+=("$k")
    done
elif [[ ${#POSITIONAL[@]} -eq 1 ]]; then
    arg="${POSITIONAL[0]}"
    if [[ "$arg" =~ ,|- ]]; then
        parse_step_spec "$arg"
    elif [[ "$arg" =~ ^[1-9]$ ]]; then
        for ((k = arg; k <= TOTAL_STEPS; k++)); do
            SELECTED_STEPS+=("$k")
        done
    else
        echo "错误: 无效的步骤参数: ${arg}" >&2
        exit 1
    fi
elif [[ ${#POSITIONAL[@]} -eq 2 ]]; then
    arg1="${POSITIONAL[0]}"
    arg2="${POSITIONAL[1]}"
    if [[ "$arg1" =~ ^[1-9]$ && "$arg2" =~ ^[1-9]$ ]]; then
        if (( arg1 > arg2 )); then
            echo "错误: 起始步骤 (${arg1}) 不能大于结束步骤 (${arg2})" >&2
            exit 1
        fi
        for ((k = arg1; k <= arg2; k++)); do
            SELECTED_STEPS+=("$k")
        done
    else
        for arg in "${POSITIONAL[@]}"; do
            parse_step_spec "$arg"
        done
    fi
else
    for arg in "${POSITIONAL[@]}"; do
        parse_step_spec "$arg"
    done
fi

if [[ ${#SELECTED_STEPS[@]} -eq 0 ]]; then
    echo "错误: 未指定任何有效步骤。" >&2
    exit 1
fi

# 预检：验证所需脚本文件均存在
for step_num in "${SELECTED_STEPS[@]}"; do
    idx=$((step_num - 1))
    IFS='|' read -r step_title script_rel_path log_rel_path <<< "${PIPELINE_STEPS[idx]}"
    script_abs_path="${ROOTPATH}/${script_rel_path}"
    if [[ ! -f "${script_abs_path}" ]]; then
        echo "错误: 脚本文件不存在: ${script_abs_path}" >&2
        exit 1
    fi
    if [[ ! -x "${script_abs_path}" ]]; then
        chmod +x "${script_abs_path}"
    fi
done

overall_start_time=$(date +%s)
echo "================================================================================"
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "🔍 [预检模式] fu 10min 全流程流水线"
else
    echo "🚀 开始执行 fu 10min 全流程流水线"
fi
echo "工作根目录: ${ROOTPATH}"
echo "当前环境: $(which python 2>/dev/null || echo '未检测到 python')"
echo "计划执行步骤: [ ${SELECTED_STEPS[*]} ] (共 ${#SELECTED_STEPS[@]} 步)"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

for step_num in "${SELECTED_STEPS[@]}"; do
    idx=$((step_num - 1))
    IFS='|' read -r step_title script_rel_path log_rel_path <<< "${PIPELINE_STEPS[idx]}"
    script_abs_path="${ROOTPATH}/${script_rel_path}"

    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "▶ [步骤 ${step_num}/${TOTAL_STEPS}] ${step_title}"
    echo "  脚本路径: ${script_rel_path}"
    echo "  主要日志: ${log_rel_path}"
    echo "  启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "--------------------------------------------------------------------------------"

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "  [DRY-RUN] 计划执行命令: bash "${script_abs_path}""
        continue
    fi

    step_start_time=$(date +%s)

    if ! bash "${script_abs_path}"; then
        step_end_time=$(date +%s)
        duration=$(format_duration $((step_end_time - step_start_time)))
        echo "" >&2
        echo "================================================================================" >&2
        echo "❌ [步骤 ${step_num}/${TOTAL_STEPS}] 执行失败: ${step_title} (耗时: ${duration})" >&2
        echo "  脚本路径: ${script_abs_path}" >&2
        echo "  日志位置: ${log_rel_path}" >&2
        echo "  流水线已终止。" >&2
        echo "================================================================================" >&2
        exit 1
    fi

    step_end_time=$(date +%s)
    duration=$(format_duration $((step_end_time - step_start_time)))
    echo "✔ [步骤 ${step_num}/${TOTAL_STEPS}] 完成: ${step_title} (耗时: ${duration})"
done

overall_end_time=$(date +%s)
total_duration=$(format_duration $((overall_end_time - overall_start_time)))
echo ""
echo "================================================================================"
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "🔍 [预检模式完成] 所有计划步骤检查通过！"
else
    echo "🎉 fu 10min 所选步骤已全部执行完成！"
fi
echo "已完成步骤: [ ${SELECTED_STEPS[*]} ]"
echo "总耗时: ${total_duration}"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
