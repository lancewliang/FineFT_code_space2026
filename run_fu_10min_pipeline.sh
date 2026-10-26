#!/usr/bin/env bash
set -euo pipefail

# Steps 7--10 share one frozen expert-hysteresis experiment.  This controller
# never scans historical output directories to fill missing configuration.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOTPATH="${ROOTPATH:-${SCRIPT_DIR}}"
cd "${ROOTPATH}"

DATASET_NAME=${DATASET_NAME:-fu}
BASE_PATH=${BASE_PATH:-dataset/10min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}
SAVE_PATH=${SAVE_PATH:-analysis_result/DiHFT/high_level_heurstic}
HIGH_LEVEL_RESULT_PATH=${HIGH_LEVEL_RESULT_PATH:-result/DiHFT/high_level}

declare -a PIPELINE_STEPS=(
    "数据预处理|data_preprocess/script_preprocess/future_upgraded/commodity/main_fu_10.sh"
    "FineFT 数据准备|FineFT/script/data/commodity_data_handler_fu_10.sh"
    "低层 Agent 并行训练|FineFT/script/train/train_commodity_fu_10.sh"
    "低层 Agent 并行测试|FineFT/script/test/DiHFT/low_level/test_util_fu_10.sh"
    "低层 Agent 二维联合筛选与分析|FineFT/script/analysis/pick_agent/low_level_fu_10.sh"
    "VAE 并行训练与评估|FineFT/script/train/DiHFT/low_level/VAE_util_fu_10.sh"
    "高层 expert_hysteresis 校准|FineFT/script/test/DiHFT/high_level/vae_optuna_fu_10.sh"
    "高层已选 Trial 发布与可视化|FineFT/script/analysis/pick_agent/high_level_heurstic_fu_10.sh"
    "主副合约同步评估与审计|FineFT/script/test/DiHFT/high_level/final_result_fu_10_main_sub.sh"
    "最终 test 评估与审计|FineFT/script/test/DiHFT/high_level/final_result_fu_10.sh"
)
TOTAL_STEPS=${#PIPELINE_STEPS[@]}

step_log_path() {
    case "$1" in
        1) printf '%s' 'log_futures/ticker_result/commodity/fu_10min_*.log' ;;
        2) printf '%s' 'terminal standard output' ;;
        3) printf '%s' "log/DiHFT/${DATASET_NAME}/low_level/train/10min/${EXPERIMENT_NAME}/*.log" ;;
        4) printf '%s' "log/DiHFT/${DATASET_NAME}/low_level/test/${EXPERIMENT_NAME}/*.log" ;;
        5) printf '%s' "log/analysis/pick_agent/DiHFT/${DATASET_NAME}/${EXPERIMENT_NAME}.log" ;;
        6) printf '%s' "log/DiHFT/${DATASET_NAME}/VAE/${EXPERIMENT_NAME}/*.log" ;;
        7) printf '%s' "log/DiHFT/${DATASET_NAME}/high_level/expert_hysteresis/${EXPERIMENT_NAME}/valid/calibration/${CALIBRATION_RUN_ID:-<CALIBRATION_RUN_ID>}/optuna.log" ;;
        8) printf '%s' "log/DiHFT/${DATASET_NAME}/high_level/expert_hysteresis/${EXPERIMENT_NAME}/valid/selected_trial_publication/publication.log" ;;
        9) printf '%s' "log/DiHFT/${DATASET_NAME}/high_level/final_result_main_sub/${EXPERIMENT_NAME}/<config_hash>/${EVAL_STAGE:-valid}/final_result.log" ;;
        10) printf '%s' "log/DiHFT/${DATASET_NAME}/high_level/final_result/${EXPERIMENT_NAME}/<config_hash>/test/final_result.log" ;;
    esac
}

show_help() {
    cat <<EOF
fu (燃料油) 10min 全流程流水线

用法: ./run_fu_10min_pipeline.sh [选项] [步骤编号/列表/区间]

无参数默认按顺序执行 1-${TOTAL_STEPS}（包含最终 test 步骤 10）。
只执行验证链请显式指定 1-9；步骤 10 只消费冻结配置，绝不重新选参。

  1,2,6 或 -s 1,2,6    离散步骤
  7-10                 闭区间
  1 9                  闭区间
  3                    从第 3 步到最后

  -s, --steps <LIST>   例如 -s 7,8,9,10 或 -s 7-10
  -n, --dry-run        仅显示计划；不激活环境、不启动任务、不写入产物
  -l, --list           列出步骤、脚本和日志路径；不修改权限
  -h, --help           显示帮助

实际运行前，步骤 7 要求 EXPERIMENT_MANIFEST、CALIBRATION_RUN_ID 和
SELECTION_MANIFEST；步骤 8--10 使用步骤 7 的冻结配置/选择证据，步骤
9--10 还必须有步骤 8 的发布回执。
EOF
}

list_steps() {
    local step idx title path
    for ((step = 1; step <= TOTAL_STEPS; step++)); do
        idx=$((step - 1)); IFS='|' read -r title path <<< "${PIPELINE_STEPS[idx]}"
        printf '[步骤 %d] %s\n  脚本: %s\n  日志: %s\n' "$step" "$title" "$path" "$(step_log_path "$step")"
    done
}

DRY_RUN=0
STEPS_ARG=""
POSITIONAL=()
while (($#)); do
    case "$1" in
        -h|--help) show_help; exit 0 ;;
        -l|--list) list_steps; exit 0 ;;
        -n|--dry-run) DRY_RUN=1; shift ;;
        -s|--steps) (($# >= 2)) || { echo "$1 requires a step list" >&2; exit 2; }; STEPS_ARG=$2; shift 2 ;;
        --steps=*) STEPS_ARG=${1#*=}; shift ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

SELECTED_STEPS=()
add_step() {
    local step=$1 existing
    [[ "$step" =~ ^[0-9]+$ ]] && ((step >= 1 && step <= TOTAL_STEPS)) || { echo "Invalid step: $step" >&2; exit 2; }
    for existing in "${SELECTED_STEPS[@]}"; do [[ "$existing" == "$step" ]] && return; done
    SELECTED_STEPS+=("$step")
}
parse_step_spec() {
    local spec=$1 token part start end
    local -a parts
    IFS=',' read -r -a parts <<< "$spec"
    for token in "${parts[@]}"; do
        part=${token//[[:space:]]/}; [[ -n "$part" ]] || continue
        if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start=${BASH_REMATCH[1]}; end=${BASH_REMATCH[2]}
            ((start <= end)) || { echo "Invalid descending range: $part" >&2; exit 2; }
            for ((step = start; step <= end; step++)); do add_step "$step"; done
        else
            add_step "$part"
        fi
    done
}
if [[ -n "$STEPS_ARG" ]]; then
    parse_step_spec "$STEPS_ARG"
elif ((${#POSITIONAL[@]} == 0)); then
    for ((step = 1; step <= TOTAL_STEPS; step++)); do add_step "$step"; done
elif ((${#POSITIONAL[@]} == 1)) && [[ "${POSITIONAL[0]}" =~ ^[0-9]+$ ]]; then
    for ((step = POSITIONAL[0]; step <= TOTAL_STEPS; step++)); do add_step "$step"; done
elif ((${#POSITIONAL[@]} == 2)) && [[ "${POSITIONAL[0]}" =~ ^[0-9]+$ && "${POSITIONAL[1]}" =~ ^[0-9]+$ ]]; then
    ((POSITIONAL[0] <= POSITIONAL[1])) || { echo "Invalid descending range." >&2; exit 2; }
    for ((step = POSITIONAL[0]; step <= POSITIONAL[1]; step++)); do add_step "$step"; done
else
    for spec in "${POSITIONAL[@]}"; do parse_step_spec "$spec"; done
fi

# Preserve the dependency order when a comma-separated list is out of order.
ORDERED_STEPS=()
for ((step = 1; step <= TOTAL_STEPS; step++)); do
    for selected in "${SELECTED_STEPS[@]}"; do [[ "$selected" == "$step" ]] && ORDERED_STEPS+=("$step"); done
done
SELECTED_STEPS=("${ORDERED_STEPS[@]}")
step_selected() { local wanted=$1 selected; for selected in "${SELECTED_STEPS[@]}"; do [[ "$selected" == "$wanted" ]] && return 0; done; return 1; }

for step in "${SELECTED_STEPS[@]}"; do
    idx=$((step - 1)); IFS='|' read -r title script_rel_path <<< "${PIPELINE_STEPS[idx]}"
    [[ -f "${ROOTPATH}/${script_rel_path}" ]] || { echo "Missing step $step script: ${ROOTPATH}/${script_rel_path}" >&2; exit 2; }
done

show_plan() {
    echo "ROOTPATH=${ROOTPATH}"
    echo "DATASET_NAME=${DATASET_NAME} EXPERIMENT_NAME=${EXPERIMENT_NAME} BASE_PATH=${BASE_PATH}"
    echo "Planned steps: ${SELECTED_STEPS[*]}"
    echo "EXPERIMENT_MANIFEST=${EXPERIMENT_MANIFEST:-<required by step 7>}"
    echo "CALIBRATION_RUN_ID=${CALIBRATION_RUN_ID:-<required by step 7>}"
    echo "SELECTION_EVIDENCE=${SELECTION_EVIDENCE:-<produced by step 7 or explicit>}"
    echo "FROZEN_CONFIG=${FROZEN_CONFIG:-<produced by step 7 or explicit>}"
    echo "PUBLICATION_DIR=${PUBLICATION_DIR:-${SAVE_PATH}/${DATASET_NAME}/${EXPERIMENT_NAME}/selected_trial_publication}"
    for step in "${SELECTED_STEPS[@]}"; do
        idx=$((step - 1)); IFS='|' read -r title script_rel_path <<< "${PIPELINE_STEPS[idx]}"
        printf '[%d] %s\n    bash %q\n    log: %s\n' "$step" "$title" "${ROOTPATH}/${script_rel_path}" "$(step_log_path "$step")"
    done
}
if ((DRY_RUN)); then
    echo "DRY-RUN: no environment activation, task, chmod, or artifact write will occur."
    show_plan
    exit 0
fi

# shellcheck source=FineFT/script/lib/require_finetf.sh
source "${ROOTPATH}/FineFT/script/lib/require_finetf.sh"
require_finetf
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT:${ROOTPATH}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}"

require_file_env() { local name=$1 value=${!1:-}; [[ -n "$value" && -f "$value" ]] || { echo "$name must name an existing file (got: ${value:-unset})." >&2; exit 2; }; }
require_dir_env() { local name=$1 value=${!1:-}; [[ -n "$value" && -d "$value" ]] || { echo "$name must name an existing directory (got: ${value:-unset})." >&2; exit 2; }; }
read_experiment_id() {
    python - "$EXPERIMENT_MANIFEST" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as source:
    experiment_id = json.load(source).get("experiment_id")
if not isinstance(experiment_id, str) or not experiment_id:
    raise SystemExit("EXPERIMENT_MANIFEST lacks experiment_id")
print(experiment_id)
PY
}
configure_step7_outputs() {
    local experiment_id calibration_root
    experiment_id=$(read_experiment_id) || { echo "Invalid EXPERIMENT_MANIFEST." >&2; exit 2; }
    calibration_root="${HIGH_LEVEL_RESULT_PATH}/${DATASET_NAME}/${EXPERIMENT_NAME}/expert_hysteresis/${experiment_id}/valid/calibration/${CALIBRATION_RUN_ID}"
    export SELECTION_EVIDENCE="${SELECTION_EVIDENCE:-${calibration_root}/selection_evidence.json}"
    export FROZEN_CONFIG="${FROZEN_CONFIG:-${calibration_root}/selected_expert_hysteresis_config.json}"
}
configure_publication_dir() { export PUBLICATION_DIR="${PUBLICATION_DIR:-${SAVE_PATH}/${DATASET_NAME}/${EXPERIMENT_NAME}/selected_trial_publication}"; }
validate_publication_receipt() { require_dir_env PUBLICATION_DIR; [[ -f "${PUBLICATION_DIR}/step8_publication_receipt.json" ]] || { echo "Step 8 publication receipt is missing: ${PUBLICATION_DIR}/step8_publication_receipt.json" >&2; exit 2; }; }

if step_selected 7; then
    require_file_env EXPERIMENT_MANIFEST
    [[ -n "${CALIBRATION_RUN_ID:-}" ]] || { echo "CALIBRATION_RUN_ID is required for step 7." >&2; exit 2; }
    SELECTION_MANIFEST=${SELECTION_MANIFEST:-analysis_result/DiHFT/low_level/${DATASET_NAME}/${EXPERIMENT_NAME}/two_dimensional_selection/two_dimensional_selection_manifest.json}
    export SELECTION_MANIFEST
    require_file_env SELECTION_MANIFEST
    configure_step7_outputs
fi
configure_publication_dir
if step_selected 7 && { step_selected 9 || step_selected 10; } && ! step_selected 8; then
    echo "Steps 9/10 after a new step 7 require step 8 in the same run to publish that configuration." >&2
    exit 2
fi
if step_selected 8 && ! step_selected 7; then require_file_env SELECTION_EVIDENCE; fi
if { step_selected 9 || step_selected 10; } && ! step_selected 7; then require_file_env FROZEN_CONFIG; fi
if { step_selected 9 || step_selected 10; } && ! step_selected 8; then validate_publication_receipt; fi

ACTIVE_CHILD_PID=""
stop_pipeline() {
    local signal=$1 code=$2
    if [[ -n "$ACTIVE_CHILD_PID" ]]; then
        echo "Received $signal; stopping pipeline child group $ACTIVE_CHILD_PID." >&2
        kill -"$signal" -- "-$ACTIVE_CHILD_PID" 2>/dev/null || true
        wait "$ACTIVE_CHILD_PID" 2>/dev/null || true
    fi
    exit "$code"
}
trap 'stop_pipeline INT 130' INT
trap 'stop_pipeline TERM 143' TERM
run_step() {
    local script_abs=$1 status=0
    command -v setsid >/dev/null 2>&1 || { echo "setsid is required for pipeline supervision." >&2; return 1; }
    setsid bash "$script_abs" & ACTIVE_CHILD_PID=$!
    wait "$ACTIVE_CHILD_PID" || status=$?
    ACTIVE_CHILD_PID=""
    return "$status"
}

show_plan
for step in "${SELECTED_STEPS[@]}"; do
    idx=$((step - 1)); IFS='|' read -r title script_rel_path <<< "${PIPELINE_STEPS[idx]}"
    echo "Starting step $step/$TOTAL_STEPS: $title"
    echo "  log: $(step_log_path "$step")"
    if ! run_step "${ROOTPATH}/${script_rel_path}"; then
        echo "Step $step failed: $title" >&2
        exit 1
    fi
    if [[ "$step" == 7 ]]; then require_file_env SELECTION_EVIDENCE; require_file_env FROZEN_CONFIG; fi
    if [[ "$step" == 8 ]]; then validate_publication_receipt; fi
    echo "Completed step $step: $title"
done
