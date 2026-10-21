#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate finetf 2>/dev/null || true
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_NAME=${DATASET_NAME:-fu}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}
BASE_PATH=${BASE_PATH:-dataset/10min}
POSITION_CHOICES=${POSITION_CHOICES:-3}
NUM_LABELS=${NUM_LABELS:-${NUM_LABEL:-3}}
LCB_Z=${LCB_Z:-1.0}
MIN_MARGINAL_CONTRACTS=${MIN_MARGINAL_CONTRACTS:-1}
MIN_JOINT_CONTRACTS=${MIN_JOINT_CONTRACTS:-3}
MIN_POSITIVE_CONTRACT_RATIO=${MIN_POSITIVE_CONTRACT_RATIO:-0.60}
MIN_MEAN_RETURN=${MIN_MEAN_RETURN:-0.0}
MIN_LCB=${MIN_LCB:-0.0}
MIN_WORST_INITIAL_POSITION_RETURN=${MIN_WORST_INITIAL_POSITION_RETURN:--0.15}
MIN_WORST_INITIAL_POSITION_RETURN_V2=${MIN_WORST_INITIAL_POSITION_RETURN_V2:--0.25}
MIN_WORST_INITIAL_POSITION_RETURN_V3=${MIN_WORST_INITIAL_POSITION_RETURN_V3:--0.35}
MISSING_JOINT_POLICY=${MISSING_JOINT_POLICY:-empty_model}
CONTRACT_WEIGHTING=${CONTRACT_WEIGHTING:-contract_equal}
MIN_SLICE_STEPS=${MIN_SLICE_STEPS:-30}
MIN_GRID_SLICE_STEPS=${MIN_GRID_SLICE_STEPS:-${MIN_SLICE_STEPS}}
ENABLE_CONJUNCTIVE_GATES=${ENABLE_CONJUNCTIVE_GATES:-1}
ENABLE_TRAFFIC_TIERING=${ENABLE_TRAFFIC_TIERING:-1}
CORE_SLOT_MIN_JOINT_CONTRACTS=${CORE_SLOT_MIN_JOINT_CONTRACTS:-3}
CORE_SLOT_MIN_POSITIVE_CONTRACT_RATIO=${CORE_SLOT_MIN_POSITIVE_CONTRACT_RATIO:-0.60}
PERIPHERAL_SLOT_MIN_JOINT_CONTRACTS=${PERIPHERAL_SLOT_MIN_JOINT_CONTRACTS:-3}
PERIPHERAL_SLOT_MIN_POSITIVE_CONTRACT_RATIO=${PERIPHERAL_SLOT_MIN_POSITIVE_CONTRACT_RATIO:-0.60}
ENABLE_MAIN_CONTRACT_SUB_GATE=${ENABLE_MAIN_CONTRACT_SUB_GATE:-1}
ENABLE_HARD_STOP_LOSS_GATE=${ENABLE_HARD_STOP_LOSS_GATE:-0}
MAX_HARD_STOP_LOSS_EPISODE_RATIO=${MAX_HARD_STOP_LOSS_EPISODE_RATIO:-0.30}
REWARD_METRIC=${REWARD_METRIC:-reward}
MIN_ENSEMBLE_TRAFFIC_COVERAGE=${MIN_ENSEMBLE_TRAFFIC_COVERAGE:-0.40}
ALLOW_SPARSE_ENSEMBLE=${ALLOW_SPARSE_ENSEMBLE:-0}

mkdir -p "log/analysis/pick_agent/DiHFT/${DATASET_NAME}"

CMD_ARGS=(
    --dataset_name "${DATASET_NAME}"
    --experiment_name "${EXPERIMENT_NAME}"
    --base_path "${BASE_PATH}"
    --position_choices "${POSITION_CHOICES}"
    --num_labels "${NUM_LABELS}"
    --lcb_z "${LCB_Z}"
    --min_marginal_contracts "${MIN_MARGINAL_CONTRACTS}"
    --min_joint_contracts "${MIN_JOINT_CONTRACTS}"
    --min_positive_contract_ratio "${MIN_POSITIVE_CONTRACT_RATIO}"
    --min_mean_return "${MIN_MEAN_RETURN}"
    --min_lcb "${MIN_LCB}"
    --hard_stop_loss_threshold "0.03"
    --min_worst_initial_position_return "${MIN_WORST_INITIAL_POSITION_RETURN}"
    --min_worst_initial_position_return_v2 "${MIN_WORST_INITIAL_POSITION_RETURN_V2}"
    --min_worst_initial_position_return_v3 "${MIN_WORST_INITIAL_POSITION_RETURN_V3}"
    --missing_joint_policy "${MISSING_JOINT_POLICY}"
    --contract_weighting "${CONTRACT_WEIGHTING}"
    --min_slice_steps "${MIN_SLICE_STEPS}"
    --min_grid_slice_steps "${MIN_GRID_SLICE_STEPS}"
    --max_hard_stop_loss_episode_ratio "${MAX_HARD_STOP_LOSS_EPISODE_RATIO}"
    --reward_metric "${REWARD_METRIC}"
    --min_ensemble_traffic_coverage "${MIN_ENSEMBLE_TRAFFIC_COVERAGE}"
)

if [ "${ENABLE_CONJUNCTIVE_GATES}" = "1" ]; then
    CMD_ARGS+=(--enable_conjunctive_gates)
fi
if [ "${ENABLE_TRAFFIC_TIERING}" = "1" ]; then
    CMD_ARGS+=(
        --enable_traffic_tiering
        --core_slot_min_joint_contracts "${CORE_SLOT_MIN_JOINT_CONTRACTS}"
        --core_slot_min_positive_contract_ratio "${CORE_SLOT_MIN_POSITIVE_CONTRACT_RATIO}"
        --peripheral_slot_min_joint_contracts "${PERIPHERAL_SLOT_MIN_JOINT_CONTRACTS}"
        --peripheral_slot_min_positive_contract_ratio "${PERIPHERAL_SLOT_MIN_POSITIVE_CONTRACT_RATIO}"
    )
fi
if [ "${ENABLE_MAIN_CONTRACT_SUB_GATE}" = "1" ]; then
    CMD_ARGS+=(--enable_main_contract_sub_gate)
fi
if [ "${ENABLE_HARD_STOP_LOSS_GATE}" = "1" ]; then
    CMD_ARGS+=(--enable_hard_stop_loss_gate)
fi
if [ "${ALLOW_SPARSE_ENSEMBLE}" = "1" ]; then
    CMD_ARGS+=(--allow_sparse_ensemble)
fi

nohup python -u FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py \
    "${CMD_ARGS[@]}" \
    >"log/analysis/pick_agent/DiHFT/${DATASET_NAME}/${EXPERIMENT_NAME}.log" 2>&1
