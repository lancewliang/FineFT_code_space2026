#!/usr/bin/env bash

set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate finetf 2>/dev/null || true
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_NAME=${DATASET_NAME:-fu}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-30min_multi_paper_formula3}
BASE_PATH=${BASE_PATH:-dataset/30min}
POSITION_CHOICES=${POSITION_CHOICES:-5}
NUM_LABELS=${NUM_LABELS:-${NUM_LABEL:-4}}
LCB_Z=${LCB_Z:-0.0}
MIN_MARGINAL_CONTRACTS=${MIN_MARGINAL_CONTRACTS:-1}
MIN_JOINT_CONTRACTS=${MIN_JOINT_CONTRACTS:-1}
MIN_POSITIVE_CONTRACT_RATIO=${MIN_POSITIVE_CONTRACT_RATIO:-0.40}
MIN_WORST_INITIAL_POSITION_RETURN=${MIN_WORST_INITIAL_POSITION_RETURN:--1.0}
MIN_WORST_INITIAL_POSITION_RETURN_V2=${MIN_WORST_INITIAL_POSITION_RETURN_V2:--3.0}
MIN_WORST_INITIAL_POSITION_RETURN_V3=${MIN_WORST_INITIAL_POSITION_RETURN_V3:--5.0}
MISSING_JOINT_POLICY=${MISSING_JOINT_POLICY:-slope_marginal_best}
CONTRACT_WEIGHTING=${CONTRACT_WEIGHTING:-step_weighted}
MIN_SLICE_STEPS=${MIN_SLICE_STEPS:-30}

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
    --min_worst_initial_position_return "${MIN_WORST_INITIAL_POSITION_RETURN}"
    --min_worst_initial_position_return_v2 "${MIN_WORST_INITIAL_POSITION_RETURN_V2}"
    --min_worst_initial_position_return_v3 "${MIN_WORST_INITIAL_POSITION_RETURN_V3}"
    --missing_joint_policy "${MISSING_JOINT_POLICY}"
    --contract_weighting "${CONTRACT_WEIGHTING}"
    --min_slice_steps "${MIN_SLICE_STEPS}"
)

nohup python -u FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py \
    "${CMD_ARGS[@]}" \
    >"log/analysis/pick_agent/DiHFT/${DATASET_NAME}/${EXPERIMENT_NAME}.log" 2>&1
