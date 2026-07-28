ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

export PYTHONPATH="${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"
DATASET_NAME=${DATASET_NAME:-fu}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-30min_multi}
BASE_PATH=${BASE_PATH:-dataset/30min}
POSITION_CHOICES=${POSITION_CHOICES:-5}
NUM_LABEL=${NUM_LABEL:-5}
mkdir -p "log/analysis/pick_agent/DiHFT/${DATASET_NAME}"

nohup python FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py \
    --dataset_name "${DATASET_NAME}" --experiment_name "${EXPERIMENT_NAME}" \
    --base_path "${BASE_PATH}" --position_choices "${POSITION_CHOICES}" --num_label "${NUM_LABEL}" \
    --epoch_num 100 --initial_position 0 \
    >"log/analysis/pick_agent/DiHFT/${DATASET_NAME}/${EXPERIMENT_NAME}.log" 2>&1 &
