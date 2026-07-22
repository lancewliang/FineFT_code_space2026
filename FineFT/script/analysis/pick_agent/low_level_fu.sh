ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

export PYTHONPATH="${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p log/analysis/pick_agent/DiHFT/fu

nohup python FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py \
    --dataset_name fu --experiment_name default \
    --base_path dataset/10min --position_choices 3 \
    >log/analysis/pick_agent/DiHFT/fu/default.log 2>&1 &
