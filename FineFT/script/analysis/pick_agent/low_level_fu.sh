ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

export PYTHONPATH="${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p log/analysis/pick_agent/DiHFT/fu

nohup python FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py \
    --dataset_name fu --experiment_name 5min_nstep6_costw5 --base_path dataset/5min --position_choices 3 \
    >log/analysis/pick_agent/DiHFT/fu/5min_nstep6_costw5.log 2>&1 &



nohup python FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py \
    --dataset_name fu --experiment_name 10min_nstep6_costw5 \
    --base_path dataset/10min --position_choices 3 --num_label 4 \
    >log/analysis/pick_agent/DiHFT/fu/10min_nstep6_costw5.log 2>&1 &
