function run_ic_correlation {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local ROOTPATH=$5
    local logdir="log_futures/ic/${target_freq}/${symbol}"

    # Check if the log directory exists, create if not
    if [ ! -d "$logdir" ]; then
        mkdir -p "$logdir"
    fi

    # Execute the Python script in the background
    nohup python -u operator_futures/feature_selection/ic_correlation.py \
        --symbols $symbol --target_freq $target_freq --start_date $start_date --end_date $end_date --root_path $ROOTPATH \
        >"$logdir/${start_date}_${end_date}.log" 2>&1 &
    local pid=$!
    echo "All processes of calculating ic have been initiated."
    wait $pid

    echo "All processes of calculating ic have been completed."
}
function run_catboost_correlation {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local ROOTPATH=$5
    local logdir="log_futures/catboost/${target_freq}/${symbol}"

    # Check if the log directory exists, create if not
    if [ ! -d "$logdir" ]; then
        mkdir -p "$logdir"
    fi

    # Execute the Python script in the background
    nohup python -u operator_futures/feature_selection/catbooost.py \
        --symbols $symbol --target_freq $target_freq --start_date $start_date --end_date $end_date --root_path $ROOTPATH \
        >"$logdir/${start_date}_${end_date}.log" 2>&1 &
    local pid=$!
    echo "All processes of calculating ic have been initiated."
    wait $pid

    echo "All processes of calculating ic have been completed."
}
function run_rank_correlation {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local ROOTPATH=$5
    local logdir="log_futures/rank_ic/${target_freq}/${symbol}"

    # Check if the log directory exists, create if not
    if [ ! -d "$logdir" ]; then
        mkdir -p "$logdir"
    fi

    # Execute the Python script in the background
    nohup python -u operator_futures/feature_selection/rank_ic_correlation.py \
        --symbols $symbol --target_freq $target_freq --start_date $start_date --end_date $end_date --root_path $ROOTPATH \
        >"$logdir/${start_date}_${end_date}.log" 2>&1 &
    local pid=$!
    echo "All processes of calculating ic have been initiated."
    wait $pid

    echo "All processes of calculating ic have been completed."
}







































function run_ic_correlation_none_wait {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local ROOTPATH=$5
    local logdir="log_futures/ic/${target_freq}/${symbol}"

    # Check if the log directory exists, create if not
    if [ ! -d "$logdir" ]; then
        mkdir -p "$logdir"
    fi

    # Execute the Python script in the background
    nohup python -u operator_futures/feature_selection/ic_correlation.py \
        --symbols $symbol --target_freq $target_freq --start_date $start_date --end_date $end_date --root_path $ROOTPATH \
        >"$logdir/${start_date}_${end_date}.log" 2>&1 &
    local pid=$!
    echo "All processes of calculating ic have been initiated."

    echo "All processes of calculating ic have been completed."
}
function run_catboost_correlation_none_wait {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local ROOTPATH=$5
    local logdir="log_futures/catboost/${target_freq}/${symbol}"

    # Check if the log directory exists, create if not
    if [ ! -d "$logdir" ]; then
        mkdir -p "$logdir"
    fi

    # Execute the Python script in the background
    nohup python -u operator_futures/feature_selection/catbooost.py \
        --symbols $symbol --target_freq $target_freq --start_date $start_date --end_date $end_date --root_path $ROOTPATH \
        >"$logdir/${start_date}_${end_date}.log" 2>&1 &
    local pid=$!
    echo "All processes of calculating ic have been initiated."

    echo "All processes of calculating ic have been completed."
}
function run_rank_correlation_none_wait {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local ROOTPATH=$5
    local logdir="log_futures/rank_ic/${target_freq}/${symbol}"

    # Check if the log directory exists, create if not
    if [ ! -d "$logdir" ]; then
        mkdir -p "$logdir"
    fi

    # Execute the Python script in the background
    nohup python -u operator_futures/feature_selection/rank_ic_correlation.py \
        --symbols $symbol --target_freq $target_freq --start_date $start_date --end_date $end_date --root_path $ROOTPATH \
        >"$logdir/${start_date}_${end_date}.log" 2>&1 &
    local pid=$!
    echo "All processes of calculating ic have been initiated."

    echo "All processes of calculating ic have been completed."
}
