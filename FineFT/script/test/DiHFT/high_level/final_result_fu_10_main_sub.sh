#!/usr/bin/env bash

# Step 9: replay the selected expert policy on the declared validation or test
# segment. All execution/risk/account settings come from FROZEN_CONFIG.
set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
cd "$ROOTPATH"

# shellcheck source=FineFT/script/lib/require_finetf.sh
source "${ROOTPATH}/FineFT/script/lib/require_finetf.sh"
# shellcheck source=FineFT/script/lib/foreground_task.sh
source "${ROOTPATH}/FineFT/script/lib/foreground_task.sh"
require_finetf
install_task_signal_traps
export PYTHONPATH="${ROOTPATH}:${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_NAME=${DATASET_NAME:-fu}
BASE_PATH=${BASE_PATH:-dataset/10min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_parallel}
# Keep the runtime input defaults aligned with vae_optuna_fu_10.sh.  The
# frozen configuration remains authoritative for execution/risk parameters.
EXPERIMENT_MANIFEST=${EXPERIMENT_MANIFEST:-${ROOTPATH}/FineFT/script/test/DiHFT/high_level/experiment_manifest_fu_10.json}
SELECTION_MANIFEST=${SELECTION_MANIFEST:-analysis_result/DiHFT/low_level/${DATASET_NAME}/${EXPERIMENT_NAME}/two_dimensional_selection/two_dimensional_selection_manifest.json}
EVAL_STAGE=${1:-${EVAL_STAGE:-valid}}
FROZEN_CONFIG=${FROZEN_CONFIG:-analysis_result/DiHFT/high_level_heurstic/${DATASET_NAME}/${EXPERIMENT_NAME}/selected_trial_publication/frozen_config.json}
STEP8_PUBLICATION_DIR=${STEP8_PUBLICATION_DIR:-$(dirname "$FROZEN_CONFIG")}

if [[ ! -f "$FROZEN_CONFIG" ]]; then
    echo "FROZEN_CONFIG does not exist: $FROZEN_CONFIG" >&2
    exit 2
fi
if [[ "$EVAL_STAGE" != "valid" && "$EVAL_STAGE" != "test" ]]; then
    echo "EVAL_STAGE must be valid or test, got: $EVAL_STAGE" >&2
    exit 2
fi

STEP8_RECEIPT="${STEP8_PUBLICATION_DIR}/step8_publication_receipt.json"
if [[ ! -f "$STEP8_RECEIPT" ]]; then
    echo "Step-8 publication receipt does not exist: $STEP8_RECEIPT" >&2
    exit 2
fi
python - "$FROZEN_CONFIG" "$STEP8_RECEIPT" "$BASE_PATH" "$DATASET_NAME" \
    "$EXPERIMENT_NAME" "$EXPERIMENT_MANIFEST" "$SELECTION_MANIFEST" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    config_path,
    receipt_path,
    base_path,
    dataset_name,
    experiment_name,
    experiment_manifest,
    selection_manifest,
) = sys.argv[1:]
config = json.load(open(config_path, encoding="utf-8"))
receipt = json.load(open(receipt_path, encoding="utf-8"))
identity = config.pop("config_hash", None)
expected = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
if identity != expected or receipt.get("document_type") != "expert_hysteresis_step8_publication_receipt":
    raise SystemExit("invalid frozen config or Step-8 publication receipt")
if receipt.get("config_hash") != identity or receipt.get("experiment_identity") != config.get("experiment_identity"):
    raise SystemExit("Step-8 receipt does not bind this frozen configuration")
if receipt.get("selected_trial") != config.get("selection"):
    raise SystemExit("Step-8 receipt study/trial identity differs from frozen configuration")

configuration = config.get("configuration", {})
if (
    configuration.get("dataset_name") != dataset_name
    or configuration.get("experiment_name") != experiment_name
    or Path(configuration.get("base_path", "")).resolve() != Path(base_path).resolve()
):
    raise SystemExit("runtime dataset/base/experiment variables differ from frozen configuration")
if Path(config.get("experiment_manifest", "")).resolve() != Path(experiment_manifest).resolve():
    raise SystemExit("EXPERIMENT_MANIFEST differs from frozen configuration")
if Path(configuration.get("selection_manifest", "")).resolve() != Path(selection_manifest).resolve():
    raise SystemExit("SELECTION_MANIFEST differs from frozen configuration")
PY

CONFIG_HASH=$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["config_hash"])' "$FROZEN_CONFIG")
RESULT_DIR="result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/expert_hysteresis/${CONFIG_HASH}/${EVAL_STAGE}/main_sub"
LOG_DIR="log/DiHFT/${DATASET_NAME}/high_level/final_result_main_sub/${EXPERIMENT_NAME}/${CONFIG_HASH}/${EVAL_STAGE}"
LOG_FILE="${LOG_DIR}/final_result.log"
mkdir -p "$LOG_DIR"

echo "Running frozen expert_hysteresis main/sub evaluation: ${EVAL_STAGE}"
run_logged_foreground "${LOG_FILE}" \
    python -u FineFT/RL/DiHFT/high_level/vae_routing_final_result_macro_action.py \
    --frozen_config "$FROZEN_CONFIG" \
    --eval_stage "$EVAL_STAGE" \
    --evaluation_purpose main_sub \
    --result_path result/DiHFT/final_result

AUDIT_OUTPUT_DIR="analysis_result/DiHFT/final_result/${DATASET_NAME}/${EXPERIMENT_NAME}/expert_hysteresis/${CONFIG_HASH}/${EVAL_STAGE}/main_sub/daily_trading_audit"
AUDIT_ARGS=(
    --result_dir "$RESULT_DIR"
    --market_data_dir "${BASE_PATH}/${DATASET_NAME}/${EVAL_STAGE}"
    --output_dir "$AUDIT_OUTPUT_DIR"
)
if [[ -n "${TRADING_DAY_MAPPING:-}" ]]; then
    AUDIT_ARGS+=(--trading_day_mapping "$TRADING_DAY_MAPPING")
fi
if [[ -n "${RUN_ID:-}" ]]; then
    AUDIT_ARGS+=(--run_id "$RUN_ID")
fi
run_logged_foreground "${LOG_FILE}" python -u FineFT/analysis/diagnostics/daily_trading_audit.py "${AUDIT_ARGS[@]}"

echo "Completed: $RESULT_DIR"
