#!/usr/bin/env bash
set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
SYMBOL=${SYMBOL:-fu}
TARGET_FREQ=${TARGET_FREQ:-5min}
START_DATE=${START_DATE:-2023-01-01}
END_DATE=${END_DATE:-2026-03-01}
TRAIN_RATIO=${TRAIN_RATIO:-5}
VALID_RATIO=${VALID_RATIO:-3}
TEST_RATIO=${TEST_RATIO:-2}
OUTPUT_ROOT=${OUTPUT_ROOT:-dataset/${TARGET_FREQ}}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
cd "${ROOTPATH}"

PYTHONPATH="${ROOTPATH}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}" \
python -m operator_futures.dataset_split.dataset_split \
  --summary_path "PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/${SYMBOL}/main_contract_summary.json" \
  --input_root "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE" \
  --output_root "${OUTPUT_ROOT}" \
  --symbol "${SYMBOL}" \
  --target_freq "${TARGET_FREQ}" \
  --start_date "${START_DATE}" \
  --end_date "${END_DATE}" \
  --train_ratio "${TRAIN_RATIO}" \
  --valid_ratio "${VALID_RATIO}" \
  --test_ratio "${TEST_RATIO}"
