#!/usr/bin/env bash
set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
START_DATE=${START_DATE:-2025-11-03}
END_DATE=${END_DATE:-2025-11-08}
TARGET_FREQ=${TARGET_FREQ:-5min}
SYMBOL=${SYMBOL:-fu}
COMMODITY_NAME=${COMMODITY_NAME:-燃料油}
MAX_PROCESSES=${MAX_PROCESSES:-4}
echo "ROOTPATH: ${ROOTPATH}"
source "${ROOTPATH}/data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"

LOG_DIR="${ROOTPATH}/log_futures/ticker_result/commodity"
mkdir -p "${LOG_DIR}"

run_commodity_full_process \
    "${ROOTPATH}" \
    "${START_DATE}" \
    "${END_DATE}" \
    "${TARGET_FREQ}" \
    "${SYMBOL}" \
    "${COMMODITY_NAME}" \
    "${MAX_PROCESSES}" \
    >"${LOG_DIR}/${SYMBOL}_${TARGET_FREQ}_${START_DATE}_${END_DATE}.log" 2>&1
