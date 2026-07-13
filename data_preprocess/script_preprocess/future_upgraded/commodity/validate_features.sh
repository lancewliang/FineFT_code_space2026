#!/usr/bin/env bash
set -euo pipefail

ROOTPATH=$(pwd)
SYMBOL=fu
TARGET_FREQ=5min
START_DATE=2025-11-03
END_DATE=2025-11-08
REPORT_DIR=
SUMMARY_PATH=

while [ "$#" -gt 0 ]; do
    case "$1" in
        --root_path)
            ROOTPATH=$2
            shift 2
            ;;
        --symbol)
            SYMBOL=$2
            shift 2
            ;;
        --target_freq)
            TARGET_FREQ=$2
            shift 2
            ;;
        --start_date)
            START_DATE=$2
            shift 2
            ;;
        --end_date)
            END_DATE=$2
            shift 2
            ;;
        --report_dir)
            REPORT_DIR=$2
            shift 2
            ;;
        --summary)
            SUMMARY_PATH=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "${REPORT_DIR}" ]; then
    REPORT_DIR="${ROOTPATH}/log_futures/feature_validation"
fi

mkdir -p "${REPORT_DIR}"

if [ -z "${SUMMARY_PATH}" ]; then
    SUMMARY_PATH="${ROOTPATH}/PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/${SYMBOL}/main_contract_summary.json"
fi

run_commodity_summary_contracts() {
    local summary_path=$1
    local pythonpath="${ROOTPATH}/data_preprocess${PYTHONPATH:+:${PYTHONPATH}}"
    PYTHONPATH="${pythonpath}" python - "${summary_path}" <<'PY'
import sys
from pathlib import Path

from operator_futures.commodity.main_contract import load_main_contract_summary

summary = load_main_contract_summary(Path(sys.argv[1]))
for item in summary.contracts:
    print(item.contract)
PY
}

if [ -f "${SUMMARY_PATH}" ]; then
    missing=0
    while IFS= read -r contract; do
        [ -n "${contract}" ] || continue
        scale_file="${ROOTPATH}/PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/${SYMBOL}/${contract}/${TARGET_FREQ}/${START_DATE}-${END_DATE}/df.feather"
        if [ -f "${scale_file}" ]; then
            echo "Validated commodity contract output: symbol=${SYMBOL} contract=${contract} path=${scale_file}"
        else
            echo "Missing commodity contract output: symbol=${SYMBOL} contract=${contract} path=${scale_file}" >&2
            missing=1
        fi
    done < <(run_commodity_summary_contracts "${SUMMARY_PATH}")
    feature_union_dir="${ROOTPATH}/PREPROCESS_DATASET/commodity-futures/FEATURE_UNION/${SYMBOL}/${TARGET_FREQ}/${START_DATE}-${END_DATE}"
    if [ -f "${feature_union_dir}/state_features.npy" ]; then
        echo "Validated commodity feature union state features: symbol=${SYMBOL} path=${feature_union_dir}/state_features.npy"
    else
        echo "Missing commodity feature union state_features.npy: path=${feature_union_dir}/state_features.npy" >&2
        missing=1
    fi
    if [ -f "${feature_union_dir}/feature_union_manifest.json" ]; then
        echo "Validated commodity feature union manifest: symbol=${SYMBOL} path=${feature_union_dir}/feature_union_manifest.json"
    else
        echo "Missing commodity feature union manifest: path=${feature_union_dir}/feature_union_manifest.json" >&2
        missing=1
    fi
    exit "${missing}"
fi

PYTHONPATH="${ROOTPATH}/data_preprocess" python -m operator_futures.feature_validation.validate_features \
    --root_path "${ROOTPATH}" \
    --symbol "${SYMBOL}" \
    --target_freq "${TARGET_FREQ}" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --report_dir "${REPORT_DIR}"
