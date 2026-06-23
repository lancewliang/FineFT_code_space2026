#!/usr/bin/env bash
set -euo pipefail

ROOTPATH=$(pwd)
SYMBOL=fu
TARGET_FREQ=5min
START_DATE=2025-11-03
END_DATE=2025-11-08
REPORT_DIR=

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

PYTHONPATH="${ROOTPATH}/data_preprocess" python -m operator_futures.feature_validation.validate_features \
    --root_path "${ROOTPATH}" \
    --symbol "${SYMBOL}" \
    --target_freq "${TARGET_FREQ}" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --report_dir "${REPORT_DIR}"
