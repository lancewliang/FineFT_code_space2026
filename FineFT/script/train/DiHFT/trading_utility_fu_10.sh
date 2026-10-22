#!/usr/bin/env bash
set -euo pipefail

: "${UTILITY_EPISODES_CSV:?set UTILITY_EPISODES_CSV to candidate_episodes.csv}"
: "${UTILITY_OUTPUT_DIR:?set UTILITY_OUTPUT_DIR to an explicit artifact directory}"
: "${UTILITY_FEATURE_NAMES:?set UTILITY_FEATURE_NAMES in feature_values order}"
: "${UTILITY_CODE_VERSION:?set UTILITY_CODE_VERSION to the immutable source revision}"

python -m RL.DiHFT.trading_utility.train \
  --episodes-csv "${UTILITY_EPISODES_CSV}" \
  --output-dir "${UTILITY_OUTPUT_DIR}" \
  --feature-names "${UTILITY_FEATURE_NAMES}" \
  --prequential \
  --code-version "${UTILITY_CODE_VERSION}" \
  --random-seed "${UTILITY_RANDOM_SEED:-42}" \
  --backend "${UTILITY_BACKEND:-auto}"
