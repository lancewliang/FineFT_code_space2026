#!/usr/bin/env bash
set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
SYMBOL=${SYMBOL:-al}
TARGET_FREQ=${TARGET_FREQ:-5min}
CHUNK_LENGTH=${CHUNK_LENGTH:-8000}
EARLY_STOP=${EARLY_STOP:-10}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
cd "${ROOTPATH}"

python FineFT/datahandler/commodity_contract_dataset.py \
  --dataset_split_manifest_path "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/${TARGET_FREQ}/${SYMBOL}/dataset_split_manifest.json" \
  --input_root "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE" \
  --rl_state_features_path "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/${TARGET_FREQ}/${SYMBOL}/train/rl_state_features.npy" \
  --vae_slope_state_features_path "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/${TARGET_FREQ}/${SYMBOL}/train/vae_slope_state_features.npy" \
  --vae_volatility_state_features_path "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/${TARGET_FREQ}/${SYMBOL}/train/vae_volatility_state_features.npy" \
  --output_root "dataset/${TARGET_FREQ}" \
  --symbol "${SYMBOL}" \
  --target_freq "${TARGET_FREQ}" \
  --chunk_length "${CHUNK_LENGTH}" \
  --early_stop "${EARLY_STOP}"

python FineFT/datahandler/valid_cross_contract_label_calibration.py \
  --valid_dir "dataset/${TARGET_FREQ}/${SYMBOL}/valid" \
  --timestamp timestamp

python FineFT/datahandler/vae_data_creation.py \
  --base_path "dataset/${TARGET_FREQ}" \
  --dataset_name "${SYMBOL}" \
  --save_path "dataset/${TARGET_FREQ}" \
  --labeling_method "slope"

python FineFT/datahandler/vae_data_creation.py \
  --base_path "dataset/${TARGET_FREQ}" \
  --dataset_name "${SYMBOL}" \
  --save_path "dataset/${TARGET_FREQ}" \
  --labeling_method "volatility"

test -f "dataset/${TARGET_FREQ}/${SYMBOL}/rl_state_features.npy"
test -f "dataset/${TARGET_FREQ}/${SYMBOL}/vae_slope_state_features.npy"
test -f "dataset/${TARGET_FREQ}/${SYMBOL}/vae_volatility_state_features.npy"
