#!/usr/bin/env bash
set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
SYMBOL=${SYMBOL:-al}
TARGET_FREQ=${TARGET_FREQ:-30min}
CHUNK_LENGTH=${CHUNK_LENGTH:-8000}
EARLY_STOP=${EARLY_STOP:-2}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
cd "${ROOTPATH}"

python FineFT/datahandler/commodity_contract_dataset.py \
  --dataset_split_manifest_path "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/${TARGET_FREQ}/${SYMBOL}/dataset_split_manifest.json" \
  --input_root "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE" \
  --state_features_path "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/${TARGET_FREQ}/${SYMBOL}/train/state_features.npy" \
  --output_root "dataset/${TARGET_FREQ}" \
  --symbol "${SYMBOL}" \
  --target_freq "${TARGET_FREQ}" \
  --chunk_length "${CHUNK_LENGTH}" \
  --early_stop "${EARLY_STOP}"

python FineFT/datahandler/slice_model.py \
  --valid_dir "dataset/${TARGET_FREQ}/${SYMBOL}/valid" \
  --timestamp timestamp

python FineFT/datahandler/vae_data_creation.py \
  --base_path "dataset/${TARGET_FREQ}" \
  --dataset_name "${SYMBOL}" \
  --save_path "dataset/${TARGET_FREQ}"
