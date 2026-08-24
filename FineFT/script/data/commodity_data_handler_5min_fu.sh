#!/usr/bin/env bash
set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
SYMBOL=${SYMBOL:-fu}
TARGET_FREQ=${TARGET_FREQ:-5min}
CHUNK_LENGTH=${CHUNK_LENGTH:-10000}
EARLY_STOP=${EARLY_STOP:-20}

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
