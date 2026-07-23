# Refactor Multi-Contract Scale Save Robust Scaler Design

## Goal
Replace the commodity split-stage multi-contract scaler with a train-only robust scaler while preserving the existing `SCALE_SAVE/{symbol}/{target_freq}/{stage}/{contract}.feather` and `.csv` output contract.

## Architecture
`muti_contract_scale_save.py` remains the public entry point for commodity split-stage scaling. Internally it will:

1. Discover train/valid/test split-stage feather inputs.
2. Fit scaler statistics once from all rows in the train split.
3. Persist a symbol+frequency manifest for the fitted scaler.
4. Apply the same manifest to every discovered split-stage file.
5. Write scaled feather/csv outputs in the existing `SCALE_SAVE` layout.

The robust scaler itself is train-only and deterministic. For each selected state feature it will record median, IQR, fallback scale, clipping configuration, and fit metadata. The implementation stays inside the commodity multi-contract path so the legacy `scale_save.py` contract remains untouched.

## Data Flow
1. Load `FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy`.
2. Scan `SPLIT-TRAIN-VALID-TEST/{target_freq}/{symbol}/train/*.feather`.
3. Concatenate train rows conceptually per feature and fit:
   - center = median
   - scale = IQR
   - fallback to std when IQR is too small
   - fallback to 1.0 when std is too small
4. Persist `scaler_manifest.json` and `scale_diagnostics.csv` under `SCALE_SAVE/{symbol}/{target_freq}/`.
5. Apply the manifest to each train/valid/test feather and write same-basename feather/csv outputs.
6. Preserve reward/execution columns, scaled state features, and `symbol` column order.

## Error Handling
- Fail fast if no train split inputs exist.
- Fail fast if the selected state feature list is empty or missing.
- Fail fast if any input file is missing a selected state feature.
- Fail fast if clip bounds are invalid.
- Fail fast if fitted statistics are non-finite.
- Continue when a stage directory is absent, because stage existence is data-dependent.

## Testing
- Unit-style coverage should exercise scaler fit/apply helpers directly with tiny synthetic frames.
- CLI coverage should verify train-only fitting, stable train/valid/test scaling, manifest and diagnostics output, and fail-fast behavior for missing inputs and invalid feature lists.
- Regression coverage should show that a train file and a test file with different raw volatility do not produce a per-file 10x scale jump in `wap_1` or `awap`.
