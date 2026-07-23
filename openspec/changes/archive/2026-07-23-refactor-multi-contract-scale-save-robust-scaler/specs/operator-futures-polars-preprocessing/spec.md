## MODIFIED Requirements

### Requirement: Operator futures Polars preprocessing compatibility
The system SHALL migrate operator-futures preprocessing paths under `data_preprocess/operator_futures` to Polars and preserve existing output contracts.

#### Scenario: Orderbook and derivative ticker downscale compatibility
- **WHEN** orderbook or derivative ticker downscale runs for a symbol, date, and target frequency supported by the existing scripts
- **THEN** the Polars implementation writes the same output path and Feather file name as the previous implementation
- **AND** output timestamps, row ordering, duplicate timestamp `first` behavior, forward fill behavior, column names, and column order match the previous contract

#### Scenario: Base feature generation compatibility
- **WHEN** quotes and trades are processed by `features_related/base_feature.py` and `feature_util.py`
- **THEN** quote counts, OHLC quote features, trade OHLCV features, side-grouped features, exchange column, symbol column, and timestamp alignment match the previous contract
- **AND** floating point feature values compare within `rtol=1e-12, atol=1e-12`

#### Scenario: Cross-section and time feature compatibility
- **WHEN** cross-section and time feature modules process existing intermediate Feather files
- **THEN** KLINE, QUOTES, SNAPSHOT, rolling time features, normalized features, generated feature columns, and output timestamps preserve the previous column names and order
- **AND** features that depend on market history do not use future timestamps

#### Scenario: Merge, concat, scale, and feature selection compatibility
- **WHEN** merge, concat, scale/save, or feature-selection preprocessing reads intermediate Feather files
- **THEN** inner joins, duplicate timestamp `first` semantics, future feature shift, forward fill, reward/execution column selection, state feature selection, and saved file paths remain compatible with existing downstream readers
- **AND** float outputs compare within `rtol=1e-12, atol=1e-12`

#### Scenario: Commodity multi-contract scale save reads split-stage input with train feature list
- **WHEN** commodity full process has written `FEATURE_SELECTION/5min/fu/train/state_features.npy`
- **AND** `SPLIT-TRAIN-VALID-TEST/5min/fu/train/fu2601.feather` exists
- **AND** `SPLIT-TRAIN-VALID-TEST/5min/fu/valid/fu2601.feather` does not exist
- **THEN** `muti_contract_scale_save.py` SHALL support reading the existing split-stage input path for contract `fu2601`
- **AND** `muti_contract_scale_save.py` SHALL use `FEATURE_SELECTION/5min/fu/train/state_features.npy` as the state feature list
- **AND** `muti_contract_scale_save.py` SHALL fit a train-only robust scaler from all train split rows once, then apply that manifest to train/valid/test split inputs
- **AND** `muti_contract_scale_save.py` SHALL NOT require an output for the missing `valid/fu2601.feather` stage
- **AND** `muti_contract_scale_save.py` SHALL continue processing all discovered split-stage inputs
- **AND** `muti_contract_scale_save.py` SHALL keep writing final feather outputs under `SCALE_SAVE/fu/5min/{stage}/{contract}.feather`
- **AND** `muti_contract_scale_save.py` SHALL write a debug csv next to each feather output under `SCALE_SAVE/fu/5min/{stage}/{contract}.csv`
- **AND** `muti_contract_scale_save.py` SHALL write `SCALE_SAVE/fu/5min/scaler_manifest.json` and `SCALE_SAVE/fu/5min/scale_diagnostics.csv`
- **AND** reward/execution columns remain unscaled and state columns are scaled by the train-only manifest with default clip bounds `[-20, 20]`

## ADDED Requirements

### Requirement: Commodity split-stage scale-save manifest and diagnostics
The system SHALL make the train-only robust split-stage scaler auditable by writing a manifest and diagnostics file next to the stage outputs.

#### Scenario: Manifest records fit scope and per-feature statistics
- **WHEN** `muti_contract_scale_save.py` fits the train-only robust scaler for `fu`
- **THEN** `scaler_manifest.json` SHALL record `symbol`, `target_freq`, `scaler_version`, `feature_list_path`, `train_input_files`, `clip_min`, `clip_max`, and `row_count`
- **AND** the manifest SHALL record each selected feature's center, scale, scale method, fallback reason, and summary statistics needed to reproduce the fit

#### Scenario: Diagnostics records clipping and output accounting
- **WHEN** `muti_contract_scale_save.py` applies the manifest to a split-stage input file
- **THEN** `scale_diagnostics.csv` SHALL record `stage`, `contract`, `input_file`, `output_file`, `rows`, `state_feature_count`, `clip_enabled`, `total_clipped_values`, and per-file clip ratios
- **AND** the diagnostics SHALL make it possible to identify features or contracts that were clipped heavily without reading the feather outputs
