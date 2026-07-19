## MODIFIED Requirements

### Requirement: Polars scale save
The system SHALL migrate scale/save processing to Polars while preserving reward/state output contracts.

#### Scenario: Scale save output compatibility
- **WHEN** `scale_save.py` reads `IC_RESULT/<symbol>/<freq>/<start>-<end>/<df_name>.feather`
- **THEN** it writes `SCALE_SAVE/<symbol>/<freq>/<start>-<end>/df.feather`
- **AND** it writes `state_features.npy`
- **AND** it writes `df_describe.csv`
- **AND** reward/execution columns remain unscaled and state columns remain scaled according to the existing scale rules

#### Scenario: Commodity scale save uses manifest columns
- **WHEN** `scale_save.py` runs with `--market_type commodity_futures --orderbook_depth 5`
- **THEN** reward/execution columns are selected from the commodity manifest for depth 5
- **AND** the implementation does not assume the first 106 columns are reward/execution columns

#### Scenario: Commodity multi-contract scale save reads split-stage input with train feature list
- **WHEN** commodity full process has written `FEATURE_SELECTION/5min/fu/train/state_features.npy`
- **AND** `SPLIT-TRAIN-VALID-TEST/5min/fu/train/fu2601.feather` exists
- **AND** `SPLIT-TRAIN-VALID-TEST/5min/fu/valid/fu2601.feather` does not exist
- **THEN** `muti_contract_scale_save.py` SHALL support reading the existing split-stage input path for contract `fu2601`
- **AND** `muti_contract_scale_save.py` SHALL use `FEATURE_SELECTION/5min/fu/train/state_features.npy` as the state feature list
- **AND** `muti_contract_scale_save.py` SHALL NOT require an output for the missing `valid/fu2601.feather` stage
- **AND** `muti_contract_scale_save.py` SHALL continue processing all discovered split-stage inputs
- **AND** `muti_contract_scale_save.py` SHALL keep writing final feather outputs under `SCALE_SAVE/fu/5min/{stage}/{contract}.feather`
- **AND** `muti_contract_scale_save.py` SHALL write a debug csv next to each feather output under `SCALE_SAVE/fu/5min/{stage}/{contract}.csv`
- **AND** reward/execution columns remain unscaled and state columns remain scaled according to existing scale rules
