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

#### Scenario: Commodity scale save reads filtered feature-selection input
- **WHEN** commodity full process has written `FEATURE_SELECTION/5min/fu/valid/fu2601/df.feather` and `FEATURE_SELECTION/5min/fu/valid/state_features.npy`
- **THEN** `scale_save.py` SHALL support reading that filtered input path for contract `fu2601`
- **AND** `scale_save.py` SHALL keep writing final outputs to `SCALE_SAVE/fu/fu2601/5min/{start_date}-{end_date}/`
- **AND** reward/execution columns remain unscaled and state columns remain scaled according to existing scale rules
- **AND** successful output file names remain `df.feather`, `df.csv`, `state_features.npy`, and `df_describe.csv`
