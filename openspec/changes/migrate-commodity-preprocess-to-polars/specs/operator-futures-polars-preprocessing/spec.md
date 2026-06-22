## ADDED Requirements

### Requirement: Shared post-merge preprocessing is pandas-free
The system SHALL migrate the shared post-merge preprocessing modules to Polars without importing pandas in the target runtime files.

#### Scenario: Target files do not import pandas
- **WHEN** maintainers inspect the migrated target files
- **THEN** `data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py` SHALL NOT import pandas
- **AND** `data_preprocess/operator_futures/time_operator/multi_processing_util.py` SHALL NOT import pandas
- **AND** `data_preprocess/operator_futures/scale_describe_save/scale_save.py` SHALL NOT import pandas
- **AND** every Python file directly under `data_preprocess/operator_futures/feature_selection/` SHALL NOT import pandas

#### Scenario: Existing entry points remain stable
- **WHEN** users run the existing post-merge preprocessing scripts with current CLI arguments
- **THEN** module paths, CLI option names, default output locations, and generated file names remain compatible
- **AND** no pandas fallback flag or pandas/Polars runtime switch is required

### Requirement: Polars time feature generation
The system SHALL generate time features with Polars-native operations while preserving the existing time feature contract.

#### Scenario: Time feature output compatibility
- **WHEN** `create_feature_multi_processing.py` reads `MERGE_CONCAT/CONCAT_FEATURE/<symbol>/<freq>/<start>-<end>.feather`
- **THEN** it writes `TIME_FEATURE/<symbol>/<freq>/<start>-<end>.feather`
- **AND** generated feature column names match the existing pandas output contract
- **AND** generated rows keep timestamp alignment and do not use future rows
- **AND** floating point values compare within the agreed tolerance for focused fixture inputs

#### Scenario: Depth-aware price and size features
- **WHEN** time feature generation runs with `--orderbook_depth 5`
- **THEN** it generates price and size rolling features only for available levels up to depth 5
- **AND** it does not require `bid6_price`, `ask6_price`, `bid25_price`, or `ask25_price`

#### Scenario: Generic depth remains supported
- **WHEN** time feature generation runs with generic 25-level input and `--orderbook_depth 25`
- **THEN** it supports existing 25-level price and size feature columns

### Requirement: Polars feature selection
The system SHALL migrate feature selection scripts to Polars while preserving selected feature artifacts and model-boundary behavior.

#### Scenario: IC feature selection output compatibility
- **WHEN** `ic_correlation.py` reads `ALL_FEATURE/<symbol>/<freq>/<start>-<end>.feather`
- **THEN** it writes `IC_RESULT/<symbol>/<freq>/<start>-<end>/df.feather`
- **AND** it writes `state_features.npy`, `correlation.csv`, and `ic_window_<window>.json`
- **AND** commodity reward/execution columns continue to come from the configured orderbook depth
- **AND** selected feature names remain deterministic for focused fixture inputs

#### Scenario: Rank IC feature selection output compatibility
- **WHEN** `rank_ic_correlation.py` runs on an existing all-feature file
- **THEN** it writes the same output artifact names as before
- **AND** rank-correlation calculations use Polars or NumPy without pandas imports

#### Scenario: CatBoost and Lasso feature selection use pandas-free model boundaries
- **WHEN** `catbooost.py` or `lasso_linear.py` prepares model inputs
- **THEN** the script may convert Polars data to NumPy arrays or library-native structures
- **AND** the script SHALL NOT import pandas
- **AND** output feature files and selected-feature metadata remain compatible

#### Scenario: Remove-duplicates feature selection stays compatible
- **WHEN** `remove_duplicates_feature.py` runs on IC result artifacts
- **THEN** it reads and writes the same paths as before
- **AND** duplicate/correlation filtering behavior is covered by focused tests

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

### Requirement: Post-merge Polars verification
The system SHALL verify the post-merge Polars migration with focused tests and a commodity end-to-end smoke run.

#### Scenario: Focused tests cover compatibility
- **WHEN** maintainers run the focused data preprocessing tests
- **THEN** tests cover time feature column compatibility, feature selection artifacts, scale/save output, commodity depth 5, and generic depth 25 behavior

#### Scenario: Commodity five-day main passes
- **WHEN** maintainers run the commodity `main.sh` default five-day flow
- **THEN** the flow reaches `SCALE_SAVE`
- **AND** logs contain no `Traceback`, `FileNotFound`, or `KeyError`

#### Scenario: Performance benchmark is not required
- **WHEN** the migration is validated
- **THEN** no fixed runtime improvement or benchmark framework is required
- **AND** correctness, pandas-free target files, compatibility, and smoke success are sufficient
