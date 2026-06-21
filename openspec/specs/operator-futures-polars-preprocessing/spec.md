# operator-futures-polars-preprocessing Specification

## Purpose
TBD - created by archiving change migrate-operator-futures-to-polars. Update Purpose after archive.
## Requirements
### Requirement: Polars preprocessing dependency
The system SHALL require Polars for core `data_preprocess/operator_futures` preprocessing while preserving the existing script-driven workflow.

#### Scenario: Dependency is reproducible
- **WHEN** a user installs the data preprocessing environment from `data_preprocess/requirements.txt`
- **THEN** the environment includes `polars` as a required dependency
- **AND** core `data_preprocess/operator_futures` preprocessing scripts can import Polars without an optional fallback path

#### Scenario: Existing entry points remain stable
- **WHEN** a user runs an existing `operator_futures` preprocessing module or shell script with the same CLI arguments as before
- **THEN** the module path, CLI argument names, default path values, output directory layout, and Feather file naming remain unchanged
- **AND** the internal DataFrame engine change does not require downstream script changes

### Requirement: Binance futures Polars preprocessing compatibility
The system SHALL migrate Binance futures preprocessing paths under `data_preprocess/operator_futures` to Polars and preserve existing output contracts.

#### Scenario: Orderbook and derivative ticker downscale compatibility
- **WHEN** Binance futures orderbook or derivative ticker downscale runs for a symbol, date, and target frequency supported by the existing scripts
- **THEN** the Polars implementation writes the same output path and Feather file name as the previous implementation
- **AND** output timestamps, row ordering, duplicate timestamp `first` behavior, forward fill behavior, column names, and column order match the previous contract

#### Scenario: Base feature generation compatibility
- **WHEN** Binance futures quotes and trades are processed by `features_related/base_feature.py` and `feature_util.py`
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

### Requirement: Output compatibility decisions
The system SHALL preserve preprocessing output compatibility and make any unresolved behavior difference explicit before implementation proceeds past the affected path.

#### Scenario: Feather schema and dtype are controllable
- **WHEN** a migrated preprocessing path writes a Feather file
- **THEN** the implementation explicitly arranges column order and key dtypes before writing when Polars inference would otherwise change the contract
- **AND** any unavoidable Arrow/Feather schema difference is recorded for user decision instead of being silently accepted

#### Scenario: Historical behavior appears incorrect
- **WHEN** migration reveals a previous pandas behavior that appears to be a bug or data-quality issue
- **THEN** the implementation records the file, function, input condition, previous output, Polars output, and recommended decision
- **AND** the implementation does not silently fix that behavior without user confirmation

### Requirement: Existing-sample verification
The system SHALL verify the Polars migration with existing small samples, existing tests, and focused smoke commands rather than a new benchmark framework.

#### Scenario: Existing tests remain usable
- **WHEN** the migration is implemented
- **THEN** existing `data_preprocess/tests` tests pass or are updated only to reflect the agreed Polars-compatible output contract
- **AND** test updates use the existing small repository samples and synthetic rows already present in tests

#### Scenario: Representative preprocessing smoke paths run
- **WHEN** maintainers run the documented smoke commands for Binance futures and commodity futures preprocessing
- **THEN** the commands generate expected files in the existing output locations
- **AND** manual before/after timing notes, or the precise reason comparable timing is unavailable, are recorded without requiring a fixed percentage improvement

