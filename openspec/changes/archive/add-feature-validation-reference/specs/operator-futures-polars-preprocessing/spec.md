## ADDED Requirements

### Requirement: Feature validation with pandas reference
The system SHALL provide an independent operator-futures feature validation workflow that compares generated preprocessing artifacts against validation-only pandas reference calculations.

#### Scenario: Independent commodity validation shell entrypoint
- **WHEN** a maintainer runs `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh` with `--root_path`, `--symbol`, `--target_freq`, `--start_date`, and `--end_date`
- **THEN** the script SHALL set up the Python module path and invoke the feature validation CLI
- **AND** the script SHALL NOT invoke or modify `data_preprocess/script_preprocess/future_upgraded/commodity/main.sh`
- **AND** the script SHALL NOT change existing preprocess output paths

#### Scenario: Validation-only pandas reference implementation
- **WHEN** the validator recomputes expected feature values
- **THEN** it SHALL use copied pandas reference modules under `data_preprocess/operator_futures/feature_validation/pandas_reference/`
- **AND** the reference modules SHALL be sourced from `/home/lanceliang/opt/aiwork/FineFT_code_space2026_2/data_preprocess/operator_futures`
- **AND** production preprocessing modules SHALL NOT import the validation-only pandas reference namespace as a runtime fallback

#### Scenario: Fixed expected columns from docs
- **WHEN** the validator checks artifact schemas
- **THEN** it SHALL use fixed expected-column definitions derived from `docs/data/*.md`
- **AND** it SHALL NOT parse Markdown documentation at runtime to discover expected columns
- **AND** it SHALL report missing columns, extra columns, and unverified columns separately

#### Scenario: Intermediate artifact validation stages
- **WHEN** the validator runs for a symbol, target frequency, and date range
- **THEN** it SHALL validate intermediate and final artifacts for `cross_section`, `merge_concat`, `time_feature`, `merge_clean`, `ic_correlation`, and `scale_save`
- **AND** it SHALL read existing intermediate artifacts rather than recomputing from raw CSV in the first version
- **AND** each stage SHALL define its input paths, actual output paths, reference calculation, expected columns, sampling behavior, and comparison tolerance

#### Scenario: Sampled row value comparison
- **WHEN** actual and reference outputs both contain a comparable column and timestamp
- **THEN** the validator SHALL align rows by `timestamp`
- **AND** it SHALL compare sampled rows column-by-column
- **AND** numeric values SHALL pass when `abs(actual - expected) <= 1e-9`
- **AND** values outside that tolerance SHALL be reported with stage, column, timestamp, actual value, expected value, and maximum absolute difference

#### Scenario: Reconciliation reports and exit codes
- **WHEN** validation completes
- **THEN** the validator SHALL write Markdown and JSON reports
- **AND** the reports SHALL include each stage status as `pass`, `fail`, `partial`, or `error`
- **AND** the reports SHALL include checked column counts, missing columns, extra columns, unverified columns, mismatched columns, maximum absolute differences, and sample failure rows
- **AND** the CLI SHALL exit `0` only when every stage passes
- **AND** the CLI SHALL exit `1` when any stage has validation failures or reference/runtime errors
- **AND** the CLI SHALL exit `2` for invalid arguments or when no comparable input artifacts are available
