# operator-futures-polars-preprocessing Specification

## ADDED Requirements

### Requirement: Scale save NaN fail-fast validation
The system SHALL stop `scale_save.py` before writing outputs when the main input DataFrame or final output DataFrame contains floating-point NaN values.

#### Scenario: Main input contains NaN
- **WHEN** `scale_save.py` reads `IC_RESULT/<symbol>/<freq>/<start>-<end>/<df_name>.feather`
- **AND** the loaded DataFrame contains one or more floating-point NaN values
- **THEN** the script SHALL fail before scaling state features
- **AND** the failure message SHALL identify the `input` stage, the input feather path, and the columns containing NaN values
- **AND** the script SHALL NOT write `SCALE_SAVE/<symbol>/<freq>/<start>-<end>/df.feather`
- **AND** the script SHALL NOT write `SCALE_SAVE/<symbol>/<freq>/<start>-<end>/df.csv`
- **AND** the script SHALL NOT write `SCALE_SAVE/<symbol>/<freq>/<start>-<end>/state_features.npy`
- **AND** the script SHALL NOT write `SCALE_SAVE/<symbol>/<freq>/<start>-<end>/df_describe.csv`

#### Scenario: Final output contains NaN
- **WHEN** `scale_save.py` has built the final `out` DataFrame
- **AND** `out` contains one or more floating-point NaN values
- **THEN** the script SHALL fail before any output file is written
- **AND** the failure message SHALL identify the `output` stage, the target `df.feather` path, and the columns containing NaN values
- **AND** the script SHALL NOT write `SCALE_SAVE/<symbol>/<freq>/<start>-<end>/df.feather`
- **AND** the script SHALL NOT write `SCALE_SAVE/<symbol>/<freq>/<start>-<end>/df.csv`
- **AND** the script SHALL NOT write `SCALE_SAVE/<symbol>/<freq>/<start>-<end>/state_features.npy`
- **AND** the script SHALL NOT write `SCALE_SAVE/<symbol>/<freq>/<start>-<end>/df_describe.csv`

#### Scenario: Main input and final output contain no NaN
- **WHEN** `scale_save.py` reads a main input DataFrame without floating-point NaN values
- **AND** the final `out` DataFrame contains no floating-point NaN values
- **THEN** the script SHALL preserve the existing successful scale-save output behavior
- **AND** the existing CLI arguments, path layout, feature selection rules, scaling rules, and output file formats SHALL remain compatible
