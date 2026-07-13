## 1. Main-contract summary generation

- [x] 1.0 Main-contract summary generation complete（与 plan-ready.md Task 1 和 superpowers plan Task 1 同步） <!-- 已实现: summary JSON generation, stitch CLI handoff, and focused tests completed -->
- [x] 1.1 Add focused tests in `data_preprocess/tests/test_commodity_main_contract.py` for natural-month top-2 selection, daily `Volume.max - Volume.min` aggregation, deterministic tie ordering, selected-contract set semantics, actual `start_trading_day` / `end_trading_day`, `trading_day_count`, duplicate `TradingDay + contract` fail-fast, and missing required columns.
- [x] 1.2 Update `data_preprocess/operator_futures/commodity/main_contract.py` to build the summary model from raw files without copying continuous raw day CSV files.
- [x] 1.3 Update `data_preprocess/operator_futures/commodity/stitch_main_contract.py` and CLI tests so the command writes `main_contract_summary.json` under `CONTINUOUS_RAW/{symbol}` and no longer writes `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv`.

## 2. Summary-driven downscale

- [x] 2.0 Summary-driven downscale complete（与 plan-ready.md Task 2 和 superpowers plan Task 2 同步） <!-- 已实现: summary-driven contract downscale CLI, validation, filtering, and focused tests completed -->
- [x] 2.1 Add CLI and behavior tests in `data_preprocess/tests/test_commodity_main_contract_cli.py` and `data_preprocess/tests/test_commodity_downscale.py` for `--summary`, all-contract processing, optional `--contract` filtering, summary validation errors, missing `source_file` fail-fast, and contract-scoped downscale output paths.
- [x] 2.2 Update `data_preprocess/operator_futures/commodity/downscale_continuous_by_trading_day.py` to load and validate `main_contract_summary.json`, read listed raw source files, optionally filter by contract, and write downscale outputs under `{FEATURE_FOLDER}/{symbol}/{contract}/{target_freq}/{date}.feather`.
- [x] 2.3 Remove the old `--input_dir --start_date --end_date` downscale handoff from active CLI behavior and tests, while preserving existing downscale feature semantics.

## 3. Contract-scoped downstream Python paths

- [x] 3.0 Contract-scoped downstream Python paths complete（与 plan-ready.md Task 3 和 superpowers plan Task 3 同步） <!-- 已实现: optional --contract path helper and downstream path wiring verified -->
- [x] 3.1 Add path contract tests for `cross_section/create_feature.py`, `merge_concat/merge.py`, `merge_concat/concat.py`, `time_operator/create_feature_multi_processing.py`, `merge_all/merge_clean.py`, `feature_selection/ic_correlation.py`, and `scale_describe_save/scale_save.py`, covering both `--contract fu2601` and no-contract legacy paths.
- [x] 3.2 Update those Python entrypoints to accept optional `--contract` and resolve paths as `{symbol}/{contract}/{target_freq}` only when contract is provided.
- [x] 3.3 Verify contract-scoped outputs preserve existing daily versus date-range file granularity and commodity feature semantics.

## 4. Commodity shell scripts, validation, and docs

- [x] 4.0 Commodity shell scripts, validation, and docs complete（与 plan-ready.md Task 4 和 superpowers plan Task 4 同步） <!-- 已实现: summary contract loop, contract logs/checks, validation, and docs updated -->
- [x] 4.1 Update `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` to generate summary, parse contracts from summary, run downscale and downstream stages by contract, pass `--contract`, and include contract in logs, skip checks, and output existence checks.
- [x] 4.2 Update or explicitly verify `main_fu.sh`, `main_al.sh`, `commodity_process.sh`, `validate_features.sh`, and `flatten_aluminum_raw_csv.sh` for summary-driven multi-contract behavior.
- [x] 4.3 Update commodity preprocessing documentation to describe `main_contract_summary.json`, summary-driven downscale, contract-scoped output paths, and multi-contract shell usage.

## 5. Summary model bean refactor

- [x] 5.0 Summary model bean refactor complete（与 plan-ready.md Task 5 和 superpowers plan Task 5 同步） <!-- 已实现: typed summary beans with unchanged dict/json serialization -->
- [x] 5.1 Add focused tests for summary model serialization so `MainContractSummary` with nested contract and trading-day models converts to the unchanged JSON/dict contract.
- [x] 5.2 Refactor `data_preprocess/operator_futures/commodity/main_contract.py` so monthly selection builds typed summary model objects first, then converts the model to dict/JSON at the writer boundary.
- [x] 5.3 Keep existing summary JSON schema, existing CLI output, and downstream summary consumers unchanged.

## 6. Verification

- [x] 6.0 Verification complete（与 plan-ready.md Task 6 和 superpowers plan Task 6 同步） <!-- 已实现: pytest, shell syntax, OpenSpec, and diff checks passed -->
- [x] 6.1 Run focused tests with `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py data_preprocess/tests/test_commodity_feature_pipeline.py -q`.
- [x] 6.2 Run shell syntax checks for all commodity shell scripts with `bash -n`.
- [x] 6.3 Run `openspec validate split-commodity-main-contracts-by-contract --strict` and `git diff --check`.

## 7. Summary model bean deserialization

- [x] 7.0 Summary model bean deserialization complete（与 plan-ready.md Task 7 和 superpowers plan Task 7 同步） <!-- 已实现: summary JSON readers now load MainContractSummary models -->
- [x] 7.1 Add focused tests for `MainContractSummary.from_dict` / `load_main_contract_summary` and for downscale consuming `MainContractSummary` objects instead of nested dicts.
- [x] 7.2 Refactor `data_preprocess/operator_futures/commodity/main_contract.py` to own summary deserialization and validation through model methods.
- [x] 7.3 Refactor `downscale_continuous_by_trading_day.py`, `fu_full_process.sh`, and `validate_features.sh` summary readers to use model-based loading while preserving CLI behavior and JSON schema.

## 8. Post-deserialization verification

- [x] 8.0 Post-deserialization verification complete（与 plan-ready.md Task 8 和 superpowers plan Task 8 同步） <!-- 已实现: post-deserialization pytest, shell syntax, OpenSpec, and diff checks passed -->
- [x] 8.1 Run focused tests with `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`.
- [x] 8.2 Run shell syntax checks for all commodity shell scripts with `bash -n`.
- [x] 8.3 Run `openspec validate split-commodity-main-contracts-by-contract --strict` and `git diff --check`.

## 9. Daily volume in summary trading days

- [x] 9.0 Daily volume in summary trading days complete（与 plan-ready.md Task 9 和 superpowers plan Task 9 同步） <!-- 已实现: summary trading-day daily_volume model, builder, and fixtures completed -->
- [x] 9.1 Add focused tests for `MainContractSummaryTradingDay` serialization/deserialization and summary generation so each `trading_days[]` entry includes `daily_volume`.
- [x] 9.2 Update `data_preprocess/operator_futures/commodity/main_contract.py` so typed summary trading-day models require `daily_volume`, and summary construction stores the existing daily `Volume.max - Volume.min` value.
- [x] 9.3 Update summary fixtures in CLI/downscale tests so model-based loading validates the new required field.

## 10. Daily-volume verification

- [x] 10.0 Daily-volume verification complete（与 plan-ready.md Task 10 和 superpowers plan Task 10 同步） <!-- 已实现: daily-volume pytest, shell syntax, OpenSpec, and diff checks passed -->
- [x] 10.1 Run focused tests with `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`.
- [x] 10.2 Run shell syntax checks for all commodity shell scripts with `bash -n`.
- [x] 10.3 Run `openspec validate split-commodity-main-contracts-by-contract --strict` and `git diff --check`.

## 11. Contract trading-window clipping

- [x] 11.0 Contract trading-window clipping complete（与 plan-ready.md Task 11 和 superpowers plan Task 11 同步） <!-- 已实现: selected contract summary trading days are clipped by first selected month and date-range-relative final-10-day cutoff -->
- [x] 11.1 Add focused tests proving selected contracts retain only actual trading days from the first selected month start through the inclusive cutoff 10 contract trading days before the last trading day in the requested date range.
- [x] 11.2 Update `data_preprocess/operator_futures/commodity/main_contract.py` so selected-contract summary entries are clipped after monthly top-2 selection but before `MainContractSummaryContract` construction.
- [x] 11.3 Add fail-fast coverage for selected contracts whose clipped trading window is empty.

## 12. Contract trading-window verification

- [x] 12.0 Contract trading-window verification complete（与 plan-ready.md Task 12 和 superpowers plan Task 12 同步） <!-- 已实现: contract trading-window pytest, shell syntax, OpenSpec, and diff checks passed -->
- [x] 12.1 Run focused tests with `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`.
- [x] 12.2 Run shell syntax checks for all commodity shell scripts with `bash -n`.
- [x] 12.3 Run `openspec validate split-commodity-main-contracts-by-contract --strict` and `git diff --check`.

## 13. High-volume-day main contract rule

- [x] 13.0 High-volume-day main contract rule complete（与 plan-ready.md Task 13 和 superpowers plan Task 13 同步） <!-- 已实现: fu high-volume threshold config, 10-day union rule, selection_rule update, and focused tests completed -->
- [x] 13.1 Add focused tests proving a contract outside monthly top 2 is selected when at least 10 actual trading days in a month have `daily_volume > threshold`.
- [x] 13.2 Add commodity config support for `main_contract_daily_volume_threshold`, with `fu` configured as `15000`.
- [x] 13.3 Update `data_preprocess/operator_futures/commodity/main_contract.py` to union monthly top-2 selection with the high-volume-day threshold rule and update summary `selection_rule` semantics.

## 14. High-volume-day verification

- [x] 14.0 High-volume-day verification complete（与 plan-ready.md Task 14 和 superpowers plan Task 14 同步） <!-- 已实现: focused pytest, shell syntax, OpenSpec strict, and diff checks passed -->
- [x] 14.1 Run focused tests with `conda run -n finetf pytest data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_main_contract_cli.py data_preprocess/tests/test_commodity_downscale.py -q`.
- [x] 14.2 Run shell syntax checks for all commodity shell scripts with `bash -n`.
- [x] 14.3 Run `openspec validate split-commodity-main-contracts-by-contract --strict` and `git diff --check`.

## 15. Cross-contract training feature union

- [x] 15.0 Cross-contract training feature union complete（与 plan-ready.md Task 15 和 superpowers plan Task 15 同步） <!-- 已实现: feature-union CLI, stable state-feature union, shell orchestration, validation checks, docs, and focused tests completed -->
- [x] 15.1 Add focused tests proving state features from all summary contracts are unioned in stable first-seen order and duplicates are removed.
- [x] 15.2 Add a commodity feature-union CLI that reads `main_contract_summary.json`, loads each contract's final `SCALE_SAVE/.../state_features.npy`, and writes `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy` plus `feature_union_manifest.json`.
- [x] 15.3 Add fail-fast coverage for summary contracts whose final `state_features.npy` is missing.
- [x] 15.4 Update `fu_full_process.sh` and `validate_features.sh` so the full process generates and validates the symbol-level feature union after all contract `scale_save` steps.

## 16. Feature-union verification

- [x] 16.0 Feature-union verification complete（与 plan-ready.md Task 16 和 superpowers plan Task 16 同步） <!-- 已实现: feature-union pytest, shell syntax, OpenSpec strict, and diff checks passed -->
- [x] 16.1 Run focused tests with `conda run -n finetf pytest data_preprocess/tests/test_commodity_feature_pipeline.py data_preprocess/tests/test_commodity_main_contract_cli.py -q`.
- [x] 16.2 Run shell syntax checks for all commodity shell scripts with `bash -n`.
- [x] 16.3 Run `openspec validate split-commodity-main-contracts-by-contract --strict` and `git diff --check`.
