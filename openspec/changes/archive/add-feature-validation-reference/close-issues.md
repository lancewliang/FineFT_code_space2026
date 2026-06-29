# Close Issues

## Current CRITICAL

- None in the latest close run.

## Current WARNING

- Full test suite passes but emits one existing pandas runtime warning.
  - Evidence: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate finetf && python -m pytest data_preprocess/tests -q` returned exit code 0 with `107 passed, 1 warning`.
  - Warning: `data_preprocess/tests/test_time_operator_polars.py::test_single_price_window_preserves_pandas_reference_nan_values` emits `RuntimeWarning: invalid value encountered in log` from pandas.
  - Impact: no test failure and not specific to `add-feature-validation-reference`; recorded for close visibility.

## Resolved During Build Repair

- `data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_main_script_uses_date_range_full_process_entrypoint` no longer fails in the latest full suite run.
  - Evidence: latest `python -m pytest data_preprocess/tests -q` run returned `107 passed, 1 warning`.
  - Current code status: `data_preprocess/script_preprocess/future_upgraded/commodity/main.sh` defaults to the expected five-day sample range.
- `data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths` no longer fails in the latest full suite run.
  - Evidence: latest `python -m pytest data_preprocess/tests -q` run returned `107 passed, 1 warning`.
  - Current code status: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh` now logs and runs `stitch_main_contract`.

## Resolved / Stale From Prior Close Attempt

- `validate_features.sh --report_dir <path>` is ignored.
  - Evidence: running `bash data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh --root_path /home/lanceliang/opt/aiwork/FineFT_code_space2026 --symbol fu --target_freq 5min --start_date 2025-11-03 --end_date 2025-11-08 --report_dir /tmp/feature_validation_report_close` wrote reports under `/home/lanceliang/opt/aiwork/FineFT_code_space2026/log_futures/feature_validation`.
  - Cause: `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh` parses `--report_dir` into `REPORT_DIR`, then unconditionally resets `REPORT_DIR="${ROOTPATH}/log_futures/feature_validation"`.
  - Impact: the independent validation entrypoint does not honor the requested report output directory, so the CLI/report behavior is not fully compliant.
  - Current status: no longer reproduces. Running `bash data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh --root_path /home/lanceliang/opt/aiwork/FineFT_code_space2026 --symbol fu --target_freq 5min --start_date 2025-11-03 --end_date 2025-11-08 --report_dir /tmp/feature_validation_report_close_check` wrote reports under `/tmp/feature_validation_report_close_check`.

## Verification Completed Before Stop

- `rg -n -- "^- \[ \]" openspec/changes/add-feature-validation-reference/tasks.md openspec/changes/add-feature-validation-reference/plan-ready.md docs/superpowers/plans/2026-06-23-add-feature-validation-reference.md || true`: no unfinished task checkboxes.
- `source ~/miniconda3/etc/profile.d/conda.sh && conda activate finetf && python -m pytest data_preprocess/tests -q`: passed with `107 passed, 1 warning`.
- `bash -n data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`: passed.
- `openspec validate add-feature-validation-reference --strict`: passed.
- `git diff --check`: passed.
