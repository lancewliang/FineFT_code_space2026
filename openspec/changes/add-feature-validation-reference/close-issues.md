# Close Issues

## Current CRITICAL

- Full test suite fails with two commodity main-contract CLI regressions.
  - Evidence: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate finetf && python -m pytest data_preprocess/tests -q` returned exit code 1 with `2 failed, 105 passed, 1 warning`.
  - Failure 1: `data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths` expected a `stitch_main_contract` step start line in the total commodity log, but the log begins at `downscale_continuous_by_trading_day`.
  - Failure 2: `data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_main_script_uses_date_range_full_process_entrypoint` expected `START_DATE=${START_DATE:-2025-11-03}`, but `data_preprocess/script_preprocess/future_upgraded/commodity/main.sh` currently defaults to `START_DATE=${START_DATE:-2023-01-01}`.
  - Impact: close Phase 1 implementation verification does not pass, so this change cannot be archived yet.

## Resolved / Stale From Prior Close Attempt

- `validate_features.sh --report_dir <path>` is ignored.
  - Evidence: running `bash data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh --root_path /home/lanceliang/opt/aiwork/FineFT_code_space2026 --symbol fu --target_freq 5min --start_date 2025-11-03 --end_date 2025-11-08 --report_dir /tmp/feature_validation_report_close` wrote reports under `/home/lanceliang/opt/aiwork/FineFT_code_space2026/log_futures/feature_validation`.
  - Cause: `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh` parses `--report_dir` into `REPORT_DIR`, then unconditionally resets `REPORT_DIR="${ROOTPATH}/log_futures/feature_validation"`.
  - Impact: the independent validation entrypoint does not honor the requested report output directory, so the CLI/report behavior is not fully compliant.
  - Current status: no longer reproduces. Running `bash data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh --root_path /home/lanceliang/opt/aiwork/FineFT_code_space2026 --symbol fu --target_freq 5min --start_date 2025-11-03 --end_date 2025-11-08 --report_dir /tmp/feature_validation_report_close_check` wrote reports under `/tmp/feature_validation_report_close_check`.

## Verification Completed Before Stop

- `rg -n -- "^- \[ \]" openspec/changes/add-feature-validation-reference/tasks.md openspec/changes/add-feature-validation-reference/plan-ready.md docs/superpowers/plans/2026-06-23-add-feature-validation-reference.md || true`: no unfinished task checkboxes.
- `source ~/miniconda3/etc/profile.d/conda.sh && conda activate finetf && python -m pytest data_preprocess/tests -q`: failed with 2 failures.
- `bash -n data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`: passed.
- `openspec validate add-feature-validation-reference --strict`: passed.
- `git diff --check`: passed.
