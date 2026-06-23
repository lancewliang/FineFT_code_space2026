# Close Issues

## CRITICAL

- `validate_features.sh --report_dir <path>` is ignored.
  - Evidence: running `bash data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh --root_path /home/lanceliang/opt/aiwork/FineFT_code_space2026 --symbol fu --target_freq 5min --start_date 2025-11-03 --end_date 2025-11-08 --report_dir /tmp/feature_validation_report_close` wrote reports under `/home/lanceliang/opt/aiwork/FineFT_code_space2026/log_futures/feature_validation`.
  - Cause: `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh` parses `--report_dir` into `REPORT_DIR`, then unconditionally resets `REPORT_DIR="${ROOTPATH}/log_futures/feature_validation"`.
  - Impact: the independent validation entrypoint does not honor the requested report output directory, so the CLI/report behavior is not fully compliant.

## Verification Completed Before Stop

- `rg -n -- "^- \[ \]" openspec/changes/add-feature-validation-reference/tasks.md openspec/changes/add-feature-validation-reference/plan-ready.md docs/superpowers/plans/2026-06-23-add-feature-validation-reference.md || true`: no unfinished task checkboxes.
- `conda run -n finetf python -m pytest data_preprocess/tests -q`: 93 passed.
- `bash -n data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`: passed.
- `openspec validate add-feature-validation-reference --strict`: passed.
- `git diff --check`: passed.
