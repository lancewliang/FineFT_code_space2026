# Close Issues: refactor-multi-contract-scale-save-robust-scaler

## Open Issues

No open CRITICAL, WARNING, or SUGGESTION issues were found during the latest
close-stage verification.

## Resolved During Close Verification

### Partial scale-save artifacts remain after later split failure

- **Evidence:** A fresh close-stage fixture run created one valid train input and one invalid valid input missing the selected state feature. The CLI exited with `returncode=1`, but `SCALE_SAVE/fu/5min/scaler_manifest.json` and `SCALE_SAVE/fu/5min/train/fu2601.feather` already existed while `scale_diagnostics.csv` did not exist.
- **Impact:** Downstream jobs can observe a partially written `SCALE_SAVE/{symbol}/{target_freq}` tree and consume stale/incomplete outputs after a fail-fast error. This weakens the split-stage fail-fast contract and makes the scaler output non-atomic from the pipeline's point of view.
- **Relevant code:** `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py` writes the manifest before processing all split-stage files and writes per-file outputs during the processing loop.
- **Resolution evidence:** Current code validates all discovered split-stage inputs in `preflight_validate_split_inputs(...)` before `write_manifest(...)` or any per-file output write. The full test suite includes `test_multi_contract_scale_save_cli_does_not_leave_partial_outputs_on_later_split_failure`, and the fresh close-stage full test command passed.

## Verification Evidence

- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH="$PWD/FineFT:$PWD/data_preprocess:${PYTHONPATH:-}" pytest -q` -> `314 passed, 17 warnings`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py data_preprocess/tests/test_feature_selection_polars.py` -> exit code 0
- `openspec validate refactor-multi-contract-scale-save-robust-scaler --strict` -> valid
- Plan/task checkbox scan found no remaining unchecked task lines.
