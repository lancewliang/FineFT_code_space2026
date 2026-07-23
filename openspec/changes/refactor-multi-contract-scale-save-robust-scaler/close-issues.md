# Close Issues: refactor-multi-contract-scale-save-robust-scaler

## CRITICAL

### Partial scale-save artifacts remain after later split failure

- **Evidence:** A fresh close-stage fixture run created one valid train input and one invalid valid input missing the selected state feature. The CLI exited with `returncode=1`, but `SCALE_SAVE/fu/5min/scaler_manifest.json` and `SCALE_SAVE/fu/5min/train/fu2601.feather` already existed while `scale_diagnostics.csv` did not exist.
- **Impact:** Downstream jobs can observe a partially written `SCALE_SAVE/{symbol}/{target_freq}` tree and consume stale/incomplete outputs after a fail-fast error. This weakens the split-stage fail-fast contract and makes the scaler output non-atomic from the pipeline's point of view.
- **Relevant code:** `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py` writes the manifest before processing all split-stage files and writes per-file outputs during the processing loop.
- **Recommended fix:** Validate all discovered split-stage inputs for required selected features before writing any manifest or output files, or write to a temporary output root and atomically move it into `SCALE_SAVE/{symbol}/{target_freq}` only after all files and diagnostics are ready.

## Verification Evidence

- `conda run -n finetf pytest data_preprocess/tests -q` -> `196 passed, 3 warnings`
- `openspec validate refactor-multi-contract-scale-save-robust-scaler --strict` -> valid
- Plan/task checkbox scan found no remaining unchecked task lines.
