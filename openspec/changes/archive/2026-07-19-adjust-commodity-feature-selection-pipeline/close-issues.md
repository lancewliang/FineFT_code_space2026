# Close Issues: adjust-commodity-feature-selection-pipeline

## 2026-07-19 Close Attempt

### Verification Evidence

- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest -q`
  - Result: failed during collection with 23 root-suite errors before the focused commodity preprocessing tests ran.
  - Failure classes: missing import paths for `RL`, `model`, `env`, `datahandler`, `FineFT.datahandler`, and missing external file `/data2/mlqin/HFT4Ind2/dataset/BNBUSDT/train/df_0.feather`.
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests -q`
  - Result: `191 passed, 3 warnings in 19.32s`.
  - Warnings: existing `fillna(method=...)` deprecation warning and NumPy divide warnings in feature validation tests.
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py -q`
  - Result: `19 passed in 4.10s` during build verification.
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`
  - Result: passed with no output during build verification.
- `openspec validate adjust-commodity-feature-selection-pipeline --strict`
  - Result: `Change 'adjust-commodity-feature-selection-pipeline' is valid`.

### Completeness

- `openspec/changes/adjust-commodity-feature-selection-pipeline/tasks.md` has no unchecked live task items.
- `openspec/changes/adjust-commodity-feature-selection-pipeline/plan-ready.md` has no unchecked task-completion items.
- `docs/superpowers/plans/2026-07-18-adjust-commodity-feature-selection-pipeline.md` has no unchecked live implementation-step items.
- Implementation evidence for the latest CSV debug-output amend:
  - `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`
  - `data_preprocess/tests/test_feature_selection_polars.py`

### Close Test Strategy

The default root command `pytest -q` currently fails during test collection because unrelated test modules require import paths and local data that are not available in this environment:

- `RL`
- `model`
- `env`
- `datahandler`
- `FineFT.datahandler`
- `/data2/mlqin/HFT4Ind2/dataset/BNBUSDT/train/df_0.feather`

Per user confirmation on 2026-07-19, close for this project uses the supported focused suite:

- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests -q`

The root historical/external-data suite is not required for this change's close gate.

### Current Close Status

Implementation verification and OpenSpec validation passed under the confirmed close strategy. Final code review was skipped by user confirmation on 2026-07-19. Close may proceed to archive confirmation.
