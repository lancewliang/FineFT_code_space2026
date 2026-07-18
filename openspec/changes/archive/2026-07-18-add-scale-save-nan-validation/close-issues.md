# Close Issues: add-scale-save-nan-validation

## Close Status

Close is blocked before archive because broad verification does not pass in the current workspace.

Fresh close attempt: 2026-07-16.

## Verified Passing Checks

- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py::test_scale_helpers_ignore_nan_like_pandas data_preprocess/tests/test_feature_selection_polars.py::test_scale_helpers_match_reference_for_tiny_std_large_mean_adjustment data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_writes_expected_files data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_rejects_input_nan_before_writing_outputs data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_rejects_output_nan_before_writing_outputs -q`
  - Result: `5 passed in 1.60s`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/scale_describe_save/scale_save.py data_preprocess/tests/test_feature_selection_polars.py`
  - Result: passed with no output
- `openspec validate add-scale-save-nan-validation --strict`
  - Result: `Change 'add-scale-save-nan-validation' is valid`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests -q`
  - Result: `140 passed, 4 warnings in 13.78s`

## Blocking Verification Failures

### Repository-wide pytest without explicit PYTHONPATH

Command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
pytest -q
```

Result: failed during collection with 23 errors.

Observed failure classes:

- `ModuleNotFoundError` for `RL`
- `ModuleNotFoundError` for `model`
- `ModuleNotFoundError` for `env`
- `ModuleNotFoundError` / package resolution error for `FineFT.datahandler`
- `FileNotFoundError` for `/data2/mlqin/HFT4Ind2/dataset/BNBUSDT/train/df_0.feather`

### Broad suite with expected import path

Command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
PYTHONPATH=FineFT:data_preprocess pytest FineFT/tests data_preprocess/tests -q
```

Result: failed during collection with 1 error:

- `FineFT/tests/env/test_env.py` reads `/data2/mlqin/HFT4Ind2/dataset/BNBUSDT/train/df_0.feather` at import time.
- The file is not present in this environment, causing `FileNotFoundError`.

### Broad runnable subset excluding external-data `test_env.py`

Command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
PYTHONPATH=FineFT:data_preprocess pytest FineFT/tests/rl FineFT/tests/datahandler FineFT/tests/env/test_commodity_env.py data_preprocess/tests -q
```

Result: `7 failed, 199 passed, 18 warnings in 16.61s`.

Failures:

- `FineFT/tests/rl/test_pretrain_qtable_diagnostics.py::test_diagnostics_cache_qtables_once_and_export_one_csv_per_df_action`
  - `TypeError: prepare_pretrain_qtable_diagnostics() got an unexpected keyword argument 'num_sample'`
- `FineFT/tests/rl/test_pretrain_qtable_diagnostics.py::test_diagnostics_read_existing_csvs_without_recomputing`
  - `TypeError: prepare_pretrain_qtable_diagnostics() got an unexpected keyword argument 'num_sample'`
- `FineFT/tests/rl/test_pretrain_qtable_diagnostics.py::test_diagnostics_ignore_existing_csvs_when_manifest_does_not_match`
  - `TypeError: prepare_pretrain_qtable_diagnostics() got an unexpected keyword argument 'num_sample'`
- `FineFT/tests/rl/test_pretrain_qtable_diagnostics.py::test_prepare_uses_qtable_cache_builder_diagnostics`
  - `TypeError: prepare_pretrain_qtable_diagnostics() got an unexpected keyword argument 'num_sample'`
- `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_run_full_df_warmup_updates_once_per_df`
  - `KeyError: 12` in `Weighted_Contexts_DQN._run_full_df_warmup`
- `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_run_full_df_warmup_logs_first_row_tech_indicators`
  - `KeyError: 12` in `Weighted_Contexts_DQN._run_full_df_warmup`
- `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_full_df_warmup_logs_rollout_balances_without_df_final_balance`
  - `KeyError: 12` in `Weighted_Contexts_DQN._run_full_df_warmup`

These failures are outside the scale-save implementation surface, which is limited to `data_preprocess/operator_futures/scale_describe_save/scale_save.py`, `data_preprocess/tests/test_feature_selection_polars.py`, and this OpenSpec change's artifacts. They still block SDDFlow close because the close-stage broad verification gate did not pass.

## Spec Consistency

- Completeness: PASS. `tasks.md`, `plan-ready.md`, and `docs/superpowers/plans/2026-07-16-add-scale-save-nan-validation.md` have all implementation checkboxes completed.
- Correctness: PASS for the scale-save requirements. Input-stage NaN failure, output-stage NaN failure, and no-NaN success behavior are covered by passing focused tests.
- Coherence: PASS. No `design.md` was required; implementation follows `proposal.md` and the delta spec while preserving CLI arguments, path layout, feature selection, scaling rules, and output file formats.

## Recommendation

Do not archive `add-scale-save-nan-validation` until the broader suite policy is clarified, the missing external test data is supplied or excluded, and the unrelated FineFT RL failures are fixed or explicitly waived.

## Forced Archive Waiver

Fresh close attempt: 2026-07-18.

User requested `/sddflow close add-scale-save-nan-validation 强制归档`, so archive proceeds despite non-target broad-suite failures.

Fresh passing checks:

- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_feature_selection_polars.py::test_scale_helpers_ignore_nan_like_pandas data_preprocess/tests/test_feature_selection_polars.py::test_scale_helpers_match_reference_for_tiny_std_large_mean_adjustment data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_writes_expected_files data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_rejects_input_nan_before_writing_outputs data_preprocess/tests/test_feature_selection_polars.py::test_scale_save_cli_rejects_output_nan_before_writing_outputs -q`
  - Result: `5 passed in 1.90s`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/scale_describe_save/scale_save.py data_preprocess/tests/test_feature_selection_polars.py`
  - Result: passed with no output
- `openspec validate add-scale-save-nan-validation --strict`
  - Result: `Change 'add-scale-save-nan-validation' is valid`

Fresh non-blocking failures waived by force archive:

- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests -q`
  - Result: `2 failed, 156 passed, 4 warnings in 18.01s`
  - Failures:
    - `data_preprocess/tests/test_commodity_downscale.py::test_second_level_fills_low_price_empty_ask_prices`
    - `data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths`
  - Scope assessment: failures are outside the scale-save NaN validation surface. At the time of forced archive, implementation files `data_preprocess/operator_futures/scale_describe_save/scale_save.py`, `data_preprocess/tests/test_feature_selection_polars.py`, and `docs/superpowers/plans/2026-07-16-add-scale-save-nan-validation.md` had no diff; `close-issues.md` was updated only to record this forced-archive waiver.

Post-archive checks:

- `openspec archive add-scale-save-nan-validation --yes`
  - Result: archived as `openspec/changes/archive/2026-07-18-add-scale-save-nan-validation/`; updated `openspec/specs/operator-futures-polars-preprocessing/spec.md`
  - Non-blocking warning: `proposal.md` uses the older proposal shape and is missing `## Why` / `## What Changes`
- `openspec validate --all --strict`
  - Result: `5 passed, 0 failed`
