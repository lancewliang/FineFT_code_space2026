# close issues: add-full-df-warmup-pretrain

## Close status

Close is blocked before archive because required broad test-suite verification does not pass in the current workspace.

Fresh close attempt: 2026-07-02.

## Verified passing checks

- `conda activate finetf && pytest FineFT/tests/rl/test_pretrain_qtable_diagnostics.py FineFT/tests/rl/test_weight_advantage_pretrain_logging.py -q`
  - Result: `18 passed in 2.48s`
- `conda activate finetf && PYTHONPATH=FineFT python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py`
  - Result: passed with no output
- `openspec validate add-full-df-warmup-pretrain --strict`
  - Result: `Change 'add-full-df-warmup-pretrain' is valid`
- `git diff --check`
  - Result: passed with no output
- Checkbox consistency:
  - `openspec/changes/add-full-df-warmup-pretrain/tasks.md`: all tracked checkboxes are `[x]`
  - `openspec/changes/add-full-df-warmup-pretrain/plan-ready.md`: all tracked checkboxes are `[x]`
  - `docs/superpowers/plans/2026-07-01-add-full-df-warmup-pretrain.md`: all implementation-step checkboxes are `[x]`

## Blocking verification failures

### Full test collection with `PYTHONPATH=FineFT`

Command:

```bash
conda activate finetf
PYTHONPATH=FineFT pytest FineFT/tests data_preprocess/tests -q
```

Result: failed during collection with 1 error:

- `FineFT/tests/env/test_env.py` reads `/data2/mlqin/HFT4Ind2/dataset/BNBUSDT/train/df_0.feather` at import time.
- The file is not present in this environment, causing `FileNotFoundError`.

### Broad available subset excluding external-data `test_env.py`

Command:

```bash
conda activate finetf
PYTHONPATH=FineFT pytest FineFT/tests/rl FineFT/tests/datahandler FineFT/tests/env/test_commodity_env.py data_preprocess/tests -q
```

Result: `125 passed`, `4 failed`, `4 warnings`.

Failures:

- `data_preprocess/tests/test_commodity_config_schema.py::test_commodity_config_rejects_non_positive_contract_unit`
  - Error: `CommodityConfig.__init__() missing 1 required positional argument: 'maintenance_margin_rate'`
- `data_preprocess/tests/test_commodity_config_schema.py::test_commodity_config_rejects_empty_trading_sessions`
  - Error: `CommodityConfig.__init__() missing 1 required positional argument: 'maintenance_margin_rate'`
- `data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths`
  - Error: commodity shell script returned non-zero exit status 1
- `data_preprocess/tests/test_commodity_main_contract_cli.py::test_commodity_main_script_uses_date_range_full_process_entrypoint`
  - Error: expected `START_DATE=${START_DATE:-2025-11-03}`, actual script contains `START_DATE=${START_DATE:-2023-01-01}`

These failures are outside the full-df warmup pretrain implementation surface, which modified RL pretrain code/tests and this OpenSpec change's artifacts. They still block SDDFlow close because the close-stage broad verification gate did not pass.

## Recommendation

Do not archive `add-full-df-warmup-pretrain` until the broader suite policy is clarified, the missing external test data is supplied or excluded, and the unrelated commodity preprocessing failures are fixed or explicitly waived.
