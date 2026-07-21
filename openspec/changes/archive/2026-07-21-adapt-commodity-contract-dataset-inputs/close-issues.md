# Close Issues

## 2026-07-21 Verification

### BLOCKING: Full FineFT test suite cannot be collected

- Command:
  `conda activate finetf && pytest FineFT/tests -q`
- Result: pytest exited with code 2 during collection.
- Failure:
  `FineFT/tests/env/test_env.py` reads
  `/data2/mlqin/HFT4Ind2/dataset/BNBUSDT/train/df_0.feather` at module import time,
  but that external file is not present in this environment.
- Impact: the close workflow requires the complete test suite to pass with zero
  failures, so OpenSpec validation and archive steps were not started.
- Relevant targeted verification:
  `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`
  passed with 14 tests.
- Required next step: use `/sddflow build` to make the full test suite
  self-contained or define and document the authoritative project test command,
  then rerun `/sddflow close adapt-commodity-contract-dataset-inputs`.

## 2026-07-21 Resolution

- Fresh verification completed successfully after the remedial build.
- `conda activate finetf && pytest FineFT/tests -q` -> `76 passed`
- `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q` -> `14 passed`
- `openspec validate adapt-commodity-contract-dataset-inputs --strict` -> valid
- The previous collection blocker in `FineFT/tests/env/test_env.py` is resolved.
