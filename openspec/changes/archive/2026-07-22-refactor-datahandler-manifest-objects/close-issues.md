# Close Issues: refactor-datahandler-manifest-objects

## Status

Resolved. Repository-level pytest now completes cleanly after removing fragile `FineFT.datahandler` and bare `manifests` imports from the datahandler refactor path.

Final close verification passed on 2026-07-22. Optional final code review was not run before archive.

## Verification Evidence

- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH="$PWD:$PWD/FineFT" pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_slice_model.py`
  - Result: passed, `20 passed in 1.74s`.
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH="$PWD:$PWD/FineFT" pytest`
  - Result: failed during collection with 4 errors.
  - Failures:
    - `FineFT/RL/base/ncqrdqn_test.py`: `ImportError: cannot import name 'NCQRDQN' from 'model.low_level'`
    - `FineFT/tests/datahandler/test_commodity_contract_dataset.py`: `ModuleNotFoundError: No module named 'FineFT.datahandler'; 'FineFT' is not a package`
    - `FineFT/tests/datahandler/test_slice_model.py`: `ModuleNotFoundError: No module named 'FineFT.datahandler'; 'FineFT' is not a package`
    - `FineFT/tests/datahandler/test_vae_data_creation.py`: `ModuleNotFoundError: No module named 'FineFT.datahandler'; 'FineFT' is not a package`
- Retry after import-path fix:
  `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH="$PWD:$PWD/FineFT" pytest`
  - Result: passed, `291 passed, 17 warnings in 31.99s`.
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH="$PWD:$PWD/FineFT" pytest FineFT/tests/datahandler -q`
  - Result: passed, `23 passed in 1.81s`.
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile FineFT/datahandler/manifests.py FineFT/datahandler/commodity_contract_dataset.py FineFT/datahandler/slice_model.py`
  - Result: passed with exit code 0.
- Final close retry:
  `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH="$PWD:$PWD/FineFT" pytest`
  - Result: passed, `291 passed, 17 warnings in 31.11s`.
- Final close retry:
  `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile FineFT/datahandler/manifests.py FineFT/datahandler/commodity_contract_dataset.py FineFT/datahandler/slice_model.py`
  - Result: passed with exit code 0.
- Final close retry:
  `openspec validate refactor-datahandler-manifest-objects --strict`
  - Result: `Change 'refactor-datahandler-manifest-objects' is valid`
- Final close retry:
  `git diff --check`
  - Result: passed with no output.
- `openspec validate refactor-datahandler-manifest-objects --strict`
  - Result: `Change 'refactor-datahandler-manifest-objects' is valid`
- `git diff --check`
  - Result: passed with no output.

## Notes

- Root cause: full pytest collection can put unrelated directories containing `FineFT.py` and `manifests.py` on `sys.path`, so global `FineFT.datahandler` and bare `manifests` imports are fragile.
- Fix: datahandler production modules now avoid the global `FineFT` package name and use `datahandler.manifests` as the script-import fallback; datahandler tests import through `datahandler.*`.
