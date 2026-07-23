# Close Issues: add-commodity-quote-microstructure-features

## Resolved

- The earlier repo-level pytest collection blocker was due to an incomplete `PYTHONPATH`.
- Fresh verification with both roots on the path now passes:
  - `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT:data_preprocess pytest -q`
- Relevant focused checks for this change also pass:
  - `PYTHONPATH=data_preprocess pytest data_preprocess/tests/test_commodity_downscale.py -k "quote_microstructure or quote_ofi" -q`
  - `python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py`
  - `openspec validate add-commodity-quote-microstructure-features --strict`

## Outcome

Close is no longer blocked on repository test collection.
