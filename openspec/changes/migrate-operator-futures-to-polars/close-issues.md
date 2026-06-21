# Close Issues: migrate-operator-futures-to-polars

## Important: commodity quote downscale no longer fails on empty intermediate windows

- Status: resolved
- File: `data_preprocess/operator_futures/commodity/downscale.py`
- Lines: `_resample()` / `downscale_quote_features()`
- Evidence: a Polars input with snapshots at `09:00` and `09:10` and `target_freq="5min"` returns two rows instead of raising `ValueError("Target window has no quote snapshots: ...")` for the empty `09:05` window.
- Impact: the migration no longer preserves the pandas `resample(...).last().index` behavior used by the previous quote feature path, where empty target windows were materialized and checked with `nquote == 0`.
- Resolution: added `test_intermediate_empty_quote_window_fails_fast` and restored fail-fast detection for gaps between quote target windows before archive.
