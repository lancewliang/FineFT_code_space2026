# Close Issues: migrate-operator-futures-to-polars

## Important: commodity quote downscale no longer fails on empty intermediate windows

- Status: open
- File: `data_preprocess/operator_futures/commodity/downscale.py`
- Lines: `_resample()` / `downscale_quote_features()`
- Evidence: a Polars input with snapshots at `09:00` and `09:10` and `target_freq="5min"` returns two rows instead of raising `ValueError("Target window has no quote snapshots: ...")` for the empty `09:05` window.
- Impact: the migration no longer preserves the pandas `resample(...).last().index` behavior used by the previous quote feature path, where empty target windows were materialized and checked with `nquote == 0`.
- Required next step: return to `/sddflow build`, add a regression test for an empty intermediate quote window, and update the Polars implementation to materialize/check expected right-closed target windows before archive.
