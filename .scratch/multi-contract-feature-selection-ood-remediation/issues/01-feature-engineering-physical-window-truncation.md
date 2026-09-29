# 01: Feature Engineering Physical Window Truncation and Lifecycle Indicator Isolation

**What to build:** Restrict the time operator feature generation window list for 10-minute sampling frequency to intraday physical windows ($w \le 48$, representing $\le 8$ trading hours) in `fu_full_process.sh` and pipeline defaults, and ensure deterministic lifecycle indicators (e.g., `contract_life_remaining_ratio`, `cm_*_open_interest_share_*`) are segregated from state feature candidates into execution/reward schemas.

**Blocked by:** None (can start immediately)

**Status:** closed

- [x] `run_commodity_time_feature` in `fu_full_process.sh` configures `--windows "2,6,12,16,24,48"` for 10min frequency, eliminating multi-day macro windows ($w \ge 96$).
- [x] `operator_futures.feature_selection.muti_contract.metrics.DEFAULT_WINDOWS_LIST` and CLI defaults reflect intraday horizons $[1, 2, 6, 12, 24, 48]$.
- [x] Segregate deterministic lifecycle and expiration progress variables from state observation candidates into execution schemas.
- [x] Unit tests in `test_commodity_feature_pipeline.py` and `test_time_operator_polars.py` verify that 10min time feature generation runs without error and outputs valid column sets without multi-day macro windows.
