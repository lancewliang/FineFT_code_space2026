# 03: Target-Horizon CatBoost Regressor with Purged Embargo Gap

**What to build:** Create `operator_futures.feature_selection.muti_contract.nonlinear_scoring` to train `CatBoostRegressor` strictly on the surviving subset (~150-200 features) on the designated target decision horizon (default: 6 bars). Enforce a Purged Embargo Gap of $w$ bars between the 80% train pool and 20% early stopping eval pool to eliminate forward label overlap leakage, and calculate composite priority scores.

**Blocked by:** 02-funnel-resequencing-and-predictive-audit.md

**Status:** completed

- [x] Implement `execute_nonlinear_scoring` in `nonlinear_scoring.py` fitting CatBoost only on `--target_decision_window`.
- [x] Insert `purge_window = window_length` buffer between train pool and eval pool in `metrics.py` early stopping split.
- [x] Compute composite priority ranking: $0.40 \cdot 	ext{Rank}(1/\overline{	ext{PSI}}) + 0.35 \cdot 	ext{Rank}(|\overline{	ext{RankIC}}|) + 0.25 \cdot 	ext{Rank}(	ext{CatBoost\_Imp})$.
- [x] In `{contract}_metrics.csv` and `aggregate_metrics.csv`, populate non-decision window CatBoost importance with 0.0 while preserving multi-window IC/RankIC decay curves.
- [x] Unit tests in `test_nonlinear_scoring.py` verify purged embargo slicing, single-horizon fitting, and composite score ranking.
