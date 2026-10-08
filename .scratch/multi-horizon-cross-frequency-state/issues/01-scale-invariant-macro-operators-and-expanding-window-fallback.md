# 01: Scale-Invariant Macro Operators and Expanding-Window Fallback

**What to build:** Continuous time feature extraction operators compute long-horizon macro indicators across rolling windows $W \in \{720, 1440\}$ steps (20 to 40 trading days) for benchmark mark price and cross-month contract spreads. All rolling operators enforce a causal expanding-window fallback with `min_periods = 48`, ensuring that contract data frames contain zero NaNs from the very first step, completely satisfying pipeline zero-NaN validation without discarding initial contract samples.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] Implement scale-invariant directional macro operators in Polars and numpy within `data_preprocess/operator_futures/time_operator/`:
  - `mark_price_ema_deviation_{w}`: $(P_t - \text{EMA}_w(P)_t) / (\text{ATR}_w(P)_t + \epsilon)$
  - `mark_price_roc_{w}`: $(P_t - P_{t-w}) / (P_{t-w} + \epsilon) \times 1000$
  - `trend_beta_{w}`: $(\text{Slope}_w(P) \times w) / (P_{t-w} + \epsilon)$
- [x] Implement scale-invariant dispersion and purity macro operators:
  - `trend_to_noise_{w}`: $|P_t - P_{t-w}| / (\sum_{i=0}^{w-1} |P_{t-i} - P_{t-i-1}| + \epsilon) \in [0, 1]$
  - `cm_main_sub_spread_rolling_zscore_{w}`: $(S_t - \mu_w(S)_t) / (\sigma_w(S)_t + \epsilon)$
- [x] Enforce causal expanding-window fallback with `min_periods = 48` for all $W \ge 48$ operators so that observations prior to step $W$ compute cumulative statistics without emitting NaNs.
- [x] Strictly prohibit generating macro rolling indicators on high-frequency orderbook volume and depth columns to prevent uninformative noise.
- [x] Add unit tests in `data_preprocess/tests/test_time_operator_polars.py` verifying that short series ($N < 720$ bars) and long series ($N = 3000$ bars) generate 100% finite, scale-invariant values with zero NaNs.
