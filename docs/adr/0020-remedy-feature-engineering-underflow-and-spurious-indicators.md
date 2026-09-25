---
status: accepted
---

# Remedy Feature Engineering Underflow and Spurious Indicators

We establish an end-to-end remediation policy for fundamental mathematical and type-safety bugs discovered in the commodity feature engineering pipeline, while decoupling state representation from environment execution/reward features.

Following rigorous loss attribution and code-level auditing, four severe structural failure modes were identified:
1. **Polars `UInt32` Unsigned Integer Underflow**: Downscaled trade count metrics (`ntrade_up_estimated`, `ntrade_down_estimated`, `ntrade_flat_estimated`) were produced as `UInt32`. Subsequent subtractions (`up - down`, `up + down - flat`) in `base_feature_util.py` and `time_operator_util.py` underflowed to $\approx 4.29 \times 10^9$ whenever selling volume or flat volume predominated, generating catastrophic multi-million value spikes and corrupting features like `ntrade_estimated_updownflat_vol_udnorm` and `trade_direction_persistence_20m`.
2. **String Prefix Over-Matching Silent Feature Erasure**: In `cross_month_feature.py:_merge_feature_frames`, `column.startswith("cm_m")` was intended to isolate delivery sequence pairs (`cm_m1_m2`, `cm_m2_m3`) but inadvertently matched `cm_main_sub_*`, dropping legitimate calculations and overwriting them with 0.0 across all dataset splits.
3. **Formula Deficiencies and Sign Inversions**: `roc_*_std_norm` omitted price subtraction (calculating nominal price level over volatility $P_{t-W}/\sigma_P$), while `beta_*` subtracted current price from past price, inverting trend sign ($r = -0.9037$ with true slope). Additionally, discrete tick orderbook spreads were subjected to continuous asset price log return logic multiplied by 1000 (`spread_oe_max_log_return_2`), injecting extreme pulse noise.
4. **Sampling Frequency Mismatch & Degenerate Zero Channels**: Rolling 192-bar intraday windows (~3 days) on 30-day and multi-week static metrics (`prev_*_quantile_rank`) caused >26% identical ties and step-function jumps. Furthermore, price limit indicators (`limit_*`) had zero occurrences in training data, causing VAE Gaussian likelihood collapse upon evaluation.

We chose to:
- Enforce explicit `Float64` casting on all count and volume columns prior to arithmetic in `downscale.py`, `base_feature_util.py`, and `time_operator_util.py`, permanently eliminating integer underflow.
- Restrict delivery column matching in `_merge_feature_frames` to the exact delivery pair prefixes (`cm_m1_m2`, `cm_m2_m3`), restoring true `cm_main_sub_*` feature values.
- Correct `roc_*_std_norm` to $(P_t - P_{t-W}) / \sigma_P$ and `beta_*` to $(P_t - P_{t-W}) / (W \cdot P_t)$ in `multi_processing_util.py`.
- **Decouple State Features from Reward/Execution Columns**: Preserve all `PRICE_LIMIT_RATIO_FEATURE_COLUMNS` (`limit_up_single_sided_ratio`, `limit_down_single_sided_ratio`, `limit_up_ask_depth_ratio_5`, `limit_down_bid_depth_ratio_5`, `limit_depth_imbalance_ratio_5`) in `get_reward_execution_columns` in `operator_futures/commodity/schema.py` so downstream RL environments and trading simulators can strictly enforce price-limit execution rules and reward penalties.
- Simultaneously blacklist all 5 `limit_*` features, all 30 `prev_*_quantile_rank` features, `buy/sell_spread_oe_max_log_return_2`, `roc_*_std_norm`, and `cm_m1_m2_open_interest_share_m2/m3` from `COMMODITY_FU_FEATURE_BLACKLIST` to keep the RL and VAE state space strictly stationary, continuous, and well-conditioned.
- Execute full pipeline regeneration from raw tick downscaling through scale save.
