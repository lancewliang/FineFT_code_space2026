---
status: accepted
---

# Residual OOD Remediation for Uncentered Pivots, Volume Shares, and Orderbook Spreads

We expand the feature selection blacklist and enforce physical lower bounds on orderbook depth spreads, resolving the residual cluster of VAE out-of-distribution (OOD) likelihood degradation revealed by the post-ADR-0028 empirical diagnostic matrix.

## Context

Following the multi-perspective VAE feature OOD matrix remediation under ADR-0027 and ADR-0028, empirical validation on the 10-minute commodity futures dataset (`fu`) confirmed major structural improvements:
- Valid vs Train total $\Delta\text{NLL}$ dropped from `+280.92` to `+163.71` (-41.7%).
- Test vs Train total $\Delta\text{NLL}$ compressed from `+38.17` to `+26.03` (-31.8%).
- Test vs Valid total $\Delta\text{NLL}$ reached `-137.68`, demonstrating that test representations do not suffer generalization collapse compared to the elevated-volatility validation period.

However, granular per-feature and per-contract diagnostics uncovered four remaining clusters of mathematical and engineering pathologies:
1. **Omitted Multi-Session Nominal Price Ratios**: ADR-0028 blacklisted 192-bar uncentered price ratios (`min_192_origin`, `max_192_origin`, `pivot_*_192_origin`), but 96-bar and 48-bar variants slipped through. Specifically, `min_96_origin` ($P_{\min, 96} / P_t$, spanning 16 continuous trading hours) ranked #3 in `valid_vs_train` ($\Delta\text{NLL} = 11.18$, variance ratio 1.75x) due to multi-session level drift across trending regimes.
2. **One-Sided Multi-Day Rolling High Distance**: `max_192_std_norm_origin` ranked #2 in `test_vs_train` ($\Delta\text{NLL} = 1.92$, 7.38% contribution) and entered the top 3 degradation ranks for **9 out of 13 independent test contracts**. Spanning 32 trading hours across weekends, this non-negative $[0, \infty)$ metric lacks a symmetric minimum counterpart, causing persistent elevated distance during secular bear markets or post-trend consolidations.
3. **Omission of Cross-Month Volume Shares**: ADR-0027 blacklisted open interest shares (`open_interest_share`) to eliminate contract rollover lifecycle leakage, but omitted the volume share counterparts. `cm_main_sub_volume_share_sub` exhibited cyclical non-stationarity, ranking #1 in degradation for contract `fu2606` and in the top 3 for 4 test contracts.
4. **Orderbook Discrete Step & Zero-Depth Underflow**: In `base_feature_util.py`, `buy_spread_oe_max` ($|bid1 - bid5|$) has a median and IQR of 4.0 ticks across training data (IQR=0.0). In validation and test splits, occasional missing depth at session boundaries produced raw 0.0 values, scaling to $-2.845$. The VAE, optimized on non-negative training values ($\ge 0.0$), incurred severe Gaussian NLL penalties on these negative outliers. Furthermore, short-window rolling z-scores on near-constant tick spreads (`sell_spread_oe_max_trend_6`) generated spurious extreme pulses.
5. **192-Bar Rolling Spread Z-Scores**: Long-window cross-month spread z-scores (`cm_current_main_spread_rolling_zscore_192`) pegged against clipping boundaries during rollover basis shifts, contributing 292.4% of single-contract OOD degradation in `fu2601`.

## Decision

We establish the following remediation policies:

### 1. Extended Feature Blacklist Expansion
Add the following 16 deterministic non-stationary, cyclical lifecycle, and discrete pulse features to `COMMODITY_FU_FEATURE_BLACKLIST` in `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`:
- **Residual Uncentered Price Ratios & Pivots**: `min_96_origin`, `max_96_origin`, `pivot_s2_48_origin`, `pivot_s1_24_origin`, `pivot_s1_6_origin`, `bollinger_lower_12_origin`
- **Asymmetric Multi-Day Rolling Extremes**: `max_192_std_norm_origin`
- **Long-Window Spread Rolling Z-Scores**: `cm_current_main_spread_rolling_zscore_192`, `cm_main_sub_spread_rolling_zscore_192` (retaining the well-behaved 48-bar intraday version `cm_spread_rolling_zscore_48`)
- **Cross-Month Rollover Volume Shares**: `cm_main_sub_volume_share_sub`, `cm_current_main_volume_share_current`, `cm_current_sub_volume_share_current`
- **Discrete Orderbook Spreads & Spurious Trend Pulses**: `sell_spread_oe_max_trend_6`, `buy_spread_oe_max_trend_6`, `buy_spread_oe_max`, `sell_spread_oe_max`

### 2. Physical Lower Bound for Orderbook Depth Spreads
In `data_preprocess/operator_futures/cross_section/base_feature_util.py`, enforce a physical minimum depth spread lower bound of `min_depth_spread = float(max(1, depth - 1))` (4.0 ticks for 5-depth books) on `buy_spread_oe_max` and `sell_spread_oe_max`:
```python
min_depth_spread = float(max(1, depth - 1))
price_related_df["buy_spread_oe_max"] = np.clip(
    np.abs(df["bid1_price"] - df[f"bid{depth}_price"]), min_depth_spread, 50.0
)
price_related_df["sell_spread_oe_max"] = np.clip(
    np.abs(df["ask1_price"] - df[f"ask{depth}_price"]), min_depth_spread, 50.0
)
```
This guarantees that missing depth or boundary auction artifacts never collapse to 0.0 or scale to negative out-of-distribution values.

## Consequences

- **Expected Test OOD Compression**: Eliminates ~7.03 NLL of test likelihood degradation, reducing total `test_vs_train` $\Delta\text{NLL}$ by an additional ~27% to < 19.0.
- **Expected Validation Drift Reduction**: Eliminates ~40.35 NLL from `valid_vs_train`, reducing validation drift by ~24.6% to ~123.0.
- **Elimination of Single-Contract Spikes**: Completely neutralizes anomalous OOD spikes in `fu2508` (orderbook spread underflow), `fu2601` (192-bar z-score boundary saturation), and `fu2606` (volume share rollover cycle).
- **Zero Interface Disruption**: Preserves all downstream dataset and model interfaces; reward/execution schema remains unaffected.
