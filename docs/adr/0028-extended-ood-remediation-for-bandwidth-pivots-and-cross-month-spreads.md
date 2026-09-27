---
status: accepted
---

# Extended OOD Remediation for Bandwidth, Price Pivots, and Cross-Month Spread Velocity

We establish an extended remediation policy to resolve the secondary cluster of VAE out-of-distribution (OOD) degradation revealed by the multi-perspective diagnostic matrix, targeting Bollinger bandwidth variance explosion, multi-day uncentered price ratio drift, and boundary saturation in cross-month spread velocity.

## Context

Empirical evaluation of the multi-perspective VAE feature OOD matrix (ADR-0026) on the 10-minute commodity futures dataset (`fu`) revealed:
1. **Severe Validation Volatility Expansion**: Total `valid_vs_train` degradation was `+280.92` NLL (7.3x larger than `test_vs_train` at `+38.17`), with `realized_volatility_192` (rank 1, 9.43%) and `bollinger_bandwidth_96_origin` (rank 2, 7.95%) exhibiting variance ratios of 2.34x and 2.20x relative to training data.
2. **Multi-Day Nominal Price Ratio Drift**: `pivot_s1_192_origin` (rank 3 in Valid, rank 2 in Test) and `bollinger_upper_192_origin` (rank 8 in Valid, rank 8 in Test) appeared in the top 3 degradation ranks for 5 and 4 out of 13 independent test contracts, respectively. Dividing 4-day rolling extremes by instantaneous close ( / P_t$) creates persistent non-stationary level shifts during directional trends and overnight gaps.
3. **Hard Boundary Mass Pileup**: Cross-month spread velocities (`cm_m1_m2_log_price_spread_velocity_10m` and `cm_m2_m3_log_price_spread_velocity_10m`) ranked in the top 7 across both validation and test sets due to hard clipping (`clip(-0.05, 0.05)`) concentrating probability density at boundary values on contract rollover days.

## Decision

Following the architectural grilling session, we implement the following three-part remediation:

### 1. Long-Window Price Ratio Blacklist Expansion
Add 192-bar uncentered nominal price ratio features to `COMMODITY_FU_FEATURE_BLACKLIST` in `fu_full_process.sh`:
- `pivot_s1_192_origin`, `pivot_r1_192_origin`, `pivot_pp_192_origin`, `pivot_s2_192_origin`, `pivot_r2_192_origin`
- `bollinger_upper_192_origin`, `bollinger_lower_192_origin`
- `min_192_origin`, `max_192_origin`

*Rationale*: Intraday 10-minute trading models do not benefit from 32-hour nominal ratios that span multiple overnight sessions and weekend discontinuities. Shorter windows (12, 24, 48 bars) provide robust intraday support/resistance without long-period non-stationary drift.

### 2. Log-Bollinger-Bandwidth Transformation
In `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`, extend Stage 3 distribution adaptation to include all features matching `"bollinger_bandwidth"`:
2599223\widetilde{\text{bw}}_t = \ln(\text{bw}_t + 10^{-6})2599223
Because Bollinger Bandwidth (\sigma_P / \mu_P$) is an uncentered volatility metric that follows a right-skewed log-normal distribution, the logarithmic mapping normalizes it into a symmetric Gaussian distribution, suppressing variance explosion in high-volatility validation and test periods.

### 3. Soft-Saturated Cross-Month Spread Velocity
In `data_preprocess/operator_futures/commodity/cross_month_feature.py`, replace discontinuous hard boundary clipping (`.clip(-0.05, 0.05)`) with smooth hyperbolic tangent soft saturation:
2599223v_t = 0.05 \cdot \tanh\left(\frac{\Delta \ln(P_1 / P_2)_{10}}{0.02}\right)2599223
This smoothly dampens contract rollover price jumps without creating artificial probability density spikes at rigid boundaries, preserving continuous gradients for representation learning.
