---
status: accepted
---

# Main-Sub Relative Benchmark and Continuous Multi-Bar Macro Momentum

We replace discrete daily-broadcast quantile rankings with continuous bar-level rolling macro momentum and main-sub relative benchmark metrics, eliminating staircase deadlocks and morning jump pulses in VAE state representation.

## Context

In ADR-0019 and ADR-0020, quantile ranking (`prev_*_quantile_rank`) was introduced to tame multi-day lagging statistics (`prev_5_day_*`, `prev_week_*`). However, empirical audits revealed severe structural deficiencies:
1. **Staircase Deadlock and High Tie Rate**:
   Computing rolling 192-bar quantiles over daily-broadcast values (which change only once per trading day) produces windows containing only 3–4 distinct values. Over 26% of intraday samples shared tied ranks, creating flat constant plateaus.
2. **Session Opening Jump Pulses**:
   Every morning at 09:00 (or evening at 21:00), daily broadcast features instantaneously step to new values, injecting artificial high-frequency jump shocks into state representations.
3. **Small Cross-Section Coarseness**:
   Commodity futures have only 2–5 active contracts at any given time (Main, Sub, and 1–2 nearby delivery months). Classical cross-sectional ranking (CSRank) yields extremely coarse discrete steps ($0, 0.5, 1.0$), degrading continuous density modeling.
4. **Gaussian Prior Incompatibility**:
   Uniform $[0, 1]$ quantile distributions conflict with VAE latent space continuous Gaussian reconstruction assumptions.

Consequently, all 53 `prev_*` and `prev_*_quantile_rank` features were blacklisted in ADR-0022 and ADR-0023.

## Decision

Following the architectural grilling session, we establish four core design principles:

### 1. Continuous Multi-Bar Macro Momentum
Instead of computing multi-day quantities outside the bar pipeline and broadcasting them flatly across trading days, compute rolling multi-bar aggregations directly on intraday bars:
- **Macro Trade Direction Imbalance (`macro_trade_imbalance_continuous_240`)**:
  $$\text{Imbalance}_{t, 240} = \frac{\sum_{k=0}^{239} (\text{up}_{t-k} - \text{down}_{t-k})}{\sum_{k=0}^{239} \text{total}_{t-k} + 240 \times 10.0} \in [-1.0, 1.0]$$
- **Macro Open Interest Shift Rate (`macro_oi_change_rate_240`)**:
  $$\text{OI\_Change}_{t, 240} = \text{clip}\left(\frac{\text{OI}_t - \text{OI}_{t-240}}{\text{SMA}_{240}(\text{OI}) + 1.0}, -0.5, 0.5\right)$$
- **Macro Volume-to-OI Ratio (`macro_turnover_rate_log_240`)**:
  $$\text{Turnover}_{t, 240} = \ln\left(1.0 + \frac{\sum_{k=0}^{239} \text{Volume}_{t-k}}{\text{OI}_t + 1.0}\right)$$

Each bar updates continuously without flat plateaus or morning discontinuity steps.

### 2. Main-Sub Relative Benchmark Strength
For cross-contract positioning, anchor to the liquidity benchmark (Main Contract) rather than performing coarse ordinal ranking over small contract sets:
- **Relative Open Interest Share (`cm_main_sub_open_interest_share_sub`, `rel_to_main_oi_share`)**:
  $$\text{OI\_Share}_{i, t} = \frac{\text{OI}_{i, t}}{\text{OI}_{\text{main}, t} + \text{OI}_{\text{sub}, t} + 1.0} \in [0.0, 1.0]$$
- **Relative Volume Share (`rel_to_main_volume_share`)**:
  $$\text{Volume\_Share}_{i, t} = \frac{\text{Volume}_{i, t}}{\text{Volume}_{\text{main}, t} + \text{Volume}_{\text{sub}, t} + 1.0} \in [0.0, 1.0]$$
- **Rolling Basis Standardized Spread (`cm_*_spread_rolling_zscore_48/192`)**:
  Stationary standardized basis relative to Main contract (settled in ADR-0024).

### 3. Gaussian Probit Quantile Mapping
When rolling quantiles are computed on continuous metrics, map the percentile $p \in [\epsilon, 1-\epsilon]$ to standard normal space via the inverse error function (Probit transform):
$$\Phi^{-1}(p) = \sqrt{2} \cdot \text{erfinv}(2p - 1) \sim \mathcal{N}(0, 1)$$
This guarantees mathematical alignment with the VAE decoder's Gaussian reconstruction assumption.

### 4. Core Macro Trio Pruning Principle
In adherence to the Selective Unblacklisting Principle (ADR-0024):
- Only the core macro trio (Trade Imbalance, Turnover Rate, Open Interest Change) are remediated into continuous scale-invariant features.
- High-noise daily candlestick morphology metrics (`prev_*_upper_shadow_pct`, `prev_*_body_to_range`, `prev_*_close_position`) remain permanently in `COMMODITY_FU_FEATURE_BLACKLIST`.

## Consequences

- Eliminates intraday staircase deadlocks and morning jump shocks from the feature state space.
- Provides smooth, continuous multi-day macroeconomic momentum signals without OOD risk.
- Normalizes rank distributions to match VAE Gaussian prior expectations.
- Retains un-remediated high-noise candlestick morphology in the blacklist, ensuring safety.
