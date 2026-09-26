---
status: accepted
---

# Scale-Invariant Feature Remediation and Selective Unblacklisting

We establish an engineering and mathematical remediation policy to transform distribution-vulnerable features into scale-invariant stationary representations, and selectively un-blacklist them while enforcing statistical feature selection competition.

## Context

Following ADR-0019, ADR-0020, ADR-0022, and ADR-0023, more than 70 features were blacklisted in `COMMODITY_FU_FEATURE_BLACKLIST` (`fu_full_process.sh`) to eliminate catastrophic VAE Negative Log-Likelihood (NLL) collapse during out-of-sample testing.

While effective at halting NLL degradation, blanket blacklisting discarded high-value quantitative trading signals:
1. **Microstructure depth queue shapes**: Raw depth size increments (`ask/bid_size_topk_size_*_increments`) were blacklisted due to liquidity growth over calendar years, losing critical queue shape and orderbook elasticity signals.
2. **Term structure mispricing**: Raw cross-month price spreads (`cm_*_log_price_ratio`, `cm_*_relative_price_spread`) were blacklisted due to macro Contango/Backwardation shifts, discarding essential basis mean-reversion signals.
3. **Multi-day range positioning**: Rolling extreme indices (`imin_192`, `imax_192`) were blacklisted due to boundary locking in strong trends, forfeiting multi-day range exhaustion indicators.
4. **Orderflow direction imbalance**: `ntrade_estimated_updown_imbalance_udnorm` was blacklisted due to discrete jump instability during low-activity night/dormant sessions.
5. **Volume volatility**: `vstd_*` was blacklisted due to denominator near-zero division during illiquid bars.

## Decision

We adopt a two-phase selective remediation architecture governed by the **Selective Unblacklisting Principle**:

### 1. Selective Unblacklisting Principle
- Only features that undergo mathematically rigorous scale-invariance transformation or are replaced by well-defined, stationary semantic new features may be removed from the feature blacklist.
- All un-remediated features (such as raw price levels, raw multi-week lagging statistics, static delivery month encodings, and discrete limit indicators) MUST strictly remain in `COMMODITY_FU_FEATURE_BLACKLIST`.

### 2. Phase 1 Implementation Decisions

1. **Orderbook Depth Share (`_share`) replacing Raw Increments**:
   - In `base_feature_util.py`, compute scale-invariant depth shares:
     $$\text{ask/bid\_size\_topk\_size\_k\_share} = \frac{\text{Ask/BidSize}_k}{\sum_{j=1}^5 \text{Ask/BidSize}_j} \in [0.0, 1.0]$$
   - Keep raw `_increments` in `COMMODITY_FU_FEATURE_BLACKLIST`.
   - Introduce `_share` columns as candidate features.

2. **Rolling Stationary Basis Z-Score (`_spread_rolling_zscore_48`) replacing Raw Spreads**:
   - In `cross_month_feature.py`, compute rolling standardized basis over a 48-bar (~2 trading days) window:
     $$Z_t = \frac{S_t - \text{SMA}_{48}(S)}{\text{Std}_{48}(S) + \epsilon}, \quad S_t = \ln(P_t^{(1)} / P_t^{(2)})$$
   - Follow contract lifetime continuous rolling across trading days, using expanding window cold-start. If $\text{Std}_{48}(S) < 10^{-6}$, clamp $Z_t = 0.0$.
   - Keep raw `_log_price_ratio` and `_relative_price_spread` in `COMMODITY_FU_FEATURE_BLACKLIST`.

3. **Relative Strength Value (`rsv_192`) replacing Boundary-Locked Indices**:
   - In `time_operator_util.py` and `multi_processing_util.py`, implement RSV:
     $$\text{RSV}_{t, 192} = \frac{P_t - \min_{192}(P)}{\max_{192}(P) - \min_{192}(P) + \epsilon} \in [0.0, 1.0]$$
   - Keep `imin_192` and `imax_192` in `COMMODITY_FU_FEATURE_BLACKLIST`.

4. **Laplace-Smoothed Trade Direction Imbalance**:
   - In `base_feature_util.py`, add pseudo-count regularization ($C = 10.0$):
     $$\text{ntrade\_estimated\_updown\_imbalance\_udnorm} = \frac{\text{up} - \text{down}}{\text{total} + 10.0} \in [-1.0, 1.0]$$
   - Un-blacklist `ntrade_estimated_updown_imbalance_udnorm` from `COMMODITY_FU_FEATURE_BLACKLIST`.

5. **Safe Volume Denominator Floor and Physical Clipping**:
   - In `multi_processing_util.py` and `time_operator_util.py`, protect `vstd_{window}` with volume floor $\max(\text{volume}, 1.0)$ and clip to $[0.0, 10.0]$.
   - Un-blacklist `vstd_*` from `COMMODITY_FU_FEATURE_BLACKLIST`.

6. **Statistical Competition over Mandatory Inclusion**:
   - Remediated and new features enter the standard feature selection pool and must compete via Rank IC, IC-IR, and correlation filtering. They are not forced into `mandatory_state_features`.

7. **Blacklisting Non-Stationary Long-Period Macro Volatility**:
   - Following empirical VAE validation on 10min data, 5 new scale-invariant features entered `state_features.npy` with strictly negative delta NLL (zero OOD drift).
   - A single residual feature, `parkinson_volatility_96`, accounted for 72.18% of positive test delta NLL (+0.72) due to macro market volatility collapse in test contracts shifting the mean by 0.416 standard deviations against a high-confidence volatility regime model.
   - Blacklist `parkinson_volatility_96` and `parkinson_volatility_192` in `COMMODITY_FU_FEATURE_BLACKLIST`, relying on stationary short/medium volatility estimators (`garman_klass_volatility_16`, `rolling_volatility_48`, `realized_volatility_192`).

## Consequences

- Fully protects the VAE state space from drastic price level and trading volume expansions.
- Successfully releases high-value orderbook depth, term structure basis, and range exhaustion Alpha signals back into feature selection.
- Preserves absolute safety: un-remediated features remain blacklisted, preventing regression to OOD likelihood explosion.
