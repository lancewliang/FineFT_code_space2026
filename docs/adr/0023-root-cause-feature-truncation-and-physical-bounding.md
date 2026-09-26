---
status: accepted
---

# Root-Cause Feature Truncation and Physical Bounding Across Hierarchy Levels

We establish a multi-tier defense architecture combining blacklist elimination of non-stationary volume quantities, operator-level physical bounds, and group-specific robust scaling clamping to eliminate VAE out-of-distribution (OOD) collapse.

## Context

Following the initial VAE OOD mitigation in ADR-0022 (eliminating `contract_life_remaining_ratio` and `imin_192_origin`), total negative log-likelihood drift fell to $-401.58$, with 103 of 115 state features demonstrating lower test NLL than the validation baseline. However, two primary residual drivers accounted for over 86% of the remaining positive OOD likelihood degradation:
1. **Unnormalized microstructure queue size increments (`ask_size_topk_size_5_increments`, `bid_size_topk_size_5_increments`)**:
   Contributing +1.30 (130.28%) and +1.04 (103.63%) to delta NLL. Because these features compute raw lot differences ($AS_k - AS_1$) across 25 orderbook depth levels, overall liquidity growth from 2023 train contracts to 2025/2026 test contracts caused structural upward mean shifts (from median 0.00 to 0.54), behaving non-stationarily across calendar years.
2. **Unnormalized trade count and ratio instability (`ntrade_estimated`, `ntrade_estimated_*_udnorm`)**:
   Contributing +0.47, +0.28, and +0.14 to delta NLL due to absolute bar-level trade frequency expansion across contract years.
3. **Operator numerical edge cases (`vstd_{window}`, `cm_*_spread_velocity_10m`)**:
   Near-zero division in `vstd` and multi-bar differencing jumps across session/roll boundaries.

## Decision

We adopt a comprehensive three-tier remediation architecture:

1. **Feature Selection Blacklist for Non-Stationary Raw Quantities**:
   - Add all 10 unnormalized microstructure depth size increments (`ask_size_topk_size_1_increments` through `5` and `bid_size_topk_size_1_increments` through `5`) to `COMMODITY_FU_FEATURE_BLACKLIST` in `fu_full_process.sh`.
   - Add raw trade count features (`ntrade_estimated`, `ntrade_estimated_up_udnorm`, `ntrade_estimated_down_udnorm`, `ntrade_estimated_updown_imbalance_udnorm`) to `COMMODITY_FU_FEATURE_BLACKLIST`.
   - Rely strictly on stationary relative orderbook metrics (e.g. `wap_balance`, `imblance_volume_oe`, and depth price increments `ask/bid_price_topk_size_*_increments` in ticks).

2. **Tier 1: Operator-Level Physical Bounds**:
   - **`vstd_{window}` Floor and Truncation**:
     In `multi_processing_util.py` and `time_operator_util.py`, floor the volume denominator with `np.maximum(df["volume"], 1.0)` / `pl.when(volume > 1.0).then(volume).otherwise(1.0)` and clip the resulting volatility ratio to `[0.0, 10.0]`.
   - **Cross-Month Differencing Bounds**:
     In `cross_month_feature.py`, clip `cm_*_log_price_spread_velocity_10m` to `[-0.05, 0.05]` and `cm_open_interest_shift_speed_10m` to `[-0.1, 0.1]`.
   - **Microstructure Orderbook Spread Bounds**:
     In `base_feature_util.py`, clip `buy_spread_oe_max` and `sell_spread_oe_max` to `[0.0, 50.0]`.

3. **Tier 2: Feature-Group Robust Scaling Clamping**:
   - In `muti_contract_scale_save.py`, implement `resolve_feature_clip_bounds` to identify fat-tailed feature patterns (`vstd_`, `spread_velocity`, `shift_speed`, `_increments`, `spread_oe_max`).
   - Clamp fat-tailed features to `[-4.0, 4.0]` while preserving the global `[-5.0, 5.0]` clipping bound for general state features.

## Consequences

- Eliminates all features causing $\Delta\text{NLL} > 0.20$ between validation baseline and out-of-sample test contracts.
- Guarantees that VAE state representation consists exclusively of scale-invariant, stationary relative financial metrics.
- Physical bounds at Tier 1 prevent numerical explosions before statistical scaling, and Tier 2 clamps fat tails safely within 4 IQR units.
