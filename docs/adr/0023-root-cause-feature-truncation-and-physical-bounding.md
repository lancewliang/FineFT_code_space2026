---
status: accepted
---

# Root-Cause Feature Truncation and Physical Bounding Across Hierarchy Levels

We establish a multi-tier physical bounding and group-specific clamping architecture to eliminate numerical explosions, near-zero denominator underflows, and fat-tailed outliers across quant feature operators and robust scaling.

## Context

Following the initial VAE out-of-distribution (OOD) mitigation in ADR-0022, a full empirical audit (`docs/research/feature_long_tail_truncation_hierarchy_and_threshold_analysis.md`) across all 119 state features over 14 train contracts, 12 valid contracts, and test contracts identified 33 features exhibiting extreme fat tails exceeding $[-5.0, 5.0]$. The audit identified mathematical root causes at the operator level:
1. **Near-zero denominator division in volume volatility (`vstd_{window}`)**:
   Dividing rolling volume standard deviation by `volume + 1e-12` when bar volume is zero or near-zero caused ratio explosions up to 20.0, resulting in a 10.20% outlier rate in test data.
2. **Multi-bar differencing jumps across session and roll boundaries (`cm_*_spread_velocity_10m`, `cm_open_interest_shift_speed_10m`)**:
   Computing 10-bar diffs across overnight sessions or delivery month rolls generated abrupt spikes, resulting in a 7.34% outlier rate.
3. **Microstructure orderbook depth illiquidity and queue increments (`buy_spread_oe_max`, `sell_spread_oe_max`, `ask/bid_size_topk_size_*_increments`)**:
   During thin orderbook intervals, depth-25 spreads widened significantly and queue increments produced high positive skewness.

## Decision

We adopt a two-tier defense architecture combining operator-level physical bounds (Tier 1) and scaling-level group clamping (Tier 2):

1. **Tier 1: Operator-Level Physical Bounds**:
   - **`vstd_{window}` Floor and Truncation**:
     In `multi_processing_util.py` and `time_operator_util.py`, floor the volume denominator with `np.maximum(df["volume"], 1.0)` / `pl.when(volume > 1.0).then(volume).otherwise(1.0)` and clip the resulting volatility ratio to `[0.0, 10.0]`.
   - **Cross-Month Differencing Bounds**:
     In `cross_month_feature.py`, clip `cm_*_log_price_spread_velocity_10m` to `[-0.05, 0.05]` and `cm_open_interest_shift_speed_10m` to `[-0.1, 0.1]`.
   - **Microstructure Orderbook Spread and Increment Bounds**:
     In `base_feature_util.py`, clip `buy_spread_oe_max` and `sell_spread_oe_max` to `[0.0, 50.0]` and clip queue size increments `ask/bid_size_topk_size_*_increments` to `[-5000.0, 5000.0]`.

2. **Tier 2: Feature-Group Robust Scaling Clamping**:
   - In `muti_contract_scale_save.py`, implement `resolve_feature_clip_bounds` to identify fat-tailed feature patterns (`vstd_`, `spread_velocity`, `shift_speed`, `_increments`, `spread_oe_max`).
   - Clamp fat-tailed features to `[-4.0, 4.0]` while preserving the global `[-5.0, 5.0]` clipping bound for general state features.

## Consequences

- Prevents extreme outliers from generating catastrophic quadratic Gaussian NLL penalties during VAE density estimation.
- Enforces numerical stability at the operator generation point before statistics calculation and scaling.
- Preserves distribution density for $>99.99\%$ of normal trading samples while eliminating spurious numerical divergence.
