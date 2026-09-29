---
status: accepted
---

# Frequency-Aware Feature Blacklist and Physical Window Truncation

We establish a frequency-stratified feature blacklist architecture that truncates multi-day macro rolling windows according to their physical calendar time span, resolving catastrophic out-of-distribution (OOD) degradation in 10-minute trading while preserving high-value intraday signals at higher sampling frequencies (5-minute and 1-minute).

## Context

Empirical evaluation of 10-minute commodity futures (`fu`) revealed catastrophic VAE likelihood degradation: out-of-sample Negative Log-Likelihood (NLL) collapsed from `+8.87` to `-30`, causing routing probabilities to fall below the defensive threshold and locking the strategy in defensive cash (>98% of the time).

Feature-level mathematical decomposition in `docs/research/feature_optimization_and_dual_stream_ood_architecture_research.md` identified the root cause:
- 19 long-window features with $w \ge 96$ and $w \ge 192$ contributed >50% of the total OOD degradation due to severe multi-year variance expansion (e.g. 2.34x in `realized_volatility_192`) and secular drift between 2023 training and 2025/2026 test splits.
- In 10-minute sampling, a trading day has only 36–45 bars (6–7.5 hours). Thus:
  - $w=96$ represents 16 trading hours (2–3 full trading days).
  - $w=192$ represents 32 trading hours (4–5 trading days, a full calendar week).
  - $w=240$ approaches two calendar weeks.
  These long windows introduce multi-day macroeconomic cycle shifts into short-term execution models.

However, window length $w$ is not scale-free; its physical duration $T = w \cdot \Delta t / 60$ hours varies by orders of magnitude across frequencies:
- At **1-minute** sampling: $w=96$ is only 1.6 hours; $w=192$ is 3.2 hours (half a trading day). Both are strictly intraday micro-momentum and volume signals.
- At **5-minute** sampling: $w=96$ is exactly 8 trading hours (~1 full trading day), capturing the essential intraday session cycle; while $w=192$ (16 hours) crosses multiple days.
- At **10-minute** sampling: both $w=96$ and $w=192$ cross multi-day boundaries.
- At **30-minute** sampling: $w=48$ already spans 24 hours (3–4 trading days).

Applying a single static global blacklist across all frequencies would either erroneously discard valid intraday Alpha in 1-minute and 5-minute models or permit catastrophic macro drift in 10-minute and 30-minute models.

## Decision

We establish the following remediation and architectural policies:

### 1. Frequency-Aware Feature Blacklist Architecture
Partition the feature blacklist into a common invariant foundation plus frequency-specific physical window truncation layers:

1. **Common Commodity Feature Blacklist (`COMMODITY_COMMON_FEATURE_BLACKLIST`)**:
   Enforces universal non-stationarity exclusions across all frequencies:
   - Raw nominal price levels (`open`, `high`, `low`, etc.) and raw trading turnover/volumes (`tradeval`, `buy_volume_oe`, `sell_volume_oe`, etc.).
   - Historical non-stationary and discrete artifacts from ADR-0019, ADR-0020, ADR-0022, ADR-0023, ADR-0027, ADR-0028, and ADR-0030 (raw cross-month price ratios, discrete limit indicators, unscaled depth increments, lifecycle share leakage, and uncentered nominal price pivots).

2. **10-Minute Frequency Blacklist (`COMMODITY_10MIN_FEATURE_BLACKLIST`)**:
   Blacklists all 19 $w \ge 96$ and $w \ge 192$ multi-day macro features:
   `realized_volatility_192`, `ema_slope_192`, `log_price_slope_96`, `bollinger_bandwidth_96_origin`, `vma_192_std_norm_origin`, `cntd_96_origin`, `macro_trade_imbalance_continuous_240`, `cvd_slope_192`, `cvd_slope_96`, `sell_volume_oe_trend_192`, `wvma_192_origin`, `wvma_96_origin`, `imax_96_origin`, `imin_96_origin`, `rsv_96_std_norm_origin`, `corr_192_origin`, `relative_amount_192`, `trend_to_noise_96`, `log_return_vol_quantile_192`.
   Retains intraday and single-day transition features ($w \le 48$).

3. **5-Minute Frequency Blacklist (`COMMODITY_5MIN_FEATURE_BLACKLIST`)**:
   Explicitly preserves $w=96$ (8 hours = 1 trading day) as an intraday cycle signal, while blacklisting multi-day $w \ge 192$ and $w=240$ features.

4. **1-Minute Frequency Blacklist (`COMMODITY_1MIN_FEATURE_BLACKLIST`)**:
   Preserves both $w=96$ (1.6h) and $w=192$ (3.2h) as purely intraday microstructure and momentum features.

5. **30-Minute Frequency Blacklist (`COMMODITY_30MIN_FEATURE_BLACKLIST`)**:
   Blacklists $w \ge 48$ (spanning multiple days).

### 2. Shell Dispatch Implementation
In `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`, define the frequency-specific arrays and dynamically construct the effective blacklist inside `run_commodity_feature_selection` by matching `target_freq`.

### 3. Pipeline Regeneration (Level 1 Scope)
Re-run the feature selection, scale save, and contract dataset creation pipeline (`feature_selection_train` -> `feature_selection_valid` -> `scale_save` -> `commodity_data_handler_10min_fu.sh`) to produce a verified 71-dimensional state feature representation for 10-minute `fu`.

### 4. Explicit Out of Scope
- No dual-stream state space or VAE/Agent architectural decoupling.
- No modifications to underlying time-operator mathematical formulas or low-level feature calculation kernels in this phase.

## Consequences

- **Catastrophic OOD Elimination**: Removes the top contributors accounting for >50% of VAE likelihood collapse in 10-minute data.
- **Signal Preservation in High Frequencies**: Protects 1-minute and 5-minute models from losing high-value intraday signals that share the same step numbers ($w=96, 192$).
- **Zero Interface Disruption**: Preserves the downstream single-stream RL environment contract, observation tensors, and execution schemas.
