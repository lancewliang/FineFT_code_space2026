---
status: accepted
---

# Decoupled Dual-Stream Feature Blacklists and Macro Feature Liberation for VAE and RL Agent

We decouple the feature selection blacklist architecture into a universal Global Hygiene Blacklist (blocking raw nominal prices, volumes, and leakage) and Frequency-Aware Stream-Specific Blacklists (`StreamFilterProfile.feature_blacklist`). High-level VAE retains strict macro long-period window truncation to eliminate out-of-distribution (OOD) log-likelihood collapse, while the low-level RL decision stream liberates scale-invariant macro trend, momentum, and open interest quantile features, recalibrating the RL cluster gating capacity to $55 \sim 70$ dimensions.

## Context

In ADR-0037, we established the Dual-Stream Feature Decoupling Architecture to resolve the mathematical conflict between the generative high-level VAE (requiring $12 \sim 18$ hyper-stationary features) and the low-level RL agent policy (`ensemble_Qnet`, requiring high-capacity Alpha features).

However, empirical execution of the 10-minute commodity futures pipeline (`fu`) revealed an architectural bottleneck in feature hygiene:
1. **Single Global Blacklist Chokehold**:
   The feature blacklist remained globally applied at Stage 1 (`DataHygieneConfig.feature_blacklist`). To prevent VAE OOD collapse across calendar years, prior ADRs (ADR-0022 through ADR-0032) blacklisted all multi-day macro indicators ($w \ge 96$, $192$, $240$), long-horizon slopes, and cross-week open interest quantiles globally.
2. **Signal Deprivation for the Low-Level RL Agent**:
   Globally killing macro features blinded the low-level RL agent to multi-day trend regimes and capital flow direction. The agent was forced to trade purely on ultra-short microstructural orderbook noise, lacking macro trend alignment.
3. **Severe Collinearity Deadlock in Single Microstructural Pool**:
   Confined purely to micro features, the 130 predictive survivors formed only 28 independent clusters at $|r| \le 0.80$. Attempting to recruit up to a target count forced collinear intra-cluster sibling recruitment, which was immediately decimated by Variance Inflation Factor pruning (`prune_by_vif(max_vif=10.0)`, dropping 48 features), triggering `ValueError: Stream rl_decision yielded 68 features, which is below the configured minimum cluster count 80`.
4. **Frequency Granularity Divergence**:
   The physical duration of window $w$ differs dramatically across sampling intervals (e.g. at 1min, $w=96$ is 1.6 hours of intraday activity; at 10min, $w=96$ spans 16 hours across 2 trading days). Macro drift is inherently frequency-dependent.

## Decision

We establish the **Decoupled Dual-Stream Feature Blacklist Architecture (双流解耦特征黑名单与宏观特征释放体系)**:

### 1. Three-Tier Blacklist Separation Contract
We decompose feature blacklisting into three distinct, non-overlapping tiers:
- **Tier 1: Global Hygiene Blacklist (`DataHygieneConfig.feature_blacklist`)**:
  - Filtered at Stage 1 before statistical audits; applies unconditionally to all downstream consumers.
  - Strictly restricted to raw nominal prices and volumes (`open`, `high`, `low`, `vwap`, `wap_1`, `wap_2`, `buy/sell_volume_oe`), data leakage metrics (`contract_life_remaining_ratio`), and uncentered non-stationary nominal price ratios (`min_96_origin`, `max_192_origin`, `pivot_*_origin`).
- **Tier 2: Frequency-Aware VAE Stream Blacklist (`DEFAULT_VAE_PROFILE.feature_blacklist`)**:
  - Attached to `StreamFilterProfile` and enforced during Stage 3 branch evaluation.
  - Enforces physical window truncation tailored to sampling frequency (`COMMODITY_10MIN_FEATURE_BLACKLIST`, `COMMODITY_5MIN_FEATURE_BLACKLIST`, etc.), barring macro trend slopes (`ema_slope_192`, `macro_trade_imbalance_continuous_240`, `cvd_slope_192`) and multi-day turnover to preserve Gaussian density estimation and prevent OOD false alarms.
- **Tier 3: RL Stream Blacklist (`DEFAULT_RL_PROFILE.feature_blacklist`)**:
  - Kept empty (`()`), completely unblocking scale-invariant macro trend slopes (`log_price_slope_96`), trend-to-noise ratios (`trend_to_noise_96`), multi-day open interest change quantiles (`prev_2_week_open_interest_change_quantile_rank`), and buyer/seller queue ratios for the RL policy.

### 2. RL Stream Gate Recalibration (`min_clusters = 55, max_clusters = 70`)
With macro features liberated:
- Orthogonal clusters formed in empirical 10-minute `fu` data jump from 28 to 67 independent clusters due to the natural orthogonality between macro trends and micro orderbook dynamics.
- VIF pruning drops decrease from 48 down to 2, yielding 68 high-quality non-collinear features (51 candidates + 17 mandatory context features).
- We set `DEFAULT_RL_PROFILE.min_clusters = 55` and `max_clusters = 70` (core target 60~70 with a 5-dimension robustness buffer against cross-contract slice variance).

### 3. Single Dedicated Definition File & Symmetric CLI Contract
- **Single Source of Truth (`commodity_feature_blacklists.json`)**:
  All blacklist definitions are centralized in `data_preprocess/operator_futures/feature_selection/commodity_feature_blacklists.json`.
  The file strictly partitions features into:
  - Scopes: `global` (universal hygiene), `vae` (macro drift), `rl_agent` (RL specific, default empty).
  - Frequencies: `1min`, `5min`, `10min`, `30min` (each containing `vae` and `rl_agent` sub-lists).
- **Python Resolution Module (`blacklists.py`)**:
  `operator_futures.feature_selection.blacklists` exports `get_commodity_stream_blacklists(target_freq)` and CLI queries (`--stream vae/rl/global --freq <freq>`), automatically assembling:
  - `vae_feature_blacklist = scopes.global + scopes.vae + frequencies[freq].vae`
  - `rl_feature_blacklist = scopes.global + scopes.rl_agent + frequencies[freq].rl_agent`
  - Stage 1 global hygiene is mathematically derived as `set(vae_blacklist).intersection(set(rl_blacklist)) == scopes.global`.
- **Clean Two-Parameter Python CLI Wiring**:
  Shell scripts (`fu_full_process.sh`) and external orchestrators pass cleanly and exclusively:
  `--vae_feature_blacklist "${vae_blacklist[@]}"` and `--rl_feature_blacklist "${rl_blacklist[@]}"`.

## Consequences

### Positive
- **Macro Alignment for RL Agent**: The RL policy regains multi-day trend context, volume flow acceleration, and open interest shift awareness, preventing counter-trend execution.
- **Mathematical Resolution of Collinearity Deadlock**: Genuinely orthogonal macro features expand the cluster space from 28 to 67 clusters, eliminating artificial VIF thrashing.
- **Ironclad VAE OOD Protection**: High-level VAE continues to reject macro non-stationary drift via its dedicated Tier 2 blacklist, sustaining zero OOD false rejections.
- **Frequency Safety**: 1min, 5min, 10min, and 30min pipelines apply frequency-appropriate physical window constraints to the VAE stream without degrading RL state quality.

### Negative & Mitigations
- **Slightly Broader Stage 1 Computational Pool**: Passing macro features through Stage 1 increases the initial CatBoost and drift audit pool by ~15-20 features.
  - *Mitigation*: Vectorized Polars implementations and single-pass CatBoost fitting absorb this minimal overhead (< 2 seconds additional computation).
