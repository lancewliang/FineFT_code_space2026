# 02: Triple-Stream Multi-Horizon Predictive Funnel and Scale Classifier

**What to build:** The multi-contract feature selection engine automatically classifies candidate indicators into Micro, Meso, and Macro tiers via non-invasive regex naming patterns. It evaluates candidates against forward return targets matched to each tier's native horizon ($k \in [1..12]$ for Micro, $k \in [16..96]$ for Meso, $k \in [192..720]$ for Macro) using tier-specific statistical thresholds ($|\text{RankIC}|$, cross-contract sign consistency, and RankIC-IR) for the RL, Slope VAE, and Volatility VAE streams.

**Blocked by:** 01: Scale-Invariant Macro Operators and Expanding-Window Fallback

**Status:** ready-for-agent

- [x] Implement regex-based scale classifier in `data_preprocess/operator_futures/feature_selection/muti_contract/` mapping feature names into `micro`, `meso`, and `macro` tiers:
  - Macro: `r"(_(720|1440|2160)_|prev_(5|10|15|20|30)_day|prev_(1|2|4|6)_week|cm_.*_(720|1440))"`
  - Meso: `r"(_(48|96|192)_|prev_day_|prev_2_day_|session_|trading_minute_)"`
  - Micro: Default fallback for remaining features
- [x] Implement vectorized calculation of multi-horizon forward returns in `predictive_audit.py`:
  - Micro Horizons: $k \in \{1, 2, 6, 12\}$
  - Meso Horizons: $k \in \{16, 24, 48, 96\}$
  - Macro Horizons: $k \in \{192, 384, 720\}$
- [x] Implement tier-specific predictive audit gating:
  - Micro: $|\text{RankIC}| \ge 0.010$, $\text{SignConsistency} \ge 0.55$ ($k_{\text{dec}}=6$), $\text{RankIC-IR} \ge 0.18$
  - Meso: $|\text{RankIC}| \ge 0.015$, $\text{SignConsistency} \ge 0.58$ ($k_{\text{dec}}=24$), $\text{RankIC-IR} \ge 0.15$
  - Macro: $|\text{RankIC}| \ge 0.020$, $\text{SignConsistency} \ge 0.60$ ($k_{\text{dec}}=192$), $\text{RankIC-IR} \ge 0.12$
  - Evaluates directional returns for RL and Slope VAE, and absolute return dispersion for Volatility VAE.
- [x] Enforce strict blacklisting of Micro features for the Slope VAE stream (0% micro features).
- [x] Add unit tests in `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py` verifying that macro factors with strong 720-step RankIC survive filtering without being penalized by short-horizon IC filters.
