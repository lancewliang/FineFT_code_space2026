---
status: accepted
---

# Remedy VAE OOD Features via Precedent Blacklist and Rolling Stationary Replacements

We establish a comprehensive remediation policy for features causing catastrophic VAE out-of-distribution (OOD) likelihood collapse during out-of-sample forward testing.

Following mathematical loss decomposition from the feature-level OOD analysis, three feature groups accounted for >95% of VAE negative log-likelihood (NLL) degradation:
1. **Cross-month price ratio and spread levels (`cm_*`)**: Raw log price ratios and relative spreads between current, main, and sub contracts (e.g., `cm_current_main_log_price_ratio`) flipped mean sign between validation (2024) and test (2025/2026) due to macro term structure contango/backwardation transitions, contributing ~90% of NLL explosion.
2. **Heavy-tailed multi-day lagging statistics (`prev_*`)**: Multi-week open interest percentage changes, turnover ratios, and orderflow imbalances exhibited extreme outliers exceeding 5 to 6 standard deviations in distant delivery contracts.
3. **Calendar delivery month cyclics (`contract_month_sin/cos`)**: Trigonometric delivery month encodings suffered domain shift between historical validation delivery clusters and out-of-sample test contracts.

We chose to:
- Blacklist all 6 raw cross-month price level/ratio features (`cm_current_main_log_price_ratio`, `cm_current_main_relative_price_spread`, `cm_current_sub_log_price_ratio`, `cm_current_sub_relative_price_spread`, `cm_main_sub_log_price_ratio`, `cm_main_sub_relative_price_spread`), while retaining bounded $[0, 1]$ role/share features and velocity features.
- Blacklist all 23 drifting `prev_*` statistical features, while explicitly exempting `prev_day_contract_role_tier` to ensure non-main contract defense (ADR-0017) remains fully operational.
- Directly blacklist `contract_month_sin` and `contract_month_cos` without replacement.
- Supersede portions of ADR-0003 and ADR-0004: update `data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py` so that `feature_blacklist` takes highest precedence over `mandatory_state_features`, cleanly dropping blacklisted features instead of raising a `ValueError`.
- Recommend replacing raw cross-month spreads with rolling stationary basis Z-scores (`cm_spread_rolling_zscore_k`) and replacing raw multi-day changes with rolling empirical quantile ranks (`prev_*_quantile_rank`), guaranteeing mathematical stationarity across years.
