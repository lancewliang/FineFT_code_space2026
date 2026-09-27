---
status: accepted
---

# Three-Stage Remediation for VAE Feature OOD Drift

We establish a comprehensive three-stage remediation framework across feature blacklisting, operator formula fixes, and distribution-adaptive scaling to eliminate the top causes of VAE out-of-distribution (OOD) likelihood degradation.

## Context

Following the implementation of the multi-perspective OOD diagnostic matrix (ADR-0026), empirical evaluation on the 10-minute commodity futures dataset (`fu`) uncovered extreme concentration in VAE likelihood degradation:
1. **Extreme Test OOD Concentration**:
   In `test_vs_train`, the top 3 features accounted for **31.58%** of the entire OOD collapse (`sell_spread_oe_max_trend_192` at 11.12%, `cm_main_sub_open_interest_share_sub` at 10.81%, and `cntn_192_origin` at 9.66%). The top 10 features accounted for **55.82%**.
2. **Per-Contract Diagnostic Consistency**:
   Across 13 independent test contracts, `cm_main_sub_open_interest_share_sub` ranked #1 in degradation for 7 contracts, while `sell_spread_oe_max_trend_192` and `cntn_192_origin` ranked in the top 3 for 6 contracts, confirming systemic mathematical pathology rather than idiosyncratic noise.
3. **Severe Validation Volatility Expansion**:
   In `valid_vs_train`, `realized_volatility_192` alone contributed **19.41%** of total likelihood degradation, with its variance expanding to **3.35x** that of the training set.
4. **Undiscovered Operator Formula Defect**:
   Audit of `multi_processing_util.py` revealed that `max_*_std_norm`, `min_*_std_norm`, `ma_*_std_norm`, and `qtlu/d_*_std_norm` divided uncentered nominal prices by volatility (e.g. $3000 / 10 = 300$), causing massive value spikes under low volatility and non-stationary level shifts across years.

## Decision

Following the architectural grilling session, we establish a unified three-stage remediation policy:

### Stage 1: High-Priority Feature Blacklist Expansion
Add the following 6 deterministic non-stationary and pathological features to `COMMODITY_FU_FEATURE_BLACKLIST` in `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`:
1. `sell_spread_oe_max_trend_192`: Eliminates micro-tick spread zero-division and clipping boundary saturation.
2. `cm_main_sub_open_interest_share_sub`: Eliminates non-stationary contract rollover lifecycle leakage (stationary derivative `cm_open_interest_shift_speed_10m` is retained).
3. `cm_current_sub_open_interest_share_current`: Eliminates non-stationary calendar rollover cycle leakage.
4. `cm_current_main_open_interest_share_current`: Eliminates non-stationary calendar rollover cycle leakage.
5. `cntn_192_origin`: Eliminates long-window asymmetric down-bar count drift (symmetric net difference `cntd_*` is retained).
6. `cntp_192_origin`: Eliminates redundant long-window up-bar count drift.

*Impact*: Immediately cuts >41% of true test OOD likelihood degradation at the feature selection gate.

### Stage 2: In-Place Correction of Standardized Distance Operators
Correct the mathematical formulas in `data_preprocess/operator_futures/time_operator/multi_processing_util.py` under identical column names:
- $\text{max\_std\_norm}_t = \frac{P_{\max, W} - P_t}{\sigma_{P, W} + \epsilon}$ (distance to rolling maximum in units of standard deviation)
- $\text{min\_std\_norm}_t = \frac{P_t - P_{\min, W}}{\sigma_{P, W} + \epsilon}$ (distance to rolling minimum in units of standard deviation)
- $\text{ma\_std\_norm}_t = \frac{P_t - \mu_{P, W}}{\sigma_{P, W} + \epsilon}$ (distance to rolling mean in units of standard deviation)
- $\text{qtlu\_std\_norm}_t = \frac{P_{q80, W} - P_t}{\sigma_{P, W} + \epsilon}$
- $\text{qtld\_std\_norm}_t = \frac{P_t - P_{q20, W}}{\sigma_{P, W} + \epsilon}$

### Stage 3: Log-Transformation for Right-Skewed Volatility
In `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`, apply an explicit log-transformation prior to robust scaling for all volatility features (`realized_volatility_*`, `rolling_volatility_*`, `garman_klass_volatility_*`, `parkinson_volatility_*`):
$$\widetilde{\text{vol}}_t = \ln(\text{vol}_t + 10^{-6})$$
This maps the right-skewed lognormal volatility distribution onto a symmetric Gaussian distribution, preventing explosive NLL penalties in the VAE decoder during elevated-volatility market regimes.
