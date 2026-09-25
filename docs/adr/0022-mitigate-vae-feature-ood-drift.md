---
status: accepted
---

# Mitigate VAE Feature OOD Drift via Elimination, Stationary Replacement, and Robust Truncation

We establish a remediation policy to resolve catastrophic VAE out-of-distribution (OOD) likelihood degradation caused by non-stationary lifecycle features, wide-window extreme-value indices, and fat-tailed orderbook depth increments.

## Context

Running the feature-level VAE OOD diagnostic (`FineFT/script/analysis/feature/vae_feature_ood_fu_10.sh`) on test contracts against the validation baseline identified the top features driving negative log-likelihood (NLL) collapse:
1. **`imin_192_origin` (146.81% contribution, +1.47 $\Delta$NLL)**: Rolling argmin of low price across a 192-bar window (~32 trading hours / 4 trading days). Prolonged trends in out-of-sample test contracts pinned the minimum to window boundaries, shifting the mean from 0.143 to 0.337 ($0.35$ mean shift) and violating density stationarity.
2. **`ask_size_topk_size_5_increments` (105.59% contribution, +1.06 $\Delta$NLL) and `bid_size_topk_size_5_increments` (57.98% contribution, +0.58 $\Delta$NLL)**: Microstructure orderbook depth level-5 queue sizes minus best size exhibit severe positive skewness and fat-tailed spikes. The default scaling bounds of $[-20.0, 20.0]$ allowed extreme values (e.g., scaled values $> 5.0$), which generate catastrophic Gaussian NLL quadratic penalties $\frac{1}{2}(x-\mu)^2/\sigma^2$ under VAE likelihood estimation.
3. **`contract_life_remaining_ratio` (65.98% contribution, +0.66 $\Delta$NLL)**: Contract lifecycle progress metric whose sampling profile shifted between validation and test contract sets (mean shift $0.44$, variance ratio $0.63$).

## Decision

1. **Eliminate `contract_life_remaining_ratio`**:
   - Remove `contract_life_remaining_ratio` from `BASE_TIME_FEATURE_COLUMNS` in `fu_full_process.sh`.
   - Add `contract_life_remaining_ratio` to `COMMODITY_FU_FEATURE_BLACKLIST` to ensure it is dropped during feature selection and excluded from downstream state representation.

2. **Stationary Replacement of `imin_192_origin`**:
   - Add `imin_192_origin`, `imin_192`, `imax_192_origin`, and `imax_192` to `COMMODITY_FU_FEATURE_BLACKLIST`.
   - Rely on stationary relative range metrics (e.g., `rsv_192` and rolling relative bounds) which naturally normalize price positions within $[0, 1]$ without boundary locking.

3. **Long-Tail Truncation for Microstructure Depth Increments and Scaled Features**:
   - Tighten default clipping bounds in `muti_contract_scale_save.py` from $[-20.0, 20.0]$ to $[-5.0, 5.0]$.
   - Explicitly configure `--clip_min -5.0 --clip_max 5.0` in `run_commodity_scale_save` in `fu_full_process.sh`.
   - Truncating robust-scaled features to $[-5.0, 5.0]$ clips fat-tail microstructure shocks to within 5 standard/IQR units from median, preserving $>99.9999\%$ of regular distribution density while capping quadratic Gaussian NLL penalties.
