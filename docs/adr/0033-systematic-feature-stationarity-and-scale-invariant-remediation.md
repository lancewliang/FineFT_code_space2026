---
status: accepted
---

# Systematic Feature Stationarity and Scale-Invariant Remediation

We implement in-place mathematical formula repairs and volume-activity log transformations for features identified in the Systematic Feature Optimization Matrix (`docs/research/feature_optimization_and_dual_stream_ood_architecture_research.md`), eliminating low-volatility denominator explosions, trend count drift, and heavy-tailed distribution collapse.

## Context

Following the establishment of the frequency-aware feature blacklist (ADR-0032), multi-day macroeconomic window drift ($w \ge 96$ at 10-minute frequency) was eliminated. However, empirical distribution audits in the research document revealed three remaining structural deficiencies in the candidate feature space:

1. **Extreme Normalization Division Explosion (`max_*_std_norm`, `min_*_std_norm`)**:
   In `data_preprocess/operator_futures/time_operator/multi_processing_util.py`, historical formulas calculated:
   $$\text{max}_{w}\_\text{std\_norm} = \frac{\text{close\_max}_{w} - \text{close}}{\sigma_{w} + \epsilon}$$
   $$\text{min}_{w}\_\text{std\_norm} = \frac{\text{close} - \text{close\_min}_{w}}{\sigma_{w} + \epsilon}$$
   When rolling price variance $\sigma_{w}$ approaches zero during quiet market sessions or narrow consolidation ranges, dividing by $\sigma_{w}$ creates extreme outlier spikes, severe kurtosis expansion, and spurious out-of-distribution (OOD) events. Furthermore, this formula fails to reflect the true dimensionless percentage drawdown or runup relative to the asset price level.

2. **Secular Trend Drift in Directional Counts (`cntd_*`)**:
   Historical formulas computed the raw arithmetic difference of positive and negative return bar counts:
   $$\text{cntd}_{w} = \text{cntp}_{w} - \text{cntn}_{w} = \frac{\sum \mathbb{I}(\Delta P > 0)}{w} - \frac{\sum \mathbb{I}(\Delta P < 0)}{w}$$
   During extended multi-week or multi-month trending market regimes (such as the 2024–2025 commodity bear market), $\text{cntd}_{w}$ accumulated severe secular mean drift (exceeding $0.70\sigma$), causing persistent OOD shift in out-of-sample test splits.

3. **Log-Normal Skewness in Volume Ratios and Trading Activity Indicators**:
   Volume indicators such as volume-to-moving-average ratios (`vma_*`), volume-volatility ratios (`wvma_*`), and relative volume/amount ratios (`relative_volume_*`, `relative_amount_*`) naturally follow right-skewed log-normal or chi-square distributions. Directly applying linear robust scaling (median / IQR) without log-compression preserves extreme right tails, directly violating the Gaussian observation prior assumed by VAEs and neural networks.

## Decision

We establish the following in-place mathematical remediations and preprocessing transformations:

### 1. In-Place Relative Extreme Normalization
In both the Polars time operator (`data_preprocess/operator_futures/time_operator/multi_processing_util.py`) and the reference pandas validation operator (`data_preprocess/operator_futures/feature_validation/pandas_reference/time_operator/multi_processing_util.py`):
- Replace volatility division with scale-invariant relative percentage return:
  $$\text{max}_{w}\_\text{std\_norm} = \frac{\text{close\_max}_{w} - \text{close}}{\text{close} + \epsilon}$$
  $$\text{min}_{w}\_\text{std\_norm} = \frac{\text{close} - \text{close\_min}_{w}}{\text{close} + \epsilon}$$
  where $\epsilon = 10^{-12}$.
- This guarantees scale-invariance, non-negativity, and immunity to division by near-zero rolling variance.

### 2. Bounded Relative Differencing for Directional Counts
In both time operators:
- Replace raw count difference with the self-normalizing bounded ratio:
  $$\text{cntd}_{w} = \frac{\text{cntp}_{w} - \text{cntn}_{w}}{\text{cntp}_{w} + \text{cntn}_{w} + \epsilon_{\text{count}}}$$
  where $\epsilon_{\text{count}} = 10^{-6}$.
- Properties:
  - Strictly bounded in $[-1.0, 1.0]$.
  - Identically zero when positive and negative bars balance or when the market is completely flat ($\text{cntp} = \text{cntn} = 0$).
  - Eliminates long-run mean drift during one-way secular trends.

### 3. Log Transformation for Right-Skewed Volume & Activity Features
In `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`:
- Expand the log-transformation policy from volatility features (`VOLATILITY_FEATURE_PATTERNS`) to general right-skewed heavy-tailed features (`LOG_FEATURE_PATTERNS`):
  - Volatility features: `realized_volatility`, `rolling_volatility`, `garman_klass_volatility`, `parkinson_volatility`, `historical_volatility`, `bollinger_bandwidth`.
  - Non-negative volume activity and ratios: `vma_`, `wvma_`, `relative_volume`, `relative_amount`.
- Transformation rule:
  $$x' = \ln(\max(x, 0.0) + \epsilon_{\text{log}}), \quad \epsilon_{\text{log}} = 10^{-6}$$
- Signed volume features (specifically those containing `trend`, `slope`, `imblance`, `log_return`, `diff`, or `persistence`) MUST NOT be log-transformed, preserving signed directional momentum information.

## Consequences

- Denominator underflow and extreme outlier spikes in `max_*_std_norm` and `min_*_std_norm` are eliminated.
- Directional count statistics `cntd_*` remain strictly within $[-1.0, 1.0]$ regardless of market regime duration.
- Right-skewed volume features are mapped to symmetric, near-Gaussian representations, dramatically reducing VAE reconstruction errors and preventing OOD likelihood collapse.
- Both Polars implementations and pandas reference operators maintain exact parity, passing all deterministic numerical parity tests.
