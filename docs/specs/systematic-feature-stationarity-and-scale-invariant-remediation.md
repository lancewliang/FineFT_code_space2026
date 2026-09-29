# Spec: Systematic Feature Stationarity and Scale-Invariant Remediation

- **Triage Label**: `ready-for-agent`
- **Related Research**: `docs/research/feature_optimization_and_dual_stream_ood_architecture_research.md` (Section 2.2 Matrix)
- **Related ADR**: `docs/adr/0033-systematic-feature-stationarity-and-scale-invariant-remediation.md`
- **Target Subsystem**: Feature Engineering (`multi_processing_util.py`) and Robust Scaling (`muti_contract_scale_save.py`)

---

## Problem Statement

Following the frequency-aware feature blacklist (ADR-0032), multi-day macroeconomic window drift was eliminated from 10-minute feature spaces. However, the Systematic Feature Optimization Matrix in `docs/research/feature_optimization_and_dual_stream_ood_architecture_research.md` identifies remaining structural vulnerabilities in feature representations:

1. **Extreme Normalization Division Explosion (`max_*_std_norm`, `min_*_std_norm`)**:
   In `multi_processing_util.py`, historical formulas calculated:
   $$\text{max}_{w}\_\text{std\_norm} = \frac{\text{close\_max}_{w} - \text{close}}{\sigma_{w} + \epsilon}$$
   $$\text{min}_{w}\_\text{std\_norm} = \frac{\text{close} - \text{close\_min}_{w}}{\sigma_{w} + \epsilon}$$
   When rolling price variance $\sigma_{w}$ approaches zero during quiet market sessions, dividing by $\sigma_{w}$ creates massive outlier spikes. Furthermore, dividing by price volatility rather than price level leaks asset price scale.
2. **Secular Trend Drift in Directional Counts (`cntd_*`)**:
   Historical formulas computed raw count difference:
   $$\text{cntd}_{w} = \text{cntp}_{w} - \text{cntn}_{w}$$
   During extended multi-week or multi-month trending market regimes, $\text{cntd}_{w}$ accumulated severe secular mean drift (exceeding $0.70\sigma$), causing persistent out-of-sample OOD degradation.
3. **Log-Normal Skewness in Volume Ratios and Trading Activity Indicators**:
   Volume indicators such as volume-to-moving-average ratios (`vma_*`), volume-volatility ratios (`wvma_*`), and relative volume/amount ratios (`relative_volume_*`, `relative_amount_*`) naturally follow right-skewed log-normal distributions. Directly applying linear robust scaling without log-compression preserves extreme right tails, violating the Gaussian observation prior assumed by VAEs.

## Solution

1. **In-place Formula Repairs in Time Operators**:
   - In `data_preprocess/operator_futures/time_operator/multi_processing_util.py` (both `_process_ohlcv_single_window_polars` and `_process_ohlc_single_window_polars`):
     - Replace `((close_max - close) / close_std)` with `((close_max - close) / (close + min_value))` for `max_{w}_std_norm`.
     - Replace `((close - close_min) / close_std)` with `((close - close_min) / (close + min_value))` for `min_{w}_std_norm`.
     - Replace `(cntp - cntn)` with `((cntp - cntn) / (cntp + cntn + 1e-6))` for `cntd_{w}`.
   - Apply identical formula updates in `data_preprocess/operator_futures/feature_validation/pandas_reference/time_operator/multi_processing_util.py` to preserve full numerical parity between Polars and reference Pandas implementations.
2. **Volume & Activity Log Transformation in Scale Save**:
   - In `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`, expand the log-transformation policy:
     - Volatility patterns: `realized_volatility`, `rolling_volatility`, `garman_klass_volatility`, `parkinson_volatility`, `historical_volatility`, `bollinger_bandwidth`.
     - Non-negative volume activity patterns: `relative_volume`, `relative_amount`, `vma_`, `wvma_`.
     - Apply $x' = \ln(\max(x, 0.0) + 10^{-6})$ prior to quantile fitting and scaling.
     - Strictly guard signed directional features (`trend`, `slope`, `imblance`, `log_return`, `diff`, `persistence`, `interaction`) so they are never log-transformed.

## User Stories

1. As a quant researcher, I want `max_{w}_std_norm` and `min_{w}_std_norm` to represent relative percentage price distance rather than inverse volatility, so that periods of near-zero volatility do not produce massive outlier spikes.
2. As a trading policy developer, I want `cntd_{w}` to be strictly bounded within $[-1.0, 1.0]$, so that long-lasting trending markets do not cause cumulative out-of-distribution drift.
3. As a machine learning engineer, I want right-skewed volume activity features like `vma_` and `relative_volume` to be log-transformed before robust scaling, so that the observation space adheres closer to standard Gaussian distributions required by VAE state reconstruction.
4. As a test engineer, I want Polars time operators and reference Pandas time operators to produce mathematically identical outputs for all repaired features, guaranteeing regression safety.
5. As a pipeline operator, I want signed features (e.g. `buy_volume_oe_trend`, `imblance_volume_oe_trend`, `sell_volume_oe_log_return`) to remain unaffected by log transformation, ensuring that directional sign semantics are preserved.

## Implementation Decisions

- **Polars Expression Updates**:
  In `_process_ohlcv_single_window_polars` and `_process_ohlc_single_window_polars`:
  - `max_{w}_std_norm`: `((close.rolling_max(window) - close) / (close + min_value)).alias(f"max_{window}_std_norm")`
  - `min_{w}_std_norm`: `((close - close.rolling_min(window)) / (close + min_value)).alias(f"min_{window}_std_norm")`
  - `cntd_{w}`:
    ```python
    cntp_col = (pl.col("__ret1").gt(0).cast(pl.Float64).rolling_sum(window) / window)
    cntn_col = (pl.col("__ret1").lt(0).cast(pl.Float64).rolling_sum(window) / window)
    ((cntp_col - cntn_col) / (cntp_col + cntn_col + 1e-6)).alias(f"cntd_{window}")
    ```
- **Pandas Reference Updates**:
  Update `pandas_process_ohlcv_single_window` and `pandas_process_ohlc_single_window` in `pandas_reference/time_operator/multi_processing_util.py` to match the exact same formulas.
- **Scale Save Pattern Dispatch**:
  Define `NON_NEGATIVE_ACTIVITY_PATTERNS`:
  `("relative_volume", "relative_amount", "vma_", "wvma_")`
  Define `SIGNED_FEATURE_EXCLUSIONS`:
  `("trend", "slope", "imblance", "log_return", "diff", "persistence", "interaction")`
  A feature undergoes log-transformation if:
  - It matches `VOLATILITY_FEATURE_PATTERNS` OR
  - (It matches `NON_NEGATIVE_ACTIVITY_PATTERNS` AND does NOT match any `SIGNED_FEATURE_EXCLUSIONS`).

## Testing Decisions

- Test in `test_time_operator_polars.py`:
  - Verify `test_ohlcv_single_window_matches_pandas_reference_formulas` and `test_ohlc_single_window_matches_pandas_reference_formulas` pass with the updated formulas.
  - Verify `cntd_{w}` values remain bounded in $[-1.0, 1.0]$.
  - Verify `max_{w}_std_norm` and `min_{w}_std_norm` are non-negative and finite even with constant prices (zero volatility).
- Test in `test_commodity_scale_save_base_time.py`:
  - Verify that `vma_*` and `relative_volume_*` receive log-transformation (`is_log_transformed=True`).
  - Verify that signed volume features (e.g. `buy_volume_oe_trend_*`, `imblance_volume_oe_trend_*`) do NOT receive log-transformation (`is_log_transformed=False`).
