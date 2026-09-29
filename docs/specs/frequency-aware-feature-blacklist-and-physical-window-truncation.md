# Spec: Frequency-Aware Feature Blacklist and Physical Window Truncation

- **Triage Label**: `ready-for-agent`
- **Related Research**: `docs/research/feature_optimization_and_dual_stream_ood_architecture_research.md`
- **Related ADR**: `docs/adr/0032-frequency-aware-feature-blacklist-and-physical-window-truncation.md`
- **Target Subsystem**: Commodity Futures Feature Selection & Preprocessing Pipeline (`fu_full_process.sh`)

---

## Problem Statement

When evaluating 10-minute commodity futures strategies (e.g. `fu`), the high-level VAE routing mechanism triggers severe out-of-distribution (OOD) alarms, causing out-of-sample Negative Log-Likelihood (NLL) to collapse from $+8.87$ to $-30$. This causes the high-level routing probabilities to sink below defensive thresholds, forcing the agent to spend over 98% of the trading period locked in defensive empty positions and diluting cumulative returns down to ~0.5%.

Mathematical error decomposition revealed that 19 multi-day and multi-week macro features ($w \ge 96$ and $w \ge 192$) contributed over 50% of this total OOD degradation due to 2.34x variance expansion and secular mean shifts between the 2023 training regime and 2025/2026 test regimes. 

However, window step counts ($w$) are not scale-free; their physical calendar duration depends entirely on the bar sampling frequency:
- At 10-minute sampling, $w=96$ (16 hours) and $w=192$ (32 hours) span 2 to 5 trading days, leaking secular macroeconomic drift into short-term execution models.
- At 5-minute sampling, $w=96$ represents exactly 8 hours (~1 trading day), capturing the essential intraday volume and momentum cycle, while $w=192$ spans multiple days.
- At 1-minute sampling, $w=96$ (1.6 hours) and $w=192$ (3.2 hours) are purely intraday microstructure and momentum signals.

The current system relies on a single static global blacklist (`COMMODITY_FU_FEATURE_BLACKLIST`). Applying a blanket blacklist across all frequencies either destroys valid intraday Alpha in 1-minute and 5-minute models or allows catastrophic multi-day macro drift in 10-minute models. Furthermore, running ad-hoc offline feature slicing breaks pipeline reproducibility and creates divergence between raw feather datasets, manifest metadata, and model state inputs.

## Solution

Implement a **Frequency-Aware Feature Blacklist** architecture using shell array dispatch in `fu_full_process.sh`:
1. Decompose the feature blacklist into a universal base layer (`COMMODITY_COMMON_FEATURE_BLACKLIST`) and frequency-tailored physical window truncation layers (`COMMODITY_10MIN_FEATURE_BLACKLIST`, `COMMODITY_5MIN_FEATURE_BLACKLIST`, `COMMODITY_1MIN_FEATURE_BLACKLIST`, etc.).
2. In 10-minute preprocessing, dynamically merge the common blacklist with the 10-minute blacklist, eliminating all 19 $w \ge 96$ and $w \ge 192$ macro features.
3. In 5-minute preprocessing, preserve $w=96$ (1 trading day) while blacklisting $w \ge 192$ (multi-day).
4. In 1-minute preprocessing, preserve both $w=96$ and $w=192$ as intraday features.
5. Re-run the canonical multi-contract preprocessing pipeline (`feature_selection_train` -> `feature_selection_valid` -> `scale_save` -> `commodity_data_handler_10min_fu.sh`) to regenerate fully consistent, clean 71-dimensional state feature arrays, manifest records, and scaled dataset splits for 10-minute `fu`.
6. Strictly bound the scope to feature engineering and blacklisting, leaving dual-stream decoupling and low-level operator math untouched for this iteration.

## User Stories

1. As a quantitative researcher, I want the 10-minute feature selection pipeline to automatically exclude multi-day macro features ($w \ge 96$), so that VAE likelihood collapse and defensive lockout are prevented during out-of-sample forward testing.
2. As a short-term trading agent developer, I want 5-minute preprocessing to retain $w=96$ features, so that full single-day intraday cycles and momentum signals remain accessible to the policy.
3. As a high-frequency trading researcher, I want 1-minute preprocessing to retain $w=96$ and $w=192$ features, so that multi-hour intraday order flow and momentum dynamics are not discarded by a coarse global blacklist.
4. As an ML engineer, I want the feature blacklist dispatch to be driven by `target_freq` within the preprocessing shell script, so that executing different frequency pipelines requires zero manual configuration editing.
5. As a system maintainer, I want common non-stationary artifacts (raw prices, unscaled depth increments, discrete limit indicators, uncentered price pivots) to remain in a shared common blacklist, so that bug fixes and OOD remediations apply uniformly to all sampling rates.
6. As a researcher running backtests, I want the feature selection manifest to record the exact frequency-specific blacklist and the list of dropped features, so that state feature lineage is fully auditable.
7. As a data pipeline operator, I want the dataset generation to flow through standard pipeline steps (`feature_selection` -> `scale_save` -> `commodity_data_handler`), so that Feather column schemas and `state_features.npy` stay strictly aligned without manual file editing.
8. As a test engineer, I want automated regression tests verifying that the shell script correctly supplies frequency-specific blacklists based on `target_freq`, so that regressions in blacklist routing are caught prior to execution.
9. As an RL researcher, I want the state space dimensionality for 10-minute `fu` to compress from 90 to 71 without breaking downstream observation tensor shapes, so that subsequent agent pretraining and VAE evaluation execute seamlessly.
10. As a project architect, I want architectural decisions to be formally documented in ADR-0032 and domain glossary terms added to `CONTEXT.md`, so that future engineers understand why window length thresholds vary across frequencies.

## Implementation Decisions

- **Frequency-Aware Partitioning**: The static blacklist array is replaced by modular arrays:
  - `COMMODITY_COMMON_FEATURE_BLACKLIST`: Universal exclusions (raw prices, raw volumes, ADR-0019/0020/0023/0028/0030 non-stationary artifacts).
  - `COMMODITY_10MIN_FEATURE_BLACKLIST`: 19 macro features ($w \ge 96$, $w \ge 192$, $w=240$).
  - `COMMODITY_5MIN_FEATURE_BLACKLIST`: Macro features with $w \ge 192$ and $w=240$, explicitly preserving $w=96$.
  - `COMMODITY_1MIN_FEATURE_BLACKLIST`: Empty or ultra-long windows only, preserving $w=96$ and $w=192$.
  - `COMMODITY_30MIN_FEATURE_BLACKLIST`: Windows with $w \ge 48$.
- **Shell-Level Dynamic Assembly**: The function `run_commodity_feature_selection` inspects `target_freq` to concatenate `COMMODITY_COMMON_FEATURE_BLACKLIST` with the corresponding frequency array, passing the combined arguments via `--feature_blacklist`.
- **Level 1 Execution Scope**: The remediation executes from the existing `SPLIT-TRAIN-VALID-TEST` stage through `feature_selection` (train & valid), `scale_save`, and `commodity_data_handler_10min_fu.sh`, avoiding multi-hour recalculation of raw base/cross-section features while guaranteeing complete end-to-end dataset consistency.
- **Strict Architecture Boundaries**: No dual-stream state space partitioning (`vae_features` vs `agent_features`) or changes to reinforcement learning environments/routing utilities are introduced in this phase.
- **Operator Formula Stability**: Existing time-operator kernel formulas in `multi_processing_util.py` remain unchanged in this phase, as volatility features already receive log-transformation during `scale_save`.

## Testing Decisions

- **What makes a good test**: Tests must verify externally observable behavior at the highest interface seam. Specifically, tests should verify that passing different `target_freq` values to the preprocessing shell script results in the correct frequency-specific blacklist flags, and that the multi-contract feature selection engine properly excludes the targeted features while generating valid manifests.
- **Modules to test**:
  - Preprocessing shell interface tests (`data_preprocess/tests/test_commodity_main_contract_cli.py`).
  - Feature selection pipeline integration tests (`data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`).
- **Prior art**:
  - `test_commodity_full_process_shell_passes_feature_blacklist` in `data_preprocess/tests/test_commodity_main_contract_cli.py`.
  - `test_train_stage_applies_feature_blacklist_only_to_final_outputs` in `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`.

## Out of Scope

- Decoupled dual-stream state representations (separate VAE regime features vs Agent micro-trading features).
- Modifying underlying time-operator formulas in Polars/Pandas kernels (e.g. rewriting `max_*_std_norm` or introducing new derivative columns).
- Modifying reinforcement learning environment state wrappers, reward calculations, or action spaces.
- Modifying VAE architecture, training loss functions, or Softmax routing normalizers.

## Further Notes

- The 19 features blacklisted in 10-minute frequency are:
  1. `realized_volatility_192`
  2. `ema_slope_192`
  3. `log_price_slope_96`
  4. `bollinger_bandwidth_96_origin`
  5. `vma_192_std_norm_origin`
  6. `cntd_96_origin`
  7. `macro_trade_imbalance_continuous_240`
  8. `cvd_slope_192`
  9. `cvd_slope_96`
  10. `sell_volume_oe_trend_192`
  11. `wvma_192_origin`
  12. `wvma_96_origin`
  13. `imax_96_origin`
  14. `imin_96_origin`
  15. `rsv_96_std_norm_origin`
  16. `corr_192_origin`
  17. `relative_amount_192`
  18. `trend_to_noise_96`
  19. `log_return_vol_quantile_192`
- Post-execution verification will confirm the reduction of `dataset/10min/fu/state_features.npy` from 90 to 71 features.
