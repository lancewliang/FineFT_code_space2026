# Specification: Triple-Stream Feature Decoupling Architecture for Slope VAE, Volatility VAE, and RL Agent with Zero Backward Compatibility

## Problem Statement

In the FineFT algorithmic trading pipeline, market regimes are partitioned along two orthogonal axes:
1. **Slope Regimes (Down / Flat / Up)**: Segmented via Butterworth turning-point filtering and signed percentage slope to capture directional trend momentum.
2. **Volatility Regimes (Low / Mid / High Vol)**: Segmented via segment log-return standard deviation to capture market fluctuation dispersion.

Currently, the feature selection pipeline outputs only a single shared VAE feature list (`vae_state_features.npy`, 18 dimensions) alongside an RL decision feature list (`rl_state_features.npy`). This design causes severe mathematical, physical, and operational failures:
- **Systematic Extinction of Volatility Features**: Predictive audit in Stage 3 evaluates features exclusively against forward directional returns $R_{t,w=6}$. Pure volatility indicators (ATR, realized volatility, Bollinger bandwidth, Parkinson volatility) have zero expected directional correlation and coin-flip sign consistency ($\approx 0.50$), causing 100% of them to be eliminated by hard filters.
- **ELBO Reconstruction Contamination**: The VAE optimizes Evidence Lower Bound under Gaussian reconstruction loss $\|x - \hat{x}\|^2$. In the Slope VAE, uninformative dispersion noise dilutes directional signal-to-noise ratio by $> 50\%$. In the Volatility VAE, zero-centered directional indicators exhibit bimodal distributions during extreme rallies and sell-offs, causing massive reconstruction error spikes and high-frequency false Out-of-Distribution (OOD) circuit-breaker alarms.
- **Paralyzed High-Level Routing**: The volatility VAE fails to produce clear probability separation, keeping `volatility_margin` below the $0.12$ gating threshold and forcing `HierarchicalDualGating` into perpetual defensive shutdown or arbitrary churn.

Furthermore, carrying legacy backward-compatibility fallbacks (`vae_state_features.npy` shims) introduces architectural bloat and silent failure modes where models consume mismatched feature sets.

## Solution

Replace the dual-stream feature architecture with a pure, zero-backward-compatibility **Triple-Stream Feature Decoupling Architecture (三流解耦特征体系)**:
1. **Slope VAE Regime Stream (`vae_slope_state_features.npy`, $12 \sim 16$ dims)**: Filtered for odd-symmetric directional momentum, price deviations, order flow imbalance, and Down/Flat/Up monotonicity, while blacklisting all pure dispersion indicators.
2. **Volatility VAE Regime Stream (`vae_volatility_state_features.npy`, $10 \sim 14$ dims)**: Evaluated against forward absolute return / dispersion $|R_{t,w=6}|$ (Vol-RankIC $\ge 0.030$), filtered for Low/Mid/High Vol monotonicity and tightened Spearman clustering ($|r| \le 0.60, \text{VIF} \le 8.0$), while blacklisting all signed directional indicators.
3. **RL Decision Stream (`rl_state_features.npy`, $55 \sim 70$ dims)**: Preserves existing parameterization with liberated macro momentum and 5-level micro orderbook signals.
4. **Complete Elimination of Legacy Compatibility**: Permanently remove `vae_state_features.npy` and `VAE_STATE_FEATURES_NPY`. Downstream consumers (`vae_data_creation.py`, `VAE/main.py`, `vae_routing_util.py`) require explicit paths to their respective stream files and fail fast with `FileNotFoundError` if missing.
5. **Physical Test-Set Partitioning**: Downstream test slices are isolated into `VAE_data/slope/test/` and `VAE_data/volatility/test/` to prevent dimension collisions.
6. **Dual-Axis Dynamic Routing**: `vae_routing_util.py` dynamically binds independent input dimensions and feeds decoupled state slices (`vae_s_slope` and `vae_s_vol`) to their respective models.

## User Stories

1. As a quantitative researcher, I want the feature selection pipeline to evaluate candidate features against forward absolute return $|R_{t,w}|$, so that pure volatility and dispersion indicators are not mistakenly eliminated by directional RankIC filters.
2. As a machine learning engineer, I want the Slope VAE stream (`vae_slope_state_features.npy`) to strictly blacklist pure volatility indicators, so that the generative latent space $z$ focuses entirely on directional trend momentum without scale noise dilution.
3. As a machine learning engineer, I want the Volatility VAE stream (`vae_volatility_state_features.npy`) to strictly blacklist signed directional indicators, so that high-volatility rallies and sell-offs do not force the Gaussian decoder into bimodal failure and false OOD spikes.
4. As a pipeline operator, I want the feature selection pipeline to produce three explicit feature files (`vae_slope_state_features.npy`, `vae_volatility_state_features.npy`, `rl_state_features.npy`), so that downstream consumers never confuse or conflate state spaces.
5. As a software developer, I want the legacy `vae_state_features.npy` artifact and `ArtifactNames.VAE_STATE_FEATURES_NPY` constant to be permanently removed, so that there is no obsolete backward-compatibility baggage or silent fallback behavior.
6. As a system architect, I want downstream data loaders to fail fast with `FileNotFoundError` if a required stream feature file is missing, so that configuration errors are caught immediately at initialization rather than during training.
7. As a data engineer, I want `muti_contract_scale_save.py` to compute the mathematical union of all three feature streams ($S_{\text{union}} = S_{\text{vae\_slope}} \cup S_{\text{vae\_vol}} \cup S_{\text{rl}}$) and scale it into a single wide `df.feather`, so that disk storage and preprocessing compute are not multiplied.
8. As a researcher, I want the Volatility VAE stream to apply tightened Spearman clustering ($|r| \le 0.60$) and variance inflation factor limits ($\text{VIF} \le 8.0$), so that the $10 \sim 14$ selected features span diverse physical dispersion mechanisms (Parkinson, Bollinger, ATR, realized vol, intraday time) rather than collapsing onto multi-window duplicates of a single metric.
9. As a researcher, I want the Volatility VAE stream to enforce a Mean PSI threshold of $\le 0.12$ and Max Pair PSI of $\le 0.25$, so that financial volatility clustering across contract cycles is accommodated without destabilizing Gaussian density estimation.
10. As a researcher, I want the Slope VAE stream to enforce strict Mean PSI $\le 0.10$, Max Pair PSI $\le 0.20$, and ANOVA $F \ge 4.0$ with Down < Flat < Up monotonic mean ordering, so that the Slope VAE operates on a hyper-stationary, high-contrast directional manifold.
11. As a data pipeline user, I want `vae_data_creation.py` to save materialized test-set contracts into method-isolated subdirectories (`VAE_data/slope/test/` and `VAE_data/volatility/test/`), so that differing feature dimensions ($12 \sim 16$ vs $10 \sim 14$) never overwrite each other.
12. As a model training runner, I want `FineFT/RL/DiHFT/VAE/main.py` to discover test contracts exclusively within `VAE_data/{labeling_method}/test/` matching the active `--labeling_method`, so that test evaluation is bound to the exact feature dimensionality of that regime axis.
13. As a trading system developer, I want `vae_routing_util.py` to independently load `vae_slope_state_features.npy` and `vae_volatility_state_features.npy`, so that `MLP_VAE` instances for slope and volatility are initialized with their respective exact input dimensions.
14. As a performance engineer, I want `vae_routing_util.py` to pre-slice continuous 2D NumPy arrays `vae_slope_array` and `vae_vol_array` upon DataFrame loading, so that per-step inference incurs zero column-lookup or memory-allocation overhead.
15. As an execution policy developer, I want the high-level `HierarchicalDualGating` strategy to receive distinct `slope_weights` and `volatility_weights` generated from clean, decoupled VAE log-likelihoods, so that `volatility_margin` expands above $0.20$ and eliminates false margin ambiguity shutdowns.
16. As a DevOps engineer, I want the feature selection CLI to reject legacy `--dual_stream` and `--vae_feature_blacklist` arguments, accepting exclusively `--vae_slope_feature_blacklist` and `--vae_volatility_feature_blacklist`, so that shell scripts and orchestrators maintain clean, unambiguous interfaces.
17. As a QA engineer, I want automated unit tests in `test_commodity_multi_contract_feature_selection.py` to assert that zero volatility indicators appear in `vae_slope_state_features.npy` and zero directional indicators appear in `vae_volatility_state_features.npy`, so that decoupling regressions are caught in CI.
18. As a QA engineer, I want automated tests to verify that `FeatureSelectionManifest` serializes clean `vae_slope_stream`, `vae_volatility_stream`, and `rl_stream` audit records with `stream_mode: "triple"`, so that complete auditability is preserved.

## Implementation Decisions

### 1. Module Boundaries and Interface Overhauls
- **`data_preprocess/operator_futures/feature_selection/muti_contract/types.py`**:
  - Replace `DEFAULT_VAE_PROFILE` with two strongly typed frozen profiles: `DEFAULT_VAE_SLOPE_PROFILE` (name: `"vae_slope"`, $K \in [12, 16]$, $|r| \le 0.65$, $\text{VIF} \le 10.0$) and `DEFAULT_VAE_VOLATILITY_PROFILE` (name: `"vae_volatility"`, $K \in [10, 14]$, $|r| \le 0.60$, $\text{VIF} \le 8.0$, $\text{Mean PSI} \le 0.12$).
  - `DEFAULT_RL_PROFILE` parameters remain strictly unchanged ($K \in [55, 70]$, $|r| \le 0.80$, $\text{VIF} \le 10.0$, $\text{Mean PSI} \le 0.25$).
  - Update `FeatureSelectionPipelineConfig`: replace `dual_stream: bool = True` and `vae_profile` with `vae_slope_profile: StreamFilterProfile` and `vae_volatility_profile: StreamFilterProfile`. Remove legacy boolean flags.
- **`data_preprocess/operator_futures/feature_selection/commodity_feature_blacklists.json`**:
  - Partition VAE blacklist definitions into `scopes.vae_slope` (blocking all pure volatility, dispersion, and band indicators) and `scopes.vae_volatility` (blocking all signed directional trends, price differences, and momentum indicators).
  - Update frequency-specific sections to declare `vae_slope` and `vae_volatility` sub-lists.
- **`data_preprocess/operator_futures/feature_selection/muti_contract/predictive_audit.py`**:
  - In `execute_predictive_audit`, compute both forward directional return $R_{t,w} = (P_{t+w} - P_t) / P_t$ and forward absolute return $V_{t,w} = |R_{t,w}|$ in a single vectorized pass.
  - Compute directional RankIC ($R_{t,w}$) and Volatility RankIC ($V_{t,w}$) across candidate features. Populate aggregate metric frames with both `RankIC_Mean` (directional) and `VolRankIC_Mean` (dispersion) alongside respective sign consistency scores.
- **`data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py`**:
  - Implement `_run_triple_stream_train_stage` dispatching three independent branch evaluations via `_evaluate_stream_branch`: Slope Stream, Volatility Stream, and RL Stream.
  - Build mathematical union $S_{\text{union}} = S_{\text{vae\_slope}} \cup S_{\text{vae\_vol}} \cup S_{\text{rl}}$.
  - Update CLI parser: remove `--dual_stream`, `--no_dual_stream`, `--vae_feature_blacklist`. Introduce `--vae_slope_feature_blacklist` and `--vae_volatility_feature_blacklist`.
- **`data_preprocess/operator_futures/feature_selection/muti_contract/io_manager.py`**:
  - Replace `save_dual_stream_features` with `save_triple_stream_features`: outputs `vae_slope_state_features.npy`, `vae_volatility_state_features.npy`, and `rl_state_features.npy`.
  - Update `FeatureSelectionManifest` schema to record `stream_mode: "triple"` with dedicated audit blocks `vae_slope_stream`, `vae_volatility_stream`, and `rl_stream`.
- **`data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py`**:
  - Load all three feature stream arrays, compute $S_{\text{union}}$, and execute single-pass adaptive rolling Z-score and soft-saturation scaling.
  - Copy all three stream files to the output dataset directory.
- **`FineFT/common/artifacts.py`**:
  - Delete `VAE_STATE_FEATURES_NPY`.
  - Register `VAE_SLOPE_STATE_FEATURES_NPY = "vae_slope_state_features.npy"` and `VAE_VOLATILITY_STATE_FEATURES_NPY = "vae_volatility_state_features.npy"`.
- **`FineFT/datahandler/vae_data_creation.py`**:
  - Select `VAE_SLOPE_STATE_FEATURES_NPY` when `--labeling_method slope` and `VAE_VOLATILITY_STATE_FEATURES_NPY` when `--labeling_method volatility`. Fail fast if file is missing.
  - Isolate test contract output to `VAE_data/{labeling_method}/test/`.
- **`FineFT/RL/DiHFT/VAE/main.py`**:
  - Update `discover_test_sources` to require test contracts under `VAE_data/{labeling_method}/test/`.
- **`FineFT/RL/DiHFT/high_level/vae_routing_util.py`**:
  - Load `vae_slope_indicators` and `vae_volatility_indicators`. Initialize `MLP_VAE` instances with dynamic dimensions $\text{INPUT\_DIM}_{\text{slope}}$ and $\text{INPUT\_DIM}_{\text{volatility}}$.
  - In `run_single_valid_df`, pre-slice `vae_slope_array` and `vae_vol_array`. In `get_quantiles`, pass `vae_s_slope` to slope models and `vae_s_vol` to volatility models.

## Testing Decisions

### Test Characteristics and Best Practices
- **Strict External Behavior Testing**: Tests must execute public functions (`run_feature_selection`, `make_data`, `vae_risk_aware_routing`) and assert outputs, file artifacts, and manifest contents rather than asserting internal private variables.
- **In-Memory Synthesized Datasets**: Unit tests must use minimal synthesized multi-contract Polars frames with known directional trends and volatility bursts, executing in $< 2$ seconds without disk fixture dependencies.
- **Fail-Fast Verification**: Specific test cases must verify that omitting either slope or volatility feature files causes immediate `FileNotFoundError` without fallback.

### Modules Under Test
1. `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`:
   - Verify triple-stream artifact generation (`vae_slope_state_features.npy`, `vae_volatility_state_features.npy`, `rl_state_features.npy`).
   - Verify that no directional features leak into volatility stream and no volatility features leak into slope stream.
   - Verify cluster dimension bounds ($12 \le N_{\text{slope}} \le 16$, $10 \le N_{\text{vol}} \le 14$, $55 \le N_{\text{rl}} \le 70$).
   - Verify CLI parsing with new arguments.
2. `FineFT/tests/datahandler/test_vae_data_creation.py`:
   - Verify method-specific feature loading and test-set isolation into `VAE_data/{method}/test/`.
3. `FineFT/tests/rl/test_commodity_vae_cross_contract.py`:
   - Verify that Slope VAE and Volatility VAE load independent dimensions and evaluate cleanly.
4. `FineFT/tests/rl/test_vae_routing_final_result.py`:
   - Verify end-to-end high-level routing with dual-axis decoupled inference.

### Prior Art
- `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py` provides the canonical pattern for multi-contract synthetic test fixtures and pipeline execution.
- `FineFT/tests/datahandler/test_vae_data_creation.py` provides existing patterns for checking feather slicing and `.npy` output structures.

## Out of Scope

- Modifying the underlying Butterworth filter or turning-point segment detection algorithms in `valid_cross_contract_label_calibration.py`.
- Modifying the low-level reinforcement learning agent architecture (`ensemble_Qnet`) or its training loss function.
- Changing the mathematical formulation of `HierarchicalDualGating` (the gating algorithm remains identical; only its input likelihood contrast is sharpened).
- Supporting backward compatibility for datasets generated prior to the triple-stream decoupling architecture.

## Further Notes

- Execution speed of Stage 1~3 feature selection on `fu` 10min remains under 25 seconds because all three streams share Stage 1 hygiene, Stage 2 CatBoost nonlinear fitting, and Stage 3 vectorized metric calculations.
- Disk usage remains strictly $1\times$ baseline because feature files are small NumPy index arrays ($< 4$ KB), and data scaling persists a single union `df.feather`.
