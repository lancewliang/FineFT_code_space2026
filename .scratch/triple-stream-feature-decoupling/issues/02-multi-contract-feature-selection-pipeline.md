# 02: Multi-Contract Feature Selection Triple-Stream Pipeline

**What to build:** The multi-contract feature selection pipeline evaluates candidate features in a single pass against both forward directional return and forward absolute dispersion. It filters three independent, orthogonal feature streams (Slope VAE, Volatility VAE, and RL Decision), constructs the mathematical union set $S_{\text{union}}$, and outputs three distinct `.npy` feature files alongside a `stream_mode: "triple"` audit manifest, rejecting legacy dual-stream and single-VAE parameters.

**Blocked by:** 01: Artifact Names & Blacklist Configuration Prefactor

**Status:** completed

- [x] `data_preprocess/operator_futures/feature_selection/muti_contract/types.py` replaces `DEFAULT_VAE_PROFILE` with `DEFAULT_VAE_SLOPE_PROFILE` (target capacity $12 \sim 16$, Spearman $|r| \le 0.65$, VIF $\le 10.0$) and `DEFAULT_VAE_VOLATILITY_PROFILE` (target capacity $10 \sim 14$, Spearman $|r| \le 0.60$, VIF $\le 8.0$, Mean PSI $\le 0.12$), while updating `FeatureSelectionPipelineConfig` to mandate both profiles and removing legacy flags.
- [x] `predictive_audit.py` computes forward directional returns $R_{t,w}$ and forward absolute returns / dispersion $|R_{t,w}|$ in a single vectorized pass, populating aggregate metrics with both directional RankIC and Volatility RankIC alongside sign consistency.
- [x] `pipeline.py` executes three independent stream evaluations via `_evaluate_stream_branch`, enforces slope Down/Flat/Up and volatility Low/Mid/High monotonicity and ANOVA F-tests, and saves `vae_slope_state_features.npy`, `vae_volatility_state_features.npy`, and `rl_state_features.npy` via `io_manager.py`.
- [x] `pipeline.py` CLI parser removes `--dual_stream`, `--no_dual_stream`, and `--vae_feature_blacklist`, accepting exclusively `--vae_slope_feature_blacklist` and `--vae_volatility_feature_blacklist`.
- [x] `FeatureSelectionManifest` records `stream_mode: "triple"` with dedicated audit blocks `vae_slope_stream`, `vae_volatility_stream`, and `rl_stream`.
- [x] `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py` asserts that three feature arrays are written, `vae_slope` contains 0 volatility indicators, `vae_volatility` contains 0 directional indicators, and all cluster count constraints are satisfied.
