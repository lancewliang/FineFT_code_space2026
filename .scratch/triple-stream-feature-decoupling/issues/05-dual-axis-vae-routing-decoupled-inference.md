# 05: Dual-Axis High-Level VAE Routing Decoupled Inference

**What to build:** The high-level VAE routing system loads independent slope and volatility feature lists, dynamically binds exact input dimensions for each VAE model family, pre-slices continuous 2D NumPy arrays for each axis upon contract DataFrame loading, and routes independent feature vectors during step evaluation. High-level dual gating operates on high-contrast likelihoods with zero false OOD spikes and clear volatility margin separation.

**Blocked by:** 04: VAE Data Creation & Test-Set Physical Partitioning

**Status:** completed

- [x] `FineFT/RL/DiHFT/high_level/vae_routing_util.py` loads `vae_slope_state_features.npy` and `vae_volatility_state_features.npy`, dynamically instantiating `MLP_VAE` instances with `INPUT_DIM=len(vae_slope_indicators)` and `INPUT_DIM=len(vae_vol_indicators)`.
- [x] In `vae_routing_util.py:run_single_valid_df`, the DataFrame is pre-sliced into continuous arrays `self.vae_slope_array = self.df[self.vae_slope_indicators].values` and `self.vae_vol_array = self.df[self.vae_volatility_indicators].values`.
- [x] In `vae_routing_util.py:get_quantiles`, `vae_s_slope` is evaluated exclusively on slope VAE models and `vae_s_vol` is evaluated exclusively on volatility VAE models.
- [x] `FineFT/tests/rl/test_vae_routing_final_result.py` and `FineFT/tests/rl/test_gating_strategies.py` pass cleanly, asserting clean likelihood evaluation, expanded volatility margin, and absence of dimension mismatch errors.
