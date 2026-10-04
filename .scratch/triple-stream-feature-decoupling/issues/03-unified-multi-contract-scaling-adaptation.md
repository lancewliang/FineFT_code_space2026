# 03: Unified Multi-Contract Scaling Adaptation

**What to build:** The data scaling stage loads all three decoupled feature streams, normalizes the mathematical union $S_{\text{union}} = S_{\text{vae\_slope}} \cup S_{\text{vae\_vol}} \cup S_{\text{rl}}$ in a single pass using adaptive rolling Z-score and hyperbolic tangent soft-saturation into a single unified `df.feather`, and exports all three feature lists into the target dataset directory with zero storage multiplication.

**Blocked by:** 02: Multi-Contract Feature Selection Triple-Stream Pipeline

**Status:** completed

- [x] `data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py` loads `vae_slope_state_features.npy`, `vae_volatility_state_features.npy`, and `rl_state_features.npy` from the feature selection output directory, raising immediate `FileNotFoundError` if any stream file is absent.
- [x] `muti_contract_scale_save.py` scales all columns in the union set into the target `df.feather` files and copies all three stream arrays to the dataset directory without creating or expecting legacy `vae_state_features.npy`.
- [x] Unit test in `data_preprocess/tests/test_commodity_scale_save.py` verifies that the scaled dataset successfully provides valid in-memory column slices for slope VAE, volatility VAE, and RL agent.
