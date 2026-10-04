# 04: VAE Data Creation & Test-Set Physical Partitioning

**What to build:** Downstream VAE data creation selects the exact feature stream matching the specified `--labeling_method` (`vae_slope_state_features.npy` for slope, `vae_volatility_state_features.npy` for volatility) and fails fast with `FileNotFoundError` if missing. Test-set contracts are saved into method-isolated subdirectories (`VAE_data/slope/test/` and `VAE_data/volatility/test/`), and model training test source discovery is scoped strictly to the active method directory, eliminating cross-method dimensional collisions.

**Blocked by:** 03: Unified Multi-Contract Scaling Adaptation

**Status:** completed

- [x] `FineFT/datahandler/vae_data_creation.py` reads `VAE_SLOPE_STATE_FEATURES_NPY` when `--labeling_method slope` and `VAE_VOLATILITY_STATE_FEATURES_NPY` when `--labeling_method volatility`. Any missing file triggers an immediate `FileNotFoundError` with no fallback logic.
- [x] `vae_data_creation.py` persists test-set contracts into `VAE_data/{labeling_method}/test/df_<contract>.npy`, completely eliminating the legacy unpartitioned `VAE_data/test/` directory.
- [x] `FineFT/RL/DiHFT/VAE/main.py:discover_test_sources` receives the active `labeling_method` and discovers test source files exclusively within `VAE_data/{labeling_method}/test/`.
- [x] `FineFT/tests/datahandler/test_vae_data_creation.py` and `FineFT/tests/rl/test_commodity_vae_cross_contract.py` pass cleanly, asserting independent feature dimensions and isolated test directories.
