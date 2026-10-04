# 01: Artifact Names & Blacklist Configuration Prefactor

**What to build:** The system provides decoupled global artifact identifiers and stream-specific blacklist definitions for Slope VAE and Volatility VAE, completely removing the legacy single VAE state feature name without backward compatibility. Downstream developers and pipeline scripts can unambiguously reference slope and volatility feature artifacts and query their respective frequency-aware blacklists via the Python API and CLI.

**Blocked by:** None (can start immediately)

**Status:** completed

- [x] `FineFT/common/artifacts.py` declares `VAE_SLOPE_STATE_FEATURES_NPY = "vae_slope_state_features.npy"` and `VAE_VOLATILITY_STATE_FEATURES_NPY = "vae_volatility_state_features.npy"` while permanently deleting `VAE_STATE_FEATURES_NPY`.
- [x] `commodity_feature_blacklists.json` partitions VAE blacklists into `scopes.vae_slope` (blocking pure volatility/dispersion metrics) and `scopes.vae_volatility` (blocking signed directional trend/momentum metrics), with frequency-aware overrides updated for 1min, 5min, 10min, and 30min.
- [x] `data_preprocess/operator_futures/feature_selection/blacklists.py` supports querying `vae_slope` and `vae_volatility` stream blacklists via `get_commodity_stream_blacklists` and CLI `--stream vae_slope|vae_volatility|rl|global`.
- [x] Unit tests in `data_preprocess/tests/test_commodity_feature_blacklists_triple_stream.py` pass cleanly, validating artifact constant imports and blacklist resolution.
