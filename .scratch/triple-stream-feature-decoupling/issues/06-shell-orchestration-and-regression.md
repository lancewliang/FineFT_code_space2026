# 06: Shell Orchestration Pipelines & End-to-End Regression

**What to build:** All full-process pipeline shell scripts and commodity workflow entry points are updated to pass decoupled slope and volatility blacklist parameters and invoke data creation with isolated method paths. The complete end-to-end commodity futures pipeline runs cleanly from feature selection through high-level routing evaluation with zero backward-compatibility warnings.

**Blocked by:** 05: Dual-Axis High-Level VAE Routing Decoupled Inference

**Status:** completed

- [x] Shell scripts (`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`, `FineFT/script/data/commodity_data_handler_fu_10.sh`, `run_fu_10min_pipeline.sh`) are updated to remove `--dual_stream` and `--vae_feature_blacklist`, supplying `--vae_slope_feature_blacklist` and `--vae_volatility_feature_blacklist`.
- [x] Data handler scripts verify copying of `vae_slope_state_features.npy` and `vae_volatility_state_features.npy` into dataset roots.
- [x] End-to-end dry-run or smoke test of the 10-minute commodity pipeline executes successfully through feature selection, data scaling, VAE data creation, VAE training, and high-level routing evaluation.
