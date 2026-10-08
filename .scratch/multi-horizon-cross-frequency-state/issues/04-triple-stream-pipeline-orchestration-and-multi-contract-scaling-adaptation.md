# 04: Triple-Stream Pipeline Orchestration and Multi-Contract Scaling Adaptation

**What to build:** The full end-to-end preprocessing pipeline generates multi-horizon continuous features, executes triple-stream multi-contract feature selection, persists `rl_state_features.npy`, `vae_slope_state_features.npy`, and `vae_volatility_state_features.npy` alongside their union set $S_{\text{union}}$, and executes robust multi-contract scaling (`scale_save.py`) with zero NaNs across all processed contracts.

**Blocked by:** 03: Triple-Stream Stratified Clustering and Multi-Tier Manifest Diagnostics

**Status:** ready-for-agent

- [x] Integrate macro operators into `data_preprocess/operator_futures/time_operator/multi_processing_util.py` and continuous feature generation scripts.
- [x] Connect feature selection pipeline outputs in `data_preprocess/operator_futures/feature_selection/muti_contract/pipeline.py` to write all three `.npy` feature files and the updated manifest.
- [x] Ensure `multi_contract_scaling_util.py` and `scale_save.py` ingest the union of triple-stream features and enforce strict `validate_no_nan` assertions.
- [x] Update pipeline shell entrypoints (`scripts/fu_full_process.sh` or equivalent data preprocessing scripts) to run seamless feature generation through multi-contract scaling.
- [x] Run end-to-end integration test on sample commodity contracts verifying that all generated feature arrays, manifest files, and scaled datasets pass zero-NaN checks.
