# 05: Full Pipeline Orchestration and Regression

**What to build:** Update commodity futures orchestration scripts (`fu_full_process.sh` and related runners) to support dual-stream parameters, run the pipeline end-to-end from dual-stream selection through unified scaling to VAE data creation, and verify that the full test suite (127+ tests) passes with zero regressions.

**Blocked by:** 03: Dynamic VAE Data Generation and Model Adaptation, 04: Unified Scale-Save on Union State Features

**Status:** completed

**Deliverables (交付物):**
- Orchestration shell script: Updated `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- End-to-end execution artifact set: Validated end-to-end run producing `vae_state_features.npy`, `rl_state_features.npy`, `state_features.npy`, scaled `df.feather`, and adapted VAE dataset arrays
- Automated regression suite verification: All 127+ regression tests passing in CI/test runner
- Documentation: Updated execution records and parameter references

- [x] `fu_full_process.sh` correctly invokes dual-stream feature selection and orchestrates subsequent scaling and VAE preparation stages.
- [x] End-to-end execution completes without error, verifying pipeline flow from raw features to VAE datasets.
- [x] All 127+ unit and integration tests in the repository pass cleanly (`pytest`).
- [x] Manifest audit verifies that `rl_state_features.npy` has $50 \sim 65$ features and `vae_state_features.npy` has $12 \sim 18$ features.
