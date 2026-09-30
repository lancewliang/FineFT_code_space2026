# 01: Dual-Stream Profiles, Manifest Schema and Artifact Constants

**What to build:** Introduce the strongly typed `StreamFilterProfile` dataclass contract and official default presets (`DEFAULT_VAE_PROFILE`, `DEFAULT_RL_PROFILE`), extend `FeatureSelectionPipelineConfig` with `dual_stream: bool = True` and profile bindings, register canonical dual-stream artifact constants (`VAE_STATE_FEATURES_NPY`, `RL_STATE_FEATURES_NPY`), and update `FeatureSelectionManifest` with dual-stream audit schema fields.

**Blocked by:** None (can start immediately)

**Status:** completed

**Deliverables (交付物):**
- Data contracts: `StreamFilterProfile`, `DEFAULT_VAE_PROFILE`, `DEFAULT_RL_PROFILE` in `operator_futures.feature_selection.muti_contract.types`
- Configuration update: `dual_stream: bool = True`, `vae_profile`, `rl_profile` fields in `FeatureSelectionPipelineConfig`
- Canonical artifact constants: `VAE_STATE_FEATURES_NPY`, `RL_STATE_FEATURES_NPY` in `FineFT.common.artifacts`
- Manifest schema: `stream_mode`, `vae_stream`, `rl_stream` audit data structures in `operator_futures.feature_selection.manifests`
- Verification test suite: Unit tests in `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py` validating profile instantiation, immutability, and manifest serialization

- [x] `StreamFilterProfile` dataclass is defined with frozen immutability and complete field typing (PSI limits, correlation caps, cluster ranges, composite weights, persistence filter flags, and mandatory regex patterns).
- [x] `DEFAULT_VAE_PROFILE` and `DEFAULT_RL_PROFILE` are instantiated with canonical quantitative parameters.
- [x] `FeatureSelectionPipelineConfig` supports `dual_stream` mode and profile bindings without breaking legacy callers.
- [x] `ArtifactNames` exports `VAE_STATE_FEATURES_NPY = "vae_state_features.npy"` and `RL_STATE_FEATURES_NPY = "rl_state_features.npy"`, with legacy `STATE_FEATURES_NPY` purged and compatibility shims removed.
- [x] `FeatureSelectionManifest` serializes and deserializes dual-stream audit payloads (`stream_mode: "dual"`, `vae_stream`, `rl_stream`).
- [x] Unit tests pass verifying type validation and serialization integrity.
