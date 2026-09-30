# 01: Extract Immutable Types and Pipeline IO Manager

**What to build:** Extract pipeline configuration dataclasses (`FeatureSelectionPipelineConfig`, `DataHygieneConfig`, `StationarityAuditConfig`, `PredictiveAuditConfig`, `NonlinearScoringConfig`, `OrthogonalDedupConfig`) and standardized step result contract `PipelineStepResult` into `operator_futures.feature_selection.muti_contract.types`, and extract Polars IPC frame loading, schema checking, filtered dataset saving, and manifest formatting into `operator_futures.feature_selection.muti_contract.io_manager`.

**Blocked by:** None (can start immediately)

**Status:** completed

- [x] Create `operator_futures.feature_selection.muti_contract.types` with frozen dataclasses defining all pipeline sub-configs and `PipelineStepResult`.
- [x] Create `operator_futures.feature_selection.muti_contract.io_manager.PipelineIOManager` encapsulating `_load_contract_frames`, `_write_filtered_outputs`, and manifest serialization.
- [x] Unit tests in `test_types_and_io_manager.py` verify that `PipelineIOManager` loads frames, filters columns, writes output files, and produces valid `FeatureSelectionManifest` structures.
