# 08: Pipeline Orchestrator Slim-Down and Backward-Compatible Adapter

**What to build:** Refactor `operator_futures.feature_selection.muti_contract.pipeline` into a clean, high-level orchestrator (< 120 lines) coordinating the three-stage funnel across all extracted deep modules. Maintain `run_feature_selection` as a backward-compatible public adapter accepting both legacy keyword arguments and `FeatureSelectionPipelineConfig`, ensuring all 33 end-to-end regression tests pass cleanly.

**Blocked by:** 01-types-and-io-manager-extraction.md, 02-funnel-resequencing-and-predictive-audit.md, 03-target-horizon-catboost-with-purged-embargo.md, 04-pre-selection-winsorization-and-data-hygiene.md, 05-spearman-rank-correlation-and-ward-hierarchical-clustering.md, 06-within-contract-adf-stationarity-gate.md, 07-forward-boundary-distribution-drift-outpost.md

**Status:** completed

- [x] Refactor `pipeline.run_feature_selection` to orchestrate `io_manager`, `data_hygiene`, `distribution_audit`, `stationarity_audit`, `predictive_audit`, `nonlinear_scoring`, `orthogonal_dedup`, and `regime_audit`.
- [x] Preserve full keyword argument signature and default parameter mappings in `run_feature_selection`.
- [x] Update `__main__.py` to parse new CLI flags (`--target_decision_window`, `--dedup_method`, `--max_vif`, `--fdr_threshold`).
- [x] Run full test suite: `pytest data_preprocess/tests/test_commodity_multi_contract_feature_selection.py` to ensure all 33 tests pass without regression.
