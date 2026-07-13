# Close Issues: refactor-commodity-feature-selection-union

## Verification Summary

- `conda run -n finetf pytest data_preprocess/tests -q`
  - Result: PASS
  - Evidence: `136 passed, 4 warnings in 10.91s`
- `openspec validate refactor-commodity-feature-selection-union --strict`
  - Result: PASS
  - Evidence: `Change 'refactor-commodity-feature-selection-union' is valid`

## Completeness

- `openspec/changes/refactor-commodity-feature-selection-union/tasks.md` has no unchecked task items.
- `openspec/changes/refactor-commodity-feature-selection-union/plan-ready.md` has no unchecked task completion items.
- `docs/superpowers/plans/2026-07-13-refactor-commodity-feature-selection-union.md` has no unchecked plan items.
- Implementation evidence:
  - Candidate-only IC selection: `data_preprocess/operator_futures/feature_selection/ic_correlation.py`
  - Candidate/finalize union: `data_preprocess/operator_futures/feature_selection/contract_feature_union.py`
  - Commodity shell orchestration: `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
  - Regression coverage: `data_preprocess/tests/test_feature_selection_polars.py`, `data_preprocess/tests/test_commodity_feature_pipeline.py`, `data_preprocess/tests/test_commodity_main_contract_cli.py`

## Correctness

- Candidate-only mode writes candidate artifacts without writing final `df.feather` / standard `state_features.npy`.
- Union finalize reads all contract candidates, writes品种级 `FEATURE_UNION`, validates every contract has all union features, and writes per-contract filtered `IC_RESULT` outputs.
- `scale_save` remains downstream of standard per-contract `IC_RESULT`.
- `fu_full_process.sh` runs per-contract `ic_candidate`, then one品种级 `ic_union_finalize`, then per-contract `scale_save`.

## Coherence

- Implementation follows `design.md` by keeping feature selection responsibilities in `ic_correlation.py` and `contract_feature_union.py`.
- `scale_save.py` is not changed and does not take on union or fallback behavior.
- Existing commodity manifest and Polars preprocessing patterns are preserved.

## Warnings

- Test suite emitted existing warnings:
  - Pandas `fillna(method=...)` deprecation warning in feature validation reference adapter.
  - NumPy invalid divide warnings in correlation calculations for fixture inputs.
  - Pandas log runtime warning in time operator fixture input.
- These warnings are not introduced as blocking failures by this close verification and are outside the feature-selection union behavior.

## Blocking Issues

None.

## Code Review

- Final code review found one Important issue: union finalize could write partial per-contract `IC_RESULT` outputs before a later contract failed validation.
- Resolution: added a regression assertion that missing union columns leave no partial `df.feather`, no per-contract standard `state_features.npy`, and no品种级 `FEATURE_UNION` manifest; updated `contract_feature_union.py` to validate all contracts and prepare outputs before writing any final artifacts.
- Post-fix verification:
  - `conda run -n finetf pytest data_preprocess/tests/test_commodity_feature_pipeline.py::test_write_contract_feature_union_fails_when_union_feature_missing_from_contract -q`
  - Result: PASS, `1 passed in 0.46s`
  - `conda run -n finetf pytest data_preprocess/tests -q`
  - Result: PASS, `136 passed, 4 warnings in 10.91s`
