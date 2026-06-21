# Compatibility Notes: migrate-operator-futures-to-polars

No unresolved compatibility differences are known at spec time.

During implementation, if a pandas behavior appears incorrect or Polars cannot reproduce a schema/dtype detail exactly, record:

- file and function
- input condition
- previous pandas output
- proposed Polars output
- recommended decision
- user decision

## Implementation concern: Task 3 legacy pandas branches

- file and function: `features_related/feature_util.py`, `cross_section/base_feature_util.py`, `time_operator/time_operator_util.py`
- input condition: Task 3 adds Polars branches for focused feature-generation paths while legacy pandas branches still exist for unported call paths.
- previous pandas output: existing pandas outputs remain available on pandas inputs.
- proposed Polars output: focused Polars inputs now return Polars DataFrames with explicit `timestamp` columns for tested quote/trade, cross-section, and OHLC rolling feature behavior.
- recommended decision: continue later tasks/final scan by removing or porting remaining legacy pandas branches only after adding focused tests for each affected path.
- user decision: pending

## Implementation concern: Task 4 scale/save and feature-selection boundaries

- file and function: `scale_describe_save/scale_save.py`, `feature_selection/*.py`
- input condition: these scripts combine preprocessing output assembly with scikit-learn/CatBoost-style consumers that commonly require pandas or NumPy boundary objects.
- previous pandas output: existing scripts read Feather with pandas and pass pandas frames/series into model or correlation code.
- proposed Polars output: merge/concat/merge_clean core joins now have Polars paths; scale/save and feature-selection need narrower tests before removing pandas imports safely.
- recommended decision: add focused tests for reward/execution manifest preservation and model-boundary conversion before removing pandas from these scripts.
- user decision: pending
