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

## Implementation concern: Task 6 migrated-engine pandas import scan

- file and function: `features_related/feature_util.py`, `cross_section/base_feature_util.py`, `time_operator/create_feature.py`, `time_operator/create_feature_multi_processing.py`, `time_operator/multi_processing_util.py`, `time_operator/time_operator_util.py`, `scale_describe_save/scale_save.py`
- input condition: final migrated-engine scan still finds pandas imports in legacy feature/time/scale entrypoints after focused Polars paths were added and tested.
- previous pandas output: unported legacy call paths keep accepting pandas inputs and model-boundary pandas objects.
- proposed Polars output: migrated downscale, feature-generation focus paths, merge/concat/merge_clean, and commodity preprocessing return Polars outputs for the tested contracts; remaining pandas imports are compatibility boundaries, not silent engine regressions.
- recommended decision: keep the documented boundaries until each remaining entrypoint has a focused Polars contract test and a safe model-boundary conversion decision.
- user decision: pending

## Smoke limitation: Binance futures raw sample

The workspace does not contain a small local `DOWNLOAD_DATASET/binance-futures` raw sample for CLI smoke execution. Compatibility is covered by focused Polars unit tests for downscale, feature generation, merge, scale, and feature-selection semantics.

## Manual timing: representative preprocessing path

- Command: `python -m operator_futures.commodity.downscale_single_day --input data/原始下载/燃料油/2023/fu2305-2023-01-04.csv --output_dir /tmp/finetf-polars-bench-small --symbol fu --target_freq 5min`
- Input dataset: `data/原始下载/燃料油/2023/fu2305-2023-01-04.csv`
- Before pandas runtime: unavailable for a comparable pass because the legacy pandas implementation fails the quote-window validation on this sample
- After Polars runtime: completed successfully
- Improvement: not comparable on this sample
- Meets expected 30% improvement: no
