## Context

The previous Polars migration moved the front and middle parts of preprocessing forward, but the post-merge processing layer still contains pandas-heavy scripts. These scripts sit on the shared path after `MERGE_CONCAT/CONCAT_FEATURE` and are used by commodity futures and older futures workflows:

- `time_operator/create_feature_multi_processing.py`
- `time_operator/multi_processing_util.py`
- `feature_selection/*.py`
- `scale_describe_save/scale_save.py`

Commodity `main.sh` now runs a small five-day default window successfully, but this late-stage pandas dependency remains the next clear migration boundary.

## Goals / Non-Goals

**Goals:**
- Remove pandas imports from the target post-merge preprocessing files.
- Reimplement time feature, feature selection, and scale/save processing with Polars-native data operations.
- Preserve existing CLI arguments, output paths, file names, and column names.
- Support commodity depth=5 and generic depth=25 data.
- Keep ML library boundaries practical by allowing NumPy or library-native objects at the model call site.
- Verify a five-day commodity `main.sh` run reaches `SCALE_SAVE` without tracebacks.

**Non-Goals:**
- Do not migrate unrelated analysis scripts, overview scripts, tests, or FineFT training/environment code.
- Do not introduce a pandas fallback path.
- Do not introduce a benchmark suite or fixed runtime improvement gate.
- Do not redesign feature definitions, output directory layout, or CLI workflow.

## Decisions

### Decision 1: Polars-native shared implementation

The target shared modules will use Polars for I/O, column selection, rolling features, joins, correlations, scaling, and output assembly. They should not import pandas after migration.

Alternatives considered:
- Commodity-only Polars path. Rejected because the user selected shared module migration.
- Dual pandas/Polars paths. Rejected because the user wants target files to stop importing pandas.

### Decision 2: No multiprocessing wrapper for time features

`time_operator/create_feature_multi_processing.py` and `multi_processing_util.py` will stop using the pandas multiprocessing structure. Polars expressions and rolling APIs will provide the feature computations, relying on Polars internal execution rather than Python process pools.

Alternatives considered:
- Keep multiprocessing and port each worker to Polars. Rejected because it preserves complexity and can offset Polars benefits.

### Decision 3: Compatibility over feature redesign

Output paths, file names, CLI parameters, column names, and window causality remain compatible. Numeric values may differ only by small floating point tolerances.

The migration must not introduce future leakage. Rolling features must use only current and historical rows according to the existing window semantics.

### Decision 4: ML boundary conversion is allowed without pandas

`catbooost.py` and `lasso_linear.py` may convert Polars columns to NumPy arrays or library-native data structures immediately before invoking CatBoost or sklearn. These files still must not import pandas.

## Risks / Trade-offs

- Reproducing all pandas rolling and correlation semantics exactly in Polars can be subtle. The implementation needs focused fixture tests for column names, row trimming, and tolerance-based numeric comparison.
- CatBoost and sklearn examples often use pandas, but this migration must keep the model boundary pandas-free.
- `feature_selection` has several historical scripts with weaker current test coverage; implementation should add regression tests before replacement.
- Polars `describe()` output differs from pandas. `scale_save.py` must explicitly preserve the expected `df_describe.csv` shape or define the accepted compatible shape in tests.

## Migration Plan

1. Build focused fixture tests for time feature, IC/rank IC, scale/save, and feature selection output contracts.
2. Replace `time_operator` internals with Polars helpers and keep CLI outputs compatible.
3. Replace `ic_correlation.py`, `rank_ic_correlation.py`, `cor_util.py`, and remove-duplicates logic with Polars/NumPy equivalents.
4. Replace ML feature-selection script data preparation with Polars and model-boundary NumPy conversion.
5. Replace `scale_save.py` with Polars scaling and describe output.
6. Run target import scan, commodity five-day `main.sh`, focused tests, and OpenSpec validation.

## Open Questions

None. The user confirmed shared-module migration, output compatibility with floating point tolerance, no multiprocessing wrapper, all `feature_selection` scripts in scope, pandas-free target files, no performance gate, and the layered Polars-native approach.
