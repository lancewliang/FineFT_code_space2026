## Context

`data_preprocess/operator_futures` is the preprocessing layer that turns raw futures data into Feather files consumed by later FineFT data preparation and environments. It currently uses pandas for CSV/Feather IO, timestamp conversion, resampling, window aggregation, cross-section feature generation, feature selection inputs, scaling output assembly, and commodity futures main-contract processing.

The migration must improve preprocessing performance without changing the script-driven workflow or downstream data contracts. Existing scripts, CLI arguments, intermediate directories, Feather file names, timestamp semantics, and column order remain part of the public contract.

## Goals / Non-Goals

**Goals:**
- Make Polars the required DataFrame engine for core `data_preprocess/operator_futures` preprocessing.
- Preserve Binance futures preprocessing outputs for downscale, base features, cross-section features, time features, merge/concat, scale/save, and feature-selection inputs.
- Preserve commodity futures outputs for main-contract stitching, continuous downscale, commodity schema, and commodity-specific preprocessing branches.
- Keep the existing command-line and shell-script workflow stable.
- Validate output compatibility with existing small samples and tests.

**Non-Goals:**
- Do not migrate `FineFT/**` training, environment, or analysis code in this change.
- Do not create pandas legacy modules or a runtime pandas/Polars switch.
- Do not introduce a new benchmark framework or large test dataset.
- Do not redesign output schemas, directory layouts, file names, or downstream training contracts.
- Do not fix suspected historical pandas behavior silently; record the difference and wait for user decision.

## Decisions

### Decision 1: Polars is required for preprocessing

`data_preprocess/requirements.txt` will include `polars` as a required dependency. Core preprocessing scripts should import and operate on Polars DataFrames or LazyFrames. Temporary pandas conversion is allowed only at narrow test or third-party API boundaries where no practical Polars path exists, and each exception must be explicit in the implementation.

Alternatives considered:
- Optional Polars acceleration with pandas fallback. This lowers migration risk but keeps two behavior paths and makes strict output compatibility harder to reason about.
- Assume Polars is installed without updating requirements. This avoids dependency churn but makes environment reproduction unreliable.

### Decision 2: Preserve script and file contracts

The migration will keep existing module paths, shell entry points, CLI parameters, default path values, output directories, and Feather file names. The implementation changes the internal table engine, not the external workflow.

Alternatives considered:
- Rebuild the preprocessing pipeline around Polars lazy scans and fewer intermediate files. This could be faster, but it would change the current staged workflow and make strict compatibility harder.
- Split legacy and Polars outputs into separate directories. This would ease comparison but would break existing downstream assumptions.

### Decision 3: Compatibility before cleanup

Output compatibility is more important than refactoring style. Implementations must preserve timestamp alignment, duplicate timestamp `first` semantics, frequency-window labels, forward-fill behavior, column names, column order, row order, and key Arrow/Feather schema expectations where practical. Float comparisons use `rtol=1e-12, atol=1e-12`.

If pandas behavior appears wrong or Polars cannot exactly reproduce a dtype/schema detail, the implementation must stop at a documented compatibility note instead of changing behavior silently.

Alternatives considered:
- Accept Polars-native dtype and ordering changes. This would be simpler, but downstream code and saved artifacts may rely on current schema details.
- Fix suspected bugs while migrating. This risks mixing behavior changes with engine migration and makes regressions hard to isolate.

### Decision 4: Use focused existing-sample verification

The migration will rely on existing small commodity samples, existing `data_preprocess/tests`, and narrow smoke commands for Binance futures preprocessing paths. It will not add a benchmark script. Performance is documented by recording manual before/after command timings, or a precise reason why comparable timing is unavailable, for a representative preprocessing path. A fixed percentage improvement is not a blocking acceptance gate for this change.

Alternatives considered:
- Add a permanent benchmark suite. This is useful long term but outside the agreed scope.
- Use only unit tests. This would miss CLI/file contract regressions in the staged preprocessing workflow.

## Risks / Trade-offs

- Polars time-window semantics differ from pandas resampling. The implementation must explicitly verify window labels, closed-side behavior, duplicate timestamp handling, and missing-window filling for each migrated path.
- Strict Feather schema compatibility may require explicit casting before writes. Some Arrow dtype differences may be unavoidable and must be documented before user decision.
- `features_related`, `cross_section`, and `time_operator` contain dense pandas idioms. Rewriting them in one change is large but matches the chosen migration strategy.
- Commodity futures code is also under active evolution. This change stays separate from date-range support and must avoid reverting or overwriting those edits.
