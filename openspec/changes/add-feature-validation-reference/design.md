## Context

The operator-futures preprocessing pipeline has been migrated toward Polars, while the previous workspace at `/home/lanceliang/opt/aiwork/FineFT_code_space2026_2/data_preprocess/operator_futures` still contains pandas implementations that are useful as a formula reference. The project also has feature-output documentation under `docs/data/*.md` that describes expected columns for data preprocess artifacts.

The new validation module should be independent from production preprocessing. It should help maintainers verify that generated feature artifacts have the expected columns and that sampled feature values match the pandas reference formulas.

## Goals / Non-Goals

**Goals:**
- Add an independent commodity validation shell entrypoint.
- Add a validation-only Python module under `operator_futures`.
- Copy the required pandas reference implementation files into a validation-only namespace.
- Validate fixed expected column lists derived from `docs/data/*.md`.
- Recompute reference values with pandas for intermediate and final artifacts, then compare sampled rows by timestamp with `abs_diff <= 1e-9`.
- Generate Markdown and JSON reconciliation reports.

**Non-Goals:**
- Do not modify production Polars preprocessing logic.
- Do not import production Polars implementation as the reference calculator.
- Do not hook the validator into `main.sh`.
- Do not dynamically parse Markdown docs at runtime.
- Do not validate from raw CSV in the first version.
- Do not copy model-training or optional feature-selection strategy modules unless later requested.

## Decisions

1. Use a separate shell entrypoint.
   - `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh` keeps the validation workflow discoverable next to the commodity preprocess scripts.
   - It remains separate from `main.sh` so preprocess runtime and failure semantics do not change.

2. Keep pandas reference code in a validation-only namespace.
   - Copy required source files into `data_preprocess/operator_futures/feature_validation/pandas_reference/`.
   - Preserve the original directory shape where practical to make diffs against `FineFT_code_space2026_2` straightforward.
   - Production preprocessing modules must not import this reference namespace.

3. Use fixed expected-column code instead of runtime docs parsing.
   - `expected_columns.py` encodes the docs-derived lists explicitly.
   - Docs changes require code review updates to the expected list and validators.

4. Validate by stage.
   - Validators cover `cross_section`, `merge_concat`, `time_feature`, `merge_clean`, `ic_correlation`, and `scale_save`.
   - Each stage defines its input paths, actual output paths, pandas reference function, expected columns, sampling, and comparison settings.

5. Prefer report completeness over early abort.
   - The CLI should try to run all independent stage validations and write a complete report.
   - It still returns non-zero when validation failures or errors exist, so automation can detect failures.

## Risks / Trade-offs

- The pandas reference may differ from current commodity-specific semantics. The report must distinguish mismatch, error, and unverified columns instead of hiding gaps.
- Copying pandas code creates duplicate logic, but it is intentionally isolated as a reference implementation and not a production fallback.
- Full-factor validation can be expensive. Sampling rows limits runtime while still catching formula drift on representative timestamps.
- Fixed expected columns avoid fragile Markdown parsing but require explicit maintenance when docs change.
