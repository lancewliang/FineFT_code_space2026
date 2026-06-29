## 1. Reference layout and entrypoint

- [x] 1.0 Complete reference layout and entrypoint. <!-- implemented: validation entrypoint and validation package skeleton -->
- [x] 1.1 Add the commodity `validate_features.sh` shell entrypoint without changing `main.sh`. <!-- 已实现: independent commodity validation entrypoint -->
- [x] 1.2 Create the `feature_validation` Python package and CLI skeleton. <!-- 已实现: feature_validation package and CLI -->
- [x] 1.3 Copy the required pandas reference modules from `FineFT_code_space2026_2` into `feature_validation/pandas_reference/`. <!-- 已实现: validation-only pandas reference tree -->

## 2. Validation rules and comparison engine

- [x] 2.0 Complete validation rules and comparison engine. <!-- implemented: compare/report/core validation modules -->
- [x] 2.1 Add fixed expected-column definitions derived from `docs/data/*.md`. <!-- 已实现: docs-derived constants -->
- [x] 2.2 Add stage validators for `cross_section`, `merge_concat`, `time_feature`, `merge_clean`, `ic_correlation`, and `scale_save`. <!-- 已实现: stage validators wired to reference adapters -->
- [x] 2.3 Add timestamp-aligned sampled row comparison with `abs_diff <= 1e-9`. <!-- 已实现: compare_frames tolerance -->
- [x] 2.4 Add Markdown and JSON report generation plus CLI exit-code handling. <!-- 已实现: report writers and exit codes -->

## 3. Verification

- [x] 3.0 Complete verification. <!-- implemented: validation tests and smoke coverage -->
- [x] 3.1 Add unit tests for expected columns, comparator behavior, and report generation. <!-- 已实现: compare/report unit tests -->
- [x] 3.2 Add reference import/adapter tests for copied pandas modules. <!-- 已实现: reference namespace import smoke -->
- [x] 3.3 Add an integration smoke test for `validate_features.sh` over the five-day commodity sample output. <!-- 已实现: shell entrypoint smoke -->
- [x] 3.4 Run focused tests and OpenSpec strict validation. <!-- 已实现: pytest + openspec validate -->
