# close issues: add-commodity-quote-ofi-features

## Verification Evidence

- `openspec validate add-commodity-quote-ofi-features --strict`: passed with `Change 'add-commodity-quote-ofi-features' is valid`.
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests -q`: passed with `210 passed, 3 warnings`.
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py`: passed with exit code 0.
- Checkbox consistency: `tasks.md`, `plan-ready.md`, and `docs/superpowers/plans/2026-07-23-add-commodity-quote-ofi-features.md` have no remaining `- [ ]` items.

## Completeness

- All OpenSpec tasks `1.1` through `1.11` are checked `[x]`.
- Implementation evidence exists in `data_preprocess/operator_futures/commodity/downscale.py`.
- Test evidence exists in `data_preprocess/tests/test_commodity_downscale.py`.
- Full `data_preprocess/tests` suite passed.

## Correctness

- Five-depth OFI computes `ofi_bid1..5`, `ofi_ask1..5`, `ofi_bid`, `ofi_ask`, and `ofi`.
- Row-window aggregation uses fixed `window_rows=12` by default and keeps tail groups.
- Boundary comparison remains continuous across row-window boundaries.
- Normalized OFI outputs `ofi_bid_norm`, `ofi_ask_norm`, and `ofi_norm` using row-window five-depth volume denominators.
- Bad OFI inputs fail fast for missing columns, null values, NaN, `inf`, `-inf`, empty input, and invalid `window_rows`.

## Coherence

- No `design.md` exists; implementation follows `proposal.md` and the delta spec.
- OFI remains independent from `downscale_quote_features()` and does not alter existing time-window quote feature semantics.

## Final Code Review

- Review performed before archive.
- Scope reviewed: `data_preprocess/operator_futures/commodity/downscale.py`, `data_preprocess/tests/test_commodity_downscale.py`, and the OpenSpec change artifacts.
- Findings: no Critical or Important issues found.
- Notes: OFI remains isolated from the existing time-window quote downscale path; tests cover raw OFI math, row-window aggregation, cross-window comparison, normalized OFI, zero denominators, missing columns, nulls, NaN, `inf`, `-inf`, empty input, and invalid `window_rows`.

## Non-blocking Observations

- Full `data_preprocess/tests` emitted 3 warnings unrelated to the OFI path:
  - pandas `DataFrame.fillna(method=...)` deprecation warning from `feature_validation/reference_adapters.py`.
  - two numpy RuntimeWarnings from `test_ic_and_scale_reference_use_commodity_reward_schema`.
