# Close Issues: add-commodity-quote-depth-imbalance-features

## Verification Evidence

- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH="$PWD/FineFT:$PWD/data_preprocess:${PYTHONPATH:-}" pytest -q`
  - Result: `319 passed, 17 warnings in 35.24s`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest data_preprocess/tests/test_commodity_downscale.py -q`
  - Result: `48 passed in 1.37s`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile data_preprocess/operator_futures/commodity/downscale.py data_preprocess/tests/test_commodity_downscale.py`
  - Result: exit code `0`
- `openspec validate add-commodity-quote-depth-imbalance-features --strict`
  - Result: `Change 'add-commodity-quote-depth-imbalance-features' is valid`

## Default Pytest Note

Running bare `pytest -q` from the repository root currently fails during collection
before running tests because legacy RL test modules import package roots such as
`RL`, `env`, `model`, and `datahandler` without the project path on `PYTHONPATH`.
The full-suite command above adds `FineFT` and `data_preprocess` to `PYTHONPATH`,
matching prior project close records.

## Completeness

- `openspec/changes/add-commodity-quote-depth-imbalance-features/tasks.md` has all tasks checked.
- `openspec/changes/add-commodity-quote-depth-imbalance-features/plan-ready.md` has all task-level checkboxes checked.
- `docs/superpowers/plans/2026-07-23-add-commodity-quote-depth-imbalance-features.md` has all Step and Task checkboxes checked.

## Correctness

- `data_preprocess/operator_futures/commodity/downscale.py` implements `imbalance_1`, `imbalance_3`, and `imbalance_5` from quote depth volumes before right-closed target-window aggregation.
- The implementation preserves `imbalance_volume` compatibility and adds `std_imbalance_volume`.
- Zero or null denominator cases return `0.0`; non-finite quote volume input and missing depth volume columns fail fast.
- `data_preprocess/tests/test_commodity_downscale.py` covers the new output statistics, compatibility, neutral denominator handling, non-finite input, missing depth input, and continuous downscale safe-output behavior.

## Coherence

- No `design.md` exists for this change; implementation follows `proposal.md`, `specs/commodity-futures-support/spec.md`, and existing Polars downscale patterns.
- No CRITICAL, WARNING, or SUGGESTION issues remain for this change.

## Code Review Notes

- Medium: `data_preprocess/tests/test_commodity_downscale.py` uses the same per-level volumes for the 1/3/5 depth regression case, so the test does not distinguish `imbalance_3` and `imbalance_5` from `imbalance_1`. This is a test-strength issue, not a runtime bug.
- Low: `data_preprocess/operator_futures/commodity/downscale.py` uses `strict=False` casts for depth volumes. That is acceptable for the current numeric upstream contract, but it would silently coerce non-numeric strings to neutral values if such frames ever reached this function.
- Reviewer verdict: mergeable with no blocking runtime issues; the main follow-up would be to strengthen the depth-formula regression test.
