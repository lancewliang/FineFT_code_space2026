# Close Issues: add-commodity-contract-unit

## Verification Evidence

- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=data_preprocess pytest data_preprocess/tests -q`
  - Result: `49 passed in 1.71s`
- `openspec validate add-commodity-contract-unit --strict`
  - Result: `Change 'add-commodity-contract-unit' is valid`
- `git diff --check`
  - Result: exit code 0, no output

## Code Review

Final review found no Critical or Important issues.

### Minor

- `proposal.md` says missing `contract_unit` should raise a clear config error. The implementation makes `contract_unit` a required dataclass field, so omission fails at construction time with Python's missing-argument `TypeError`; tests explicitly cover non-positive values but do not add a separate missing-field assertion. This does not block archive because runtime configuration cannot omit the field in the checked-in `fu` config, and the requirement is enforced by the dataclass constructor.

## Spec Consistency

- Completeness: all `tasks.md`, `plan-ready.md`, and superpowers plan checkboxes are complete.
- Correctness: `CommodityConfig.contract_unit`, `fu=10`, `second_avg_price`, `vwap`, and raw `tradeval` behavior match the delta spec.
- Coherence: no `design.md` exists for this small change; implementation follows the proposal and existing commodity preprocessing patterns.
