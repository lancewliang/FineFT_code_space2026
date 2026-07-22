## 1. Implementation

- [ ] 1.1 Update low-level test-agent discovery tests and helper fixtures to model `valid/<contract>/<label>/df_*.feather`.
- [ ] 1.2 Refactor `test_agent_index.py` validation file discovery and aggregate result construction to emit pure `label`, aligned `contract`, and contract-relative `df_path` arrays.
- [ ] 1.3 Update aggregate CSV serialization and Chinese headers to include `合约` while preserving JSON array cells.
- [ ] 1.4 Add picker schema-validation and sample-equal scoring tests for the new cross-contract result schema and legacy schema rejection.
- [ ] 1.5 Refactor `FineFT_single_agent_with_different_position.py` to validate new schema, preserve current first-stage and final-stage selection logic, and fail fast on invalid labels or metrics.
- [ ] 1.6 Add `selection_manifest.json` output and enforce label-order model assembly for `model.pth`.
- [ ] 1.7 Update commodity low-level test and picker shell scripts so `fu` runs pass the required base path, experiment name, position choices, and label count.

## 2. Verification

- [ ] 2.1 Run `conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/analysis/test_pick_agent.py -q`.
- [ ] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`.
- [ ] 2.3 Run `openspec validate select-cross-contract-low-level-agents --strict`.

