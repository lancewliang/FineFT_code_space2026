# Design: refactor-rl-diagnostics-dataclasses

## Context

The target modules already isolate much of the Stage I diagnostics flow, but they still exchange many anonymous dict payloads across functions, tests, and multiprocessing queues. The refactor should make those data contracts explicit without changing training behavior or external diagnostics files.

## Decisions

- Keep dataclasses local to the modules that own the contracts. This avoids a shared model module until these objects are reused outside the current low-level diagnostics boundary.
- Convert business records and cross-boundary payloads to dataclasses: diagnostics summaries, qtable sample records, worker messages/results, rollout metrics, and round summaries.
- Keep cache/map containers as dictionaries where they are the natural indexing structure, such as `dict[int, pd.DataFrame]`, `dict[int, q_table]`, and `dict[SamplePlanItem, list[int]]`.
- Provide `to_dict()` only at compatibility boundaries: JSON/CSV writing, log formatting where needed, and tests that compare against legacy dict shapes.
- Define multiprocessing payload dataclasses at module top level so Python's spawn-based multiprocessing can pickle them.

## Compatibility

`manifest.json` and `df_*_initial_action_*.csv` are external artifacts and must keep the current field structure. Existing readable CSV cache behavior remains: matching manifest and valid CSV files are reused; missing or malformed cache files trigger recomputation. Worker failures remain fail-fast through `RuntimeError`.

## Risk

The main risk is changing dict payloads that multiprocessing workers or tests still expect. The implementation should proceed test-first, first making object expectations explicit, then converting one module boundary at a time. Focused verification should cover direct helper tests and Python compilation before any broader training smoke run.
