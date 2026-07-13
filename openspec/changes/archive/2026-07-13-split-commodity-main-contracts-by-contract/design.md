## Context

Commodity futures preprocessing currently materializes a daily continuous main-contract raw file under `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv`. That artifact still represents a stitched sequence: each trading day has one selected main contract, and downstream factor files are written under `{symbol}/{target_freq}`. This mixes different real contracts into one symbol-level factor series.

The requested behavior is contract-scoped. Main-contract discovery should identify all contracts that qualify as monthly leaders, write a summary index, and let downstream preprocessing generate independent factor artifacts for each selected contract.

This change affects commodity preprocessing only: main-contract discovery, commodity downscale, shared operator-futures feature scripts when called with a commodity contract, commodity shell orchestration, validation scripts, tests, and documentation. It does not change feature formulas, trading sessions, order-book depth, reward/execution manifests, FineFT environments, or model training.

## Goals / Non-Goals

**Goals:**

- Replace continuous main-contract daily CSV output with `CONTINUOUS_RAW/{symbol}/main_contract_summary.json`.
- Select main contracts by natural month: sum each contract's daily `Volume.max - Volume.min` and choose the top 2 contracts per month, then union with contracts that have at least 10 days above the configured daily-volume threshold in that month.
- Treat selected contracts as a set, then clip each contract's trading-day window from its first selected month through the tenth trading day before its last trading day in the requested date range.
- Generate downscale and downstream factors under `{symbol}/{contract}/{target_freq}` while preserving current daily versus date-range file granularity.
- Update all commodity shell entrypoints and validation scripts to understand the contract dimension.
- Generate a symbol-level union of selected state features across all selected contracts for downstream single-model training.
- Keep legacy shared-script behavior when `--contract` is not provided.

**Non-Goals:**

- Do not introduce back-adjustment, spread adjustment, or price continuity across contracts.
- Do not change commodity feature formulas, depth=5 behavior, quote gap checks, target definitions, or scale/save manifests.
- Do not change FineFT runtime, training, model artifacts, fees, leverage, or environment semantics.
- Do not convert date-range outputs such as concat, time feature, all feature, IC result, or scale save into daily files.
- Do not add a trading calendar.

## Decisions

1. Summary JSON is the main-contract handoff.
   - Decision: `stitch_main_contract.py` writes `PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/{symbol}/main_contract_summary.json`.
   - Rationale: JSON can represent contract-level metadata and per-trading-day source file references without duplicating raw CSV data.
   - Alternative considered: copy raw CSV files to `CONTINUOUS_RAW/{symbol}/{contract}/{date}.csv`. Rejected because the user explicitly preferred summary-driven raw source lookup.

2. Monthly selection is deterministic and combines two inclusion rules.
   - Decision: monthly volume is `sum(daily Volume.max - daily Volume.min)`. Each natural month selects the top 2 contracts by monthly volume, breaking ties by contract name ascending. It also includes any contract with at least 10 actual trading days in that month where daily volume is strictly greater than that symbol's configured high-volume threshold.
   - Rationale: top-2 preserves the original requested rank rule, while the threshold-day rule captures contracts with sustained high liquidity that may fall outside monthly top 2.
   - Interface: the two rules are combined as a set union. `fu` uses `main_contract_daily_volume_threshold=15000`; other symbols must read their threshold from commodity config before using this rule.

3. Selected contracts use a clipped actual-source trading window.
   - Decision: if a contract is selected in any month, the contract enters the summary set. Its date-range-filtered source days are then clipped to actual `TradingDay` values that are on or after the first calendar day of the earliest selected month and on or before the cutoff trading day that is 10 contract trading days before that contract's max `TradingDay` within the requested date range. `start_trading_day` and `end_trading_day` are the min/max retained actual `TradingDay` values, and `trading_day_count == len(trading_days)`.
   - Rationale: the user refined the valid per-contract period: start when the contract first becomes a selected main contract month, and avoid the last 10 trading days before the contract's final trading day.

4. The contract dimension is optional in shared Python scripts.
   - Decision: downstream scripts add `--contract`. When present, path resolution uses `{symbol}/{contract}/{target_freq}`; when absent, existing `{symbol}/{target_freq}` paths remain.
   - Rationale: commodity full process needs contract-scoped paths, but shared operator-futures scripts may still be used by non-commodity or legacy callers.

5. File granularity stays unchanged.
   - Decision: downscale, cross-section, and merge continue writing daily files. Concat, time feature, all feature, IC result, and scale save continue writing date-range files or directories.
   - Rationale: this limits the change to the contract dimension and avoids unnecessary churn in downstream readers.

6. Commodity shell scripts are in scope.
   - Decision: `fu_full_process.sh`, `main_fu.sh`, `main_al.sh`, `commodity_process.sh`, `validate_features.sh`, and `flatten_aluminum_raw_csv.sh` are either enhanced or explicitly verified as unaffected.
   - Rationale: the end-to-end CLI contract is part of the user-facing workflow, and single-symbol assumptions in shell scripts can break multi-contract output.

7. Summary construction uses typed model beans behind the JSON handoff.
   - Decision: `main_contract.py` constructs summary data through a top-level `MainContractSummary` model containing `list[MainContractSummaryContract]`, and each contract contains typed trading-day entries. Serialization to `main_contract_summary.json` goes through the model's `to_dict()`/JSON conversion instead of hand-built nested dict literals.
   - Rationale: the summary schema has enough nested structure that handwritten dict/list construction makes field ownership unclear and spreads serialization details into selection logic.
   - Interface: external JSON fields and values stay unchanged. The model is an internal construction and serialization interface for locality; callers that consume `main_contract_summary.json` do not need to know the model classes.

8. Summary reading uses the same typed model beans.
   - Decision: Python readers of `main_contract_summary.json` use `MainContractSummary.from_dict(...)` or `load_main_contract_summary(...) -> MainContractSummary` before consuming contracts and trading days. Downscale and validation helpers should iterate `MainContractSummary.contracts` and `MainContractSummaryContract.trading_days` instead of indexing nested dicts.
   - Rationale: creation and consumption share one model interface, keeping schema validation, derived fields, and compatibility rules local to `main_contract.py`.
   - Interface: external JSON, CLI arguments, output paths, and shell behavior stay unchanged. The shell may still invoke a small Python snippet, but that snippet should parse through the model rather than `json.loads(...).get("contracts", [])`.

9. Trading-day summary entries include daily volume.
   - Decision: `MainContractSummaryTradingDay` carries `daily_volume`, serialized as `contracts[].trading_days[].daily_volume` in `main_contract_summary.json`.
   - Rationale: daily volume is already computed for monthly top-2 selection. Persisting it in the day-level summary avoids forcing readers to reopen each raw CSV only to inspect the same value.
   - Interface: `daily_volume` is a required numeric field on each summary trading-day entry and equals that contract source file's `Volume.max - Volume.min`.

10. Contract last-trading-day clipping uses raw contract trading-day order.
   - Decision: "last trading day minus 10 trading days" is computed from that contract's actual `TradingDay` sequence after the requested date-range scan, not from a separate exchange calendar or from the contract's full raw lifecycle outside the requested range. The retained end cutoff is inclusive; the final 10 in-range trading days are excluded from summary `trading_days`.
   - Rationale: the current pipeline has no trading calendar and already relies on actual source files as the authoritative available trading days.
   - Interface: if the first-selected-month start boundary and the 10-trading-day end boundary leave no retained trading days for a selected contract, summary generation fails fast instead of emitting an empty contract.

11. High-volume-day thresholds live in commodity configuration.
   - Decision: commodity config owns `main_contract_daily_volume_threshold` per symbol. The `fu` threshold is `15000`, representing 1.5 万 in the same units as `Volume.max - Volume.min`.
   - Rationale: thresholds are product-specific and should not be hard-coded inside summary selection logic.
   - Interface: the high-volume-day rule uses strict greater-than (`daily_volume > threshold`) and counts any 10 actual trading days in the month, not necessarily consecutive days.

12. Cross-contract training uses a symbol-level state-feature union.
   - Decision: after all contract-scoped `scale_save` outputs exist, the commodity full process writes `PREPROCESS_DATASET/commodity-futures/FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy` plus `feature_union_manifest.json`. The union is built from each summary contract's `SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/state_features.npy`.
   - Rationale: feature selection runs per contract, so selected state features can differ. The selected contracts train one shared model, so downstream training needs one stable feature set for the whole symbol/date-range run.
   - Interface: the union preserves deterministic order by iterating contracts in summary order and features in each contract's saved order, keeping only the first occurrence of duplicates. Reward/execution columns remain governed by the commodity schema and are not duplicated into the state-feature union. Missing per-contract `state_features.npy` is a fail-fast error.

## Risks / Trade-offs

- Adding `--contract` to shared scripts touches several path-building modules. Mitigation: keep the argument optional and cover old and new path shapes with focused tests.
- Summary source paths can become stale if raw data moves after summary generation. Mitigation: downscale fails fast when a listed `source_file` is missing.
- Shell loops over multiple contracts may produce overlapping logs if contract is omitted from log names. Mitigation: include contract in step logs, skip messages, and existence checks.
- The new main-contract rule intentionally replaces the prior previous-day continuous selection rule. Mitigation: encode the new rule in OpenSpec and tests so the breaking behavior is explicit.
- The feature union may contain state features that were selected by one contract but not another. Mitigation: the union artifact is explicit metadata for downstream single-model training; it does not silently merge contract rows or overwrite single-contract scale outputs.

## Migration Plan

1. Add failing tests for summary generation, monthly top-2 selection, summary validation, and no continuous daily CSV output.
2. Implement summary generation in `main_contract.py` and update `stitch_main_contract.py`.
3. Update downscale to read summary source files and write contract-scoped daily outputs.
4. Add optional `--contract` path resolution to downstream Python feature scripts while preserving legacy path behavior.
5. Update commodity shell scripts and validation/docs for summary-driven multi-contract processing.
6. Run focused pytest suites, shell syntax checks, OpenSpec validation, and diff checks.

Rollback is a code rollback plus regeneration of preprocessing artifacts. No data migration is required because generated outputs can be rebuilt from raw source files.

## Open Questions

None. The user confirmed natural-month selection, actual raw-data start/end dates, actual trading-day-only factor generation, JSON summary structure with `trading_day_count`, contract-scoped paths, unchanged file granularity, and all commodity shell scripts in scope.
