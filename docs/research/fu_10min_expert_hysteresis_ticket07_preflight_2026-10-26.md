# FU 10-minute expert-hysteresis independent-evaluation preflight

**Date:** 2026-10-26
**Ticket:** `07-independent-retrospective-evaluation`
**Decision:** **BLOCKED — do not run calibration, replay, or report performance.**

This is a read-only preflight against the available FU 10-minute inputs. It is
not a backtest, calibration result, or claim about strategy performance.

## Scope and acceptance gate

Ticket 07 requires a chronological fit/selection → calibration → final split,
immutable model/Selection Manifest/reference-score provenance, a nine-dimensional
`expert_hysteresis` calibration, then a frozen-config replay through steps 8–10.
The implementation enforces those requirements:

- [`prepare_experiment`](../../FineFT/RL/DiHFT/high_level/frozen_experiment.py)
  requires a schema-2 independent experiment manifest, TradingDay-atomic source
  evidence, retrained models, non-overlapping periods, matching artifact hashes,
  and data with explicit timezone-aware `timestamp` and `trading_day` columns.
- [`validate_quantile_cache`](../../FineFT/RL/DiHFT/high_level/frozen_experiment.py)
  requires a cache manifest bound to the frozen raw-score source identity.
- Step 10 accepts only the step-8 published frozen configuration; it does not
  select parameters from test results.

The audit inspected the current implementation at commit
`94c7647a149155bc71d0e55fe7b470056483973d`, plus the untracked local FU data
and historical artifacts at their configured paths. No model, data, cache, or
result file was created, changed, or deleted.

## Evidence inventory

| Required evidence | Observed evidence | Result |
| --- | --- | --- |
| Immutable schema-2 expert experiment manifest with `experiment_id`, split identity, four periods, source-file hashes, scoring protocol, and search-space evidence | No `experiment_manifest.json` was found in either the primary checkout or this worktree. `dataset_manifest.json` is a legacy dataset inventory, not this manifest. | **Missing** |
| Chronological, TradingDay-atomic split | The runtime data contains 14 train, 12 valid, and 13 test Feather files (46,111 / 30,363 / 28,110 rows), but **all 39 lack `trading_day`**. Train ends `2024-07-22 23:00:00` while valid begins `2024-07-23 09:00:00`; this is the exact night-session boundary ADR-0060 requires to be established from explicit TradingDay evidence, not natural dates. | **Invalid for independent use** |
| Fixed, provenance-bound specialist/VAE/Selection Manifest inputs | The historical VAE weights, references, specialist models, and a two-dimensional manifest exist. However, the Selection Manifest is `schema_version: 1` and carries no schema-2 experiment identity, period evidence, or file hashes binding it to a corrected split. It therefore cannot establish immutable sources for this experiment. | **Present but unverified; not usable** |
| Raw quantile cache with source/scoring provenance | No `quantile_cache_manifest.json` was found. Therefore no cache can prove the required raw-score source identity, scoring protocol, contract coverage, bounds, or read-only reuse. | **Missing** |
| Independent nine-dimensional calibration | The only high-level study is historical `vae_risk_aware_routing` (400 complete trials, database hash `1dfc457f…2e90cc89`). Its CSV records `params_gating_strategy=absolute`; its parameter set has no `expert_switch_lead` or `expert_confirm_steps`. It is not an `expert_hysteresis` calibration and cannot be relabelled. | **Incompatible historical artifact** |
| Frozen best trial, step-8 publication, and step-9/10 receipts | No `selected_expert_hysteresis_config.json`, `frozen_config.json`, `frozen_evaluation_receipt.json`, or `selection_evidence.json` was found. | **Missing** |
| Comparable historical-control evidence | A legacy absolute-routing result directory exists, but lacks the frozen split/scoring/cache identities above. It may remain an archive, but is not yet a controlled comparison for this ticket. | **Not comparable** |

The observed legacy inventory fingerprints are retained here to make a later
re-audit unambiguous:

| Artifact | SHA-256 |
| --- | --- |
| `dataset/10min/fu/dataset_manifest.json` | `fd8458f1354e9d48690deda6edeba778727924547259becd8279fb0f21df8d77` |
| `analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_selection_manifest.json` | `fe3390a28cba2f255d8792f1ef9d6c19d2f86379c21db0dfecdd0a2533671ed2` |
| `result/DiHFT/high_level/fu/10min_parallel/vae_risk_aware_routing_optuna/optuna_study.db` | `1dfc457f93f7d1f3b14260d1c4472169fa4e85bbd0631d004a4145bd2e90cc89` |

## Why no evaluation was run

Running step 7 would require an `EXPERIMENT_MANIFEST`; steps 9 and 10 require a
frozen configuration. More importantly, fabricating either document from the
legacy inputs would conceal the missing TradingDay boundary and unproven model
provenance. That would violate the independent-evaluation requirement, so no
training, calibration, scoring, backtest, report table, or historical result
reinterpretation was performed.

## Conditions to unblock

1. Rebuild the FU split from explicit source `TradingDay` mapping, including the
   affected night session; preserve timezone-aware timestamps and add
   `trading_day` to every runtime Feather file.
2. Retrain the VAE and specialist inputs, repeat selection on that split, and
   produce a Selection Manifest whose referenced artifacts are hashable and
   immutable.
3. Supply a schema-2 independent experiment manifest that names the fit,
   selection, calibration, and final periods; proves the required chronological
   non-overlap; declares `trading_day_atomic: true` and
   `models_retrained_for_split: true`; and records the raw-scoring protocol.
4. Run step 7 in a new expert-hysteresis namespace only after that manifest
   validates. Freeze the selected complete nine-dimensional configuration, then
   publish it in step 8 and generate a provenance-bound raw quantile cache.
5. Replay that frozen configuration in steps 9 and 10. Only then produce the
   requested stability, confirmation-delay, rule-occupancy, turnover, friction,
   net-return, drawdown, and reproducibility report. A historical comparator
   must use the same declared data/model/scoring protocol, or its remaining
   differences must be reported explicitly.

Until those inputs exist, Ticket 07 remains an evidence blocker rather than an
incomplete performance result.
