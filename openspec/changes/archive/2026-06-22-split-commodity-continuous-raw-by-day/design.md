## Context

Commodity futures `CONTINUOUS_RAW` currently materializes one date-range CSV such as `fu_2023-01-01_2024-01-01.csv`. Long windows make that file large, and `downscale_continuous_by_trading_day.py` must read the entire file before it can split by `TradingDay`.

This change affects the data preprocessing pipeline only: commodity main-contract stitching, commodity continuous downscale entrypoint, shell orchestration, and commodity preprocessing documentation. It does not change FineFT environment behavior, training, feature semantics, order-book execution, fees, leverage, or model artifacts.

## Goals / Non-Goals

**Goals:**

- Make `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv` the only commodity continuous raw output.
- Make stitch and downscale CLIs date-range and directory based.
- Keep the existing left-closed/right-open date range semantics.
- Let missing source days or missing daily files be skipped with clear logs.
- Keep existing bad-data checks fail-fast.
- Preserve downstream downscale output folders and date-range driven cross-section/merge/concat/time-feature/feature-selection contracts.

**Non-Goals:**

- Do not change main-contract selection rules.
- Do not introduce a trading calendar.
- Do not keep backward compatibility for the old `--output <big.csv>` or `--input <big.csv>` semantics.
- Do not change Feather output names, columns, feature definitions, or environment-facing data contracts.

## Decisions

1. Daily `CONTINUOUS_RAW` files are the only raw continuous artifact.
   - Decision: write `PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv`, using ISO date filenames derived from `TradingDay`.
   - Rationale: downstream processing already runs by date range, and daily files bound IO and memory to one trading day.
   - Alternative considered: keep one large file and add chunked reading. Rejected because it preserves the oversized artifact and keeps downscale coupled to whole-range reads.

2. CLI migration is intentionally breaking.
   - Decision: `stitch_main_contract.py` uses `--output_dir --start_date --end_date`; `downscale_continuous_by_trading_day.py` uses `--input_dir --start_date --end_date`.
   - Rationale: retaining both single-file and directory modes would keep the old path alive and add compatibility branching for a format the user explicitly wants to remove.
   - Alternative considered: support both modes temporarily. Rejected by user decision.

3. Missing days are skipped, malformed data fails.
   - Decision: stitch skips dates with no source data and logs skipped dates; downscale warns and skips missing daily files. Existing malformed data cases still raise.
   - Rationale: absent non-trading days are normal without a trading calendar, but present corrupted files must not be silently ignored.
   - Alternative considered: fail on every missing calendar date. Rejected because no trading calendar is being added.

4. Downstream contracts remain date-range driven.
   - Decision: `fu_full_process.sh` passes the continuous raw directory plus date range to downscale, while cross-section, merge, concat, time feature, feature selection, and scale/save keep their existing date-range inputs and output paths.
   - Rationale: this localizes the data-volume fix to the continuous raw handoff and avoids unnecessary downstream churn.

## Risks / Trade-offs

- Breaking CLI call sites may fail if any untested script still passes `--output` or `--input` as a big file. Mitigation: update commodity shell entrypoints, CLI tests, and docs in the same change.
- Missing daily files may hide accidental upstream skips. Mitigation: log each skipped date and a skipped-date summary; keep malformed present files fail-fast.
- Date filename conversion can drift between `YYYYMMDD` and `YYYY-MM-DD`. Mitigation: centralize conversion in tests and assert exact `YYYY-MM-DD.csv` paths.
- Large raw source scans still load candidate contract files for the requested years. Mitigation: this change reduces the continuous handoff size; broader source-scan streaming is out of scope.

## Migration Plan

1. Add tests for daily stitch output, overwrite logging, directory-based downscale, missing daily file warning, and shell/docs command changes.
2. Update main-contract code and stitch CLI to write daily CSV files under `CONTINUOUS_RAW/{symbol}`.
3. Update downscale CLI to iterate the left-closed/right-open date range and read daily files.
4. Update commodity full-process shell and commodity preprocessing docs to use directory/date-range arguments.
5. Run focused commodity tests, shell syntax checks, OpenSpec validation, and diff checks.

Rollback is a code rollback of this change. No data migration is required because generated `CONTINUOUS_RAW` artifacts can be regenerated.

## Open Questions

None. The user confirmed daily files, breaking CLI migration, missing-day skip behavior, overwrite behavior, and no trading calendar.
