---
status: accepted
---

# Commodity Preprocess Parallel Scaling and CSV Elimination

We eliminate redundant CSV disk writes in commodity continuous downscale and introduce a two-stage parallel extraction pipeline for main-contract stitching with strict Polars thread anti-contention guards (`POLARS_MAX_THREADS=1`), accelerating the preprocessing bottleneck from ~25 minutes to under 1 minute with zero backward-compatibility baggage.

## Context

Empirical audits of commodity futures preprocessing (`stitch_main_contract` and `downscale_continuous_by_trading_day`) revealed three severe computational and architectural inefficiencies:

1. **Unparallelized Single-Core CSV Ingestion in Stitching**:
   In `main_contract.py:build_main_contract_summary_model_for_date_range`, scanning ~8,000 raw contract CSVs (31GB) to compute daily trading volume and open interest was executed entirely in a single Python thread with full-table parsing (`pl.read_csv`), consuming ~19.6 minutes.
2. **Artificial Worker Bottleneck and Polars Thread Oversubscription in Downscaling**:
   In `fu_full_process.sh`, downscaling 2,875 contract-day tasks was hardcoded to `--max_workers 7`, utilizing only ~3.6% of available CPU cores on multi-core servers (192 logical cores). Furthermore, Polars spawned up to 192 Rayon worker threads per child process, causing thousands of concurrent OS threads to violently thrash CPU caches and fight over context switching.
3. **Redundant CSV Disk Output**:
   In `downscale_continuous_by_trading_day.py:_write_downscaled_day`, each contract-day wrote out 5 binary Feather IPC files alongside 5 plain-text CSV files. Downstream operators strictly consume `.feather` only; writing CSV incurred massive, unneeded float-to-ASCII serialization and I/O overhead.

## Decision

We establish the **High-Throughput Commodity Preprocessing Architecture** without backward-compatibility baggage:

### 1. Downscale Feather Purification (Eliminate CSV Output)
In `operator_futures/commodity/downscale_continuous_by_trading_day.py`:
- Completely remove `frame.write_csv(path / f"{output_name}.csv")`.
- Solely persist high-performance `.feather` binary IPC files.
- Update test assertions that previously inspected downscale CSV outputs.

### 2. Two-Stage Parallel Extraction for Main-Contract Stitching
In `operator_futures/commodity/main_contract.py` and `stitch_main_contract.py`:
- Stage 1: Fast serial header scan (`n_rows=1`) building the universe mapping `(InstrumentID, TradingDay)` and computing contract lifespans without reading full data bodies.
- Stage 2: Filter candidate contracts belonging to target date range `[start_date, end_date)` and target active months, then dispatch projected reading (`columns=["Volume", "OpenInterest"]`) across a `multiprocessing.Pool`.
- Reducer: Perform in-memory sub-second dictionary aggregation of monthly trading volume and main/sub role rankings on the main process.
- Expose `--max_workers` in CLI and support dynamic CPU concurrency.

### 3. Polars Thread Anti-Contention Guard (`POLARS_MAX_THREADS=1`)
In worker initialization (`configure_worker`) of all multiprocessing pools across `stitch_main_contract` and `downscale_continuous_by_trading_day`:
- Explicitly inject `os.environ["POLARS_MAX_THREADS"] = "1"`.
- Prevent Rayon thread pool multiplication, restricting task parallelism to the outer process pool and scaling throughput linearly with CPU cores.

### 4. Dynamic Concurrency Orchestration in Shell
In `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`:
- Remove hardcoded `--max_workers 7` in `run_commodity_downscale_continuous_by_trading_day`.
- Propagate `$max_processes` / `${MAX_PREPROCESS_WORKERS}` (defaulting to 32 workers) to both `run_commodity_stitch_main_contract` and `run_commodity_downscale_continuous_by_trading_day`.
- Export `POLARS_MAX_THREADS=1` prior to Python operator invocations.

## Consequences

- `stitch_main_contract`: Latency drops from ~19.6 minutes to **10~20 seconds** (50x~100x speedup).
- `downscale_continuous_by_trading_day`: Processing throughput increases from 9.7 tasks/s to **116.4 tasks/s**, completing all 2,875 contract-days in **~25 seconds** (12x speedup).
- Combined preprocessing wall-clock time drops from **~25 minutes to < 1 minute**.
- Zero backward-compatibility baggage: redundant CSV generation is completely removed, simplifying disk footprints.
