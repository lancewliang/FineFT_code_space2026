# 01: Concurrent Watchdog Shutdown and Queue Feeder Safety

**What to build:** Re-architect worker process shutdown in `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py` (`shutdown_workers`) and `FineFT/RL/DiHFT/low_level/parallel_diverse_train.py` (`shutdown_exploration_workers`). Replace sequential `for p in processes: p.join(timeout=10)` with a concurrent global-deadline watchdog polling all processes, coupled with `cancel_join_thread()` on communication queues to prevent Python background feeder thread lockups. Ensure process teardown is bounded within 12 seconds total regardless of concurrency scale or individual worker stalls.

**Blocked by:** None (can start immediately)

**Status:** completed

- [x] All unique input and result queues invoke `cancel_join_thread()` before waiting on child processes, eliminating background thread pipe-drain hangs.
- [x] Shutdown logic broadcasts `ShutdownWorker` to all unique queues in parallel.
- [x] Process joining runs under a single global deadline (default 10.0 seconds) that concurrently polls `all(not p.is_alive() for p in processes)`.
- [x] Any child processes remaining alive after the global deadline receive batch `terminate()` calls.
- [x] A secondary grace window (2.0 seconds) allows terminated processes to exit; lingering processes receive `SIGKILL` cleanup.
- [x] Unit tests in `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py` verify that shutdown completes within the global deadline even when multiple mock workers fail to exit gracefully, proving no sequential timeout accumulation.
