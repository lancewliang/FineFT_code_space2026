# 03: Cross-Epoch Persistent Rollout Worker Pool Integration

**What to build:** Implement `PersistentRolloutPool` in `FineFT/RL/DiHFT/low_level/persistent_pool.py` as an explicit context manager and integrate it into `FineFT/RL/DiHFT/low_level/parallel_diverse_train.py:run_parallel_diverse_training`. Spawn workers once at the start of diverse training, persist them across epochs to reuse internal environment instances, trigger in-place weight sync after each epoch's training phase, actively shut down the pool early when exploration is permanently skipped (`skip_exploration` / buffer full), and wire watchdog cleanup with `try...finally` and `atexit`.

**Blocked by:** 01 (Concurrent Watchdog Shutdown and Queue Feeder Safety), 02 (Single-Copy Shared Inference Model and In-place Weight Hot Sync)

**Status:** completed

- [x] `PersistentRolloutPool` manages worker process lifecycles via a context manager (`__enter__` and `__exit__`), spawning child processes only once upon pool creation.
- [x] Worker processes execute an event loop responding to `ExploreTask` and `WorkerShutdownCommand`, returning transitions and maintaining environment state across tasks without process termination.
- [x] `run_parallel_diverse_training` wraps the outer epoch loop in `with PersistentRolloutPool(...) as pool:` and dispatches tasks to the persistent pool.
- [x] After each epoch's neural network update, the main process calls `pool.sync_model_weights(trainer.eval_net)` before the next epoch's rollout begins.
- [x] If `skip_exploration` is permanently activated (buffer capacity reached or consecutive no-new-experience threshold exceeded), `pool.shutdown()` is explicitly invoked to immediately free worker memory and CPU threads for downstream training.
- [x] `atexit.register` and `try...finally` ensure the pool is unconditionally closed on abnormal termination, unhandled exceptions, or `KeyboardInterrupt`.
- [x] Tests verify that worker PIDs remain identical across successive epochs and confirm early shutdown triggers when the buffer becomes full.
