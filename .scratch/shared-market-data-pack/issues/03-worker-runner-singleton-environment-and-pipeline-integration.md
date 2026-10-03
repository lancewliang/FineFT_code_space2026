# 03: Worker Runner Singleton Environment and Pipeline Integration

**What to build:** Integrate `SharedMarketDataPack` and in-place environment reset into `PersistentRolloutPool` and `DfRolloutWorkerRunner`. Remove `/dev/shm` pickle file serialization (`shm_df_cache_path`) completely. In `DfRolloutWorkerRunner`, cache `Demo_Env` singletons keyed by `df_index` and invoke `env.reset(initial_state=initial_state)` in-place for every `ExploreTask`, eliminating environment reconstruction and DataFrame slicing per task.

**Blocked by:** 01, 02

**Status:** completed

- [x] Update `PersistentRolloutPool` in `FineFT/RL/DiHFT/low_level/persistent_pool.py` to accept `shared_market_data: SharedMarketDataPack | None` or construct it from `train_df_cache`, and pass it to `start_parallel_workers`.
- [x] Remove `pickle.dump(train_df_cache, ...)` and `shm_df_cache_path` from `start_parallel_workers` and `shutdown_exploration_workers` in `FineFT/RL/DiHFT/low_level/parallel_diverse_train.py`.
- [x] Update `DfRolloutWorkerRunner.__init__` to receive `shared_market_data: SharedMarketDataPack` from `worker_config` and initialize an empty `self.cached_envs: dict[int, Demo_Env] = {}`.
- [x] Update `DfRolloutWorkerRunner.run_task(task: ExploreTask)`: retrieve or lazily initialize the singleton `Demo_Env` for `task.df_index` via `create_demo_env_from_pack`, calculate `initial_state` using `pack.markprice_tensor[0]`, and invoke `env.reset(initial_state=initial_state)` directly on the cached instance.
- [x] Update `run_epoch_exploration` and `run_parallel_diverse_training` to support passing and reusing `shared_market_data`.
