# Spec: Shared-Market-Data-Pack and In-place Environment Reset Architecture

- **Triage Label**: `ready-for-agent`
- **Related ADR**: `docs/adr/0040-shared-market-data-pack-and-in-place-environment-reset.md`
- **Target Subsystems**: Low-Level RL Diverse Training Engine (`parallel_diverse_train`), Trading Environment Core (`base_env`, `demo_env`), Persistent Rollout Pool (`persistent_pool`)

---

## Problem Statement

Following the successful implementation of `SharedInferenceManager` (which reduced policy network memory across 90 workers from 18GB to 200MB), the remaining major memory and execution bottleneck in Stage I Low-Level RL diverse training is the market data distribution mechanism (`train_df_cache`):

1. **Massive Market Data Physical Memory Duplication**:
   - `train_df_cache` contains 45 to 90 market segments (DataFrames), each containing tens of thousands of rows of 5-level order book and technical features (500MB to 2GB in memory).
   - Currently, the main process writes `train_df_cache` into a single pickle file in `/dev/shm`.
   - Each of the 90 child processes independently executes `pickle.load()` and instantiates a complete replica of the DataFrame dictionary in its own heap memory.
   - For a 500MB dataset, 90 worker processes consume $90 \times 500\text{MB} = 45\text{GB}$ of physical RAM solely to store identical tabular market data, creating severe OOM risks and heavy swap thrashing.
2. **Heavy Disk & IPC Serialization Overhead**:
   - Serializing and unpickling large Pandas DataFrames with complex indices and object columns burns significant CPU cycles and disk I/O on every training startup.
3. **Repeated Column Parsing and NumPy Allocation per Task**:
   - Upon receiving an `ExploreTask`, `DfRolloutWorkerRunner` accesses DataFrame columns by string name (`df["mark_price"].values`, `df[feature_list].values`, `df[ask_prices_names].values`).
   - Every task invocation incurs repeated Pandas column hash lookups, type conversions, and temporary NumPy array allocations.
4. **Continuous Environment Creation and Destruction**:
   - Because each task tests a distinct `initial_action` (producing a distinct `initial_state`), the current code instantiates a brand new `Demo_Env` for every single task and discards it at round completion.
   - This creates massive Python object churn and GC pressure across 90 worker processes.

---

## Solution

We architect and implement the **Shared-Market-Data-Pack and In-place Environment Reset Architecture** to eliminate tabular memory redundancy and environment recreation churn:

1. **Pre-extracted Numerical Env Tensor Pack (`EnvTensorPack`)**:
   - The main process parses raw DataFrames once at initialization, stripping Pandas index and column metadata into 9 contiguous PyTorch CPU numerical tensors:
     - `state_tensor`: `(L, feature_dim)`, `float32`
     - `ask_prices_tensor`: `(L, order_book_depth)`, `float32`
     - `bid_prices_tensor`: `(L, order_book_depth)`, `float32`
     - `ask_qtys_tensor`: `(L, order_book_depth)`, `float32`
     - `bid_qtys_tensor`: `(L, order_book_depth)`, `float32`
     - `markprice_tensor`: `(L,)`, `float32`
     - `timestamp_tensor`: `(L,)`, `int64`
     - `funding_rate_tensor`: `(L,)`, `float32`
     - `funding_timestamp_tensor`: `(L,)`, `int64`
     - Optional adaptive limit columns: `is_limit_up_tensor`, `is_limit_down_tensor`, `regime_grid_ids_tensor`.
   - Invokes `tensor.share_memory_()` on each tensor, mapping them directly to POSIX shared memory.
2. **Global Shared Market Data Registry (`SharedMarketDataPack`)**:
   - Packages all `EnvTensorPack` instances into a structured registry `SharedMarketDataPack: dict[int, EnvTensorPack]`.
   - Passes the single shared registry reference to all 90 child processes upon pool startup.
   - Child processes access any `df_index` zero-copy via `tensor.numpy()` views, reducing physical memory consumption from 45GB to a single 500MB footprint (98.8% memory savings).
3. **Persistent Trading Environment and In-place State Reset (`PersistentTradingEnv`)**:
   - `DfRolloutWorkerRunner` caches a single `Demo_Env` per `df_index`.
   - When a new `ExploreTask` arrives, the worker calculates `initial_state` and invokes `env.reset(initial_state=initial_state)` directly on the cached environment.
   - The underlying shared array memory pointers remain untouched; only episode cursors (`self.day = 0`, history lists, wallet/position variables) are reset.
4. **Deterministic Resource Cleanup**:
   - `SharedMarketDataPack` is held within `PersistentRolloutPool`. When the pool closes (at training completion or upon replay buffer saturation), the shared tensors are unreferenced and cleaned up with zero disk file remnants in `/dev/shm`.

---

## User Stories

1. As a quantitative RL researcher, I want 90 rollout worker processes to share market data from a single shared memory repository, so that system RAM usage drops from 45GB to 500MB without OOM risks.
2. As a performance engineer, I want market data to be pre-converted into contiguous numerical tensors, so that worker processes never execute Pandas string indexing or `pickle.load()` on startup.
3. As a machine learning practitioner, I want worker processes to cache trading environment instances and reset them in-place with `initial_state`, so that exploration rounds avoid recreating environment objects and allocating temporary NumPy arrays.
4. As an infrastructure engineer, I want shared market data tensors to be mapped via PyTorch `share_memory_()`, so that the system maintains a unified POSIX shared memory architecture identical to policy network sharing.
5. As a core maintainer, I want adaptive handling of price limit and regime grid columns, so that datasets lacking limit columns fail fast only when `enable_limit_reward=True`.
6. As a DevOps engineer, I want market data sharing to clean up deterministically when the worker pool exits, so that no orphaned `.pkl` files remain in `/dev/shm`.

---

## Architecture and Contracts

### New and Modified Modules

- **`FineFT/RL/DiHFT/low_level/shared_data_manager.py` (New Module)**:
  - Defines `EnvTensorPack` dataclass containing contiguous shared CPU tensors.
  - Defines `SharedMarketDataPack` managing the collection of `EnvTensorPack` by `df_index`.
  - Implements `SharedMarketDataPack.from_dataframes(train_df_cache, env_kwargs)`.
- **`FineFT/env/env_class/base_env.py` (Modified Module)**:
  - Extends `Base_Env.reset(self, initial_state=None)` to accept dynamic `initial_state` overrides for in-place re-initialization.
- **`FineFT/RL/DiHFT/low_level/parallel_diverse_train.py` (Modified Module)**:
  - Updates `start_parallel_workers` to pass `shared_market_data` into `worker_config` and deprecates `shm_df_cache_path` pickling.
  - Updates `DfRolloutWorkerRunner` to build environments from `shared_market_data` and cache `Demo_Env` singletons.
  - Updates `DfRolloutWorkerRunner.run_task` to execute in-place environment resets.
- **`FineFT/RL/DiHFT/low_level/persistent_pool.py` (Modified Module)**:
  - Coordinates `SharedMarketDataPack` lifecycle alongside `SharedInferenceManager`.

---

## Testing Decisions

1. **Zero-Copy Verification**: Verify that NumPy arrays inside `Demo_Env` in worker processes point to the same memory addresses (`data_ptr`) as the parent shared tensors.
2. **In-place Reset Behavioral Invariant**: Verify that calling `env.reset(initial_state=...)` produces exact numerical matches to newly instantiated `create_demo_env(...)`.
3. **High-Concurrency Scale Test**: Spin up 90 workers with shared market data packs, assert zero `pickle.load` disk activity, and verify total physical memory remains strictly bounded.
4. **Adaptive Column Validation**: Verify Fail-fast behavior when `enable_limit_reward=True` and limit columns are absent, while allowing graceful default behavior when disabled.

---

## Out of Scope

- Modifying pretrain Q-table diagnostics data loading (kept on standard DataFrames).
- Modifying VAE feature calculation pipelines.
