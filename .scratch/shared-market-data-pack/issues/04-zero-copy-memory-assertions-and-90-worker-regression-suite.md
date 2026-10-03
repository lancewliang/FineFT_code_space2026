# 04: Zero-copy Memory Assertions and 90-Worker Regression Suite

**What to build:** Build comprehensive unit and concurrency test suite in `FineFT/tests/rl/test_shared_data_manager.py` verifying zero-copy shared memory semantics, in-place reset invariants, fail-fast schema checks, and clean shutdown with zero disk artifacts. Run full diverse rollout and parallel training regression tests across 90 workers.

**Blocked by:** 01, 02, 03

**Status:** completed

- [x] Unit tests for `EnvTensorPack` and `SharedMarketDataPack`: verify all tensors reside in POSIX shared memory (`is_shared() is True`), NumPy array views share underlying data pointers (`np.shares_memory() is True`), and shape/dtypes match expected environment requirements.
- [x] Unit tests for fail-fast schema validation: verify missing limit columns raise `ValueError` when `enable_limit_reward=True`, and succeed with `None` when `enable_limit_reward=False`.
- [x] In-place reset invariant tests: assert that step transitions and reset returns on reused environments produce identical numerical outputs across consecutive tasks as newly constructed environments.
- [x] Concurrency and scale tests: execute diverse exploration with `SharedMarketDataPack` and `PersistentRolloutPool` across multiple workers (including 90-worker scale assertion), verifying no `.pkl` files created in `/dev/shm` or temp directory, and process memory stays bounded.
- [x] Run full regression test suite: `test_shared_model_manager.py`, `test_persistent_pool.py`, `test_parallel_weight_advantage_pretrain.py`, `test_parallel_diverse_cpu_exploration.py`, and `test_shared_data_manager.py`.
