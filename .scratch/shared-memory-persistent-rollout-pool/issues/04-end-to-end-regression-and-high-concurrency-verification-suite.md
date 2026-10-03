# 04: End-to-End Regression and High-Concurrency Verification Suite

**What to build:** Adapt existing unit and integration test fixtures in `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py` to conform to the persistent worker pool lifecycle and shared memory architecture. Add high-concurrency simulation tests (verifying up to 90 workers with dummy trading environments) to prove zero file descriptor leakage, verify total model memory consumption stays strictly bounded, and prune obsolete legacy helpers (`make_cpu_state_dict`, disk-based model pickling).

**Blocked by:** 03 (Cross-Epoch Persistent Rollout Worker Pool Integration)

**Status:** ready-for-agent

- [x] All existing test cases in `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py` are updated to match the new pool and shared model signatures, maintaining a 100% pass rate.
- [x] A dedicated high-concurrency test spins up 90 simulated worker processes under `file_system` sharing strategy, verifying no `OSError: [Errno 24] Too many open files` is raised.
- [x] Memory assertions verify that shared model parameters across 90 simulated workers share physical tensor storage and do not duplicate memory.
- [x] Legacy dead code relating to disk-based state-dict pickling in `/dev/shm` (`make_cpu_state_dict`, `shm_model_path`) is safely pruned from `parallel_diverse_train.py`.
- [x] Full RL unit test suite (`pytest FineFT/tests/rl/`) passes cleanly without deprecation warnings or dangling process warnings.
