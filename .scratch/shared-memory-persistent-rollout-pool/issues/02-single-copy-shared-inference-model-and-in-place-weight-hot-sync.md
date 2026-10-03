# 02: Single-Copy Shared Inference Model and In-place Weight Hot Sync

**What to build:** Create `SharedInferenceManager` in `FineFT/RL/DiHFT/low_level/shared_model_manager.py` to manage a CPU-based `ensemble_Qnet` residing in POSIX shared memory via PyTorch `model.share_memory_()`, enforcing the `file_system` multiprocessing sharing strategy and bumping system file descriptor limits. Update `df_rollout_worker` and `DfRolloutWorkerRunner` to bind `worker_config["shared_model"]` directly without re-instantiating models or loading pickled disk states, and implement `sync_weights_from_gpu` to in-place copy updated GPU weights to shared memory in <5ms.

**Blocked by:** None (can start immediately)

**Status:** completed

- [x] `SharedInferenceManager` configures `torch.multiprocessing.set_sharing_strategy('file_system')` and raises `RLIMIT_NOFILE` soft and hard limits to 65536.
- [x] `SharedInferenceManager` instantiates a CPU `ensemble_Qnet`, places it in `eval()` mode, and invokes `share_memory_()`, guaranteeing `param.is_shared() is True` for all parameters and buffers.
- [x] `DfRolloutWorkerRunner` accepts `worker_config["shared_model"]`, assigns `self.model = shared_model`, and skips model creation and disk unpickling.
- [x] Forward action selection in `DfRolloutWorkerRunner` executes under `torch.no_grad()` and `self.model.eval()`, ensuring read-only thread-safe evaluation.
- [x] `SharedInferenceManager.sync_weights_from_gpu` executes in-place parameter and buffer updates (`param.copy_()`) from GPU to the shared CPU tensors.
- [x] Unit tests verify zero-copy memory addresses between parent and child processes, test in-place weight synchronization visibility across processes, and assert no `Errno 24 Too many open files` errors under high simulated concurrency.
