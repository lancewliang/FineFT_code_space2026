# Spec: Shared-Memory Inference and Persistent Rollout Worker Pool Architecture

- **Triage Label**: `ready-for-agent`
- **Related Research**: `docs/research/2026-10-03-persistent-workers-and-shared-memory-architecture-research.md`
- **Related ADR**: `docs/adr/0039-shared-memory-inference-and-persistent-worker-pool.md`
- **Target Subsystems**: Low-Level RL Diverse Training Engine (`parallel_diverse_train`), Parallel Pretraining Infrastructure (`parallel_weight_advantage_pretrain`), Multiprocessing Rollout Pool

---

## Problem Statement

During Stage I Low-Level Reinforcement Learning diverse training and pretraining exploration, the system leverages multi-process parallel rollout workers to sample transitions across diverse market dynamics and initial action combinations. When scaling up to massive worker concurrency (e.g., 90 worker processes), the existing architecture exhibits severe memory bloat, high system overhead, and catastrophic deadlock vulnerabilities:

1. **Massive Physical Memory Duplication**:
   - Because early development encountered `Errno 24: Too many open files` when naively dispatching tensors over multiprocessing queues, the system adopted an isolated disk-serialization fallback.
   - Every epoch, model parameters are pickled to disk, and each of the 90 child processes independently unpickles and instantiates a separate PyTorch policy network in its private heap memory.
   - For a 200MB model, 90 worker processes consume $90 \times 200\text{MB} = 18\text{GB}$ of physical RAM solely to store identical model weights, risking immediate Out-Of-Memory (OOM) failures under high-concurrency setups.
2. **Epoch-Level Worker Process Churning**:
   - Worker processes are short-lived. Each training epoch spawns 90 new Python interpreter processes via `spawn` mode and destroys them at the end of the rollout phase.
   - Each spawn forces dynamic library loading, PyTorch runtime imports, and environment re-initialization, burning 3 to 8 seconds of idle CPU latency per epoch and causing heavy operating system context-switching turbulence.
3. **Serial Join Timeout Accumulation and Deadlock Hazard**:
   - Process pool destruction uses a sequential `for process in processes: process.join(timeout=10)` loop.
   - If worker processes become unresponsive or hang due to unconsumed queue buffers (Python `multiprocessing.Queue` background feeder thread blocking), the main process waits up to 10 seconds per worker sequentially.
   - In a 90-worker configuration, this causes a worst-case cumulative hang of up to $90 \times 10\text{s} = 900\text{s}$ (15 minutes), appearing as an unrecoverable system freeze.
4. **Execution Disconnection Between Exploration and Training**:
   - Exploration and network gradient optimization are rigidly serialized into alternating batch phases. During rollout, the GPU remains completely idle; during network training, 90 CPU cores remain completely unutilized.

---

## Solution

We architect and implement the **Shared-Memory Inference and Persistent Rollout Worker Pool Architecture** to resolve memory redundancy, eliminate lifecycle churning, and guarantee bounded safe shutdown:

1. **Single-Copy Shared Inference Model (`SharedInferenceModel`)**:
   - Configure PyTorch multiprocessing sharing strategy to `file_system` and raise the operating system open file limits to permanently eliminate `Errno 24`.
   - The main process initializes a single CPU-based inference policy network and invokes `share_memory_()`, placing parameter and buffer storage directly into POSIX shared memory.
   - Child processes receive the shared model reference upon pool initialization and perform zero-copy, read-only forward inference under `torch.no_grad()` and `eval()` mode. Physical memory consumption drops from 18GB to 200MB (98.9% memory savings).
2. **Zero-Overhead In-place Weight Hot Synchronization (`In-place Weight Sync`)**:
   - Following GPU training in each epoch, the main process writes updated GPU parameters directly into the CPU shared memory pages via in-place tensor copying (`copy_`).
   - All 90 child processes immediately observe the latest policy weights on their next forward pass within 5 milliseconds, requiring zero inter-process communication serialization or file exchange.
3. **Cross-Epoch Persistent Worker Pool (`PersistentRolloutPool`)**:
   - Worker processes are spawned once at the start of diverse training and maintained as a persistent pool across all training epochs.
   - Workers execute a lightweight event-driven state loop: idling on incoming rollout tasks, executing rollouts with cached trading environments, reporting transitions, and returning to idle state without process destruction.
   - If exploration is permanently skipped due to buffer saturation or exploration exhaustion, the pool is immediately reclaimed to release operating system resources for downstream network updates.
4. **Concurrent Safe Shutdown Watchdog (`Watchdog Shutdown`)**:
   - Replace sequential join timeouts with a global countdown watchdog. Rollout queues invoke `cancel_join_thread()` to eliminate feeder thread lockups.
   - The main process issues shutdown commands, polls all worker processes concurrently against a unified 10-second deadline, and batch-terminates (`SIGTERM` followed by `SIGKILL`) any lingering workers, guaranteeing bounded shutdown within 12 seconds.

---

## User Stories

1. As a quantitative RL researcher, I want 90 rollout worker processes to share a single 200MB policy network in physical memory, so that total model memory consumption stays strictly bounded at 200MB instead of exploding to 18GB.
2. As a system operator running large-scale training jobs, I want the training system to configure the `file_system` sharing strategy automatically, so that multi-process tensor sharing never crashes with `Errno 24 Too many open files`.
3. As a reinforcement learning practitioner, I want worker processes to persist across epochs during diverse training, so that the pipeline eliminates 3 to 8 seconds of redundant process spawning and library import overhead every epoch.
4. As a machine learning engineer, I want the main process to sync updated GPU weights into shared memory via in-place memory copy in under 5ms, so that all workers immediately evaluate the updated policy without IPC serialization overhead.
5. As an infrastructure engineer, I want child worker processes to perform forward inference strictly under `torch.no_grad()` and `eval()` mode, so that shared model parameters are guaranteed to remain untouched and thread-safe.
6. As a DevOps engineer, I want worker pool shutdown to enforce a global unified timeout rather than sequential per-process timeouts, so that pool teardown never hangs for hundreds of seconds when worker exceptions occur.
7. As a system architect, I want inter-process communication queues to bypass feeder thread joining on shutdown, so that unconsumed transition remnants never deadlock the main process during early termination.
8. As a quantitative researcher, I want the persistent worker pool to shut down immediately when the replay buffer reaches maximum capacity, so that unused CPU workers are freed during the remaining pure-training epochs.
9. As a terminal user, I want the persistent worker pool to register graceful exit handlers, so that interrupting training with `Ctrl+C` cleanly terminates all 90 child processes without leaving orphan zombie processes.
10. As a test automation engineer, I want unit and integration tests to verify that model parameters in child processes point to the identical shared memory addresses as the parent model, so that zero-copy behavior is verified.
11. As a performance engineer, I want worker processes to reuse their internal `TradingEnv` instances across rollout tasks, so that environment construction and memory allocations are executed only once.
12. As an RL practitioner, I want pretraining exploration to retain its decoupled single-pass semantics while adopting safe concurrency, so that existing pretraining diagnostics and baseline metrics remain backwards compatible.

---

## Implementation Decisions

### Modules Modified and Built

- **`FineFT/RL/DiHFT/low_level/shared_model_manager.py` (New Module)**:
  - Houses `SharedInferenceManager`, responsible for configuring `torch.multiprocessing.set_sharing_strategy('file_system')`, initializing CPU `ensemble_Qnet`, executing `share_memory_()`, and performing in-place GPU-to-CPU weight synchronization via `param.copy_()`.
- **`FineFT/RL/DiHFT/low_level/persistent_pool.py` (New Module)**:
  - Implements `PersistentRolloutPool` as a context manager. Encapsulates process lifecycle, task queue dispatching, result streaming, and global-deadline concurrent watchdog shutdown with `cancel_join_thread()`.
- **`FineFT/RL/DiHFT/low_level/parallel_diverse_train.py` (Modified Module)**:
  - Replaces the per-epoch `start_parallel_workers` and `shutdown_exploration_workers` inside `run_epoch_exploration` with the outer-level `PersistentRolloutPool` managed context inside `run_parallel_diverse_training`.
  - Replaces disk-based state-dict pickling with `worker_config["shared_model"]`.
  - Updates `DfRolloutWorkerRunner` to directly bind the shared model instance and evaluate forward actions without re-instantiating networks.
- **`FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py` (Modified Module)**:
  - Replaces sequential process joining in `shutdown_workers` with concurrent deadline polling and queue feeder cancellation.

### Interfaces and Contracts

- **Shared Model Configuration Interface**:
  - Worker configurations convey the shared model via `worker_config["shared_model"] = shared_inference_model`.
  - When `"shared_model"` is present, `DfRolloutWorkerRunner` assigns `self.model = worker_config["shared_model"]`, sets `self.model.eval()`, and skips model creation.
- **Persistent Worker State Machine**:
  - Prototype snippet encoding the worker loop:
    ```python
    while True:
        task = task_queue.get()
        if isinstance(task, WorkerShutdownCommand):
            break
        result = runner.run_task(task)
        result_queue.put(result)
    ```
- **Watchdog Shutdown Contract**:
  - Step 1: Broadcast `WorkerShutdownCommand()` to task queues.
  - Step 2: Invoke `cancel_join_thread()` on task and result queues.
  - Step 3: Poll `all(not p.is_alive() for p in processes)` until a global 10.0-second deadline.
  - Step 4: Batch issue `process.terminate()` to any lingering processes, followed by a 2.0-second grace window before issuing `os.kill(pid, signal.SIGKILL)`.

---

## Testing Decisions

### What Makes a Good Test

Tests must verify external behavioral contracts, memory sharing invariants, and process cleanup guarantees rather than private implementation details:
1. Verify that weights modified in the parent model via in-place sync are immediately reflected in child processes without restarting the pool.
2. Verify that child processes do not allocate independent model weights (verifying `param.is_shared() is True`).
3. Verify that the pool persists across simulated epochs without recreating processes.
4. Verify that shutdown completes within the bounded deadline even when a worker process is simulated to hang or ignore shutdown signals.

### Primary Testing Seams

- **Seam 1 (Highest Integration Seam)**: `FineFT/RL/DiHFT/low_level/parallel_diverse_train.py:run_parallel_diverse_training`
  - Integration seam driving end-to-end multi-epoch diverse exploration, in-place weight sync, and pool teardown.
- **Seam 2 (Component Lifecycle Seam)**: `PersistentRolloutPool` and `SharedInferenceManager`
  - Direct unit seam verifying process initialization, task dispatching, in-place weight synchronization, and concurrent shutdown behavior.
- **Seam 3 (Watchdog Teardown Seam)**: `shutdown_workers`
  - Robustness seam verifying graceful termination under simulated worker blockage without sequential timeout accumulation.

### Prior Art in Codebase

- `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py:1282-1390` (`test_start_parallel_workers_*`, `test_shutdown_exploration_workers_*`)
- `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py:782-920` (`test_run_parallel_diverse_training_completes_exploration_before_training`)

---

## Out of Scope

1. **Shared Memory for Market DataFrames (Phase 2)**:
   - Transforming tabular DataFrame caches into zero-copy PyTorch tensors or Apache Arrow shared memory tables is deferred to Phase 2. Workers will continue loading their assigned DataFrame cache once at initial startup.
2. **Fully Asynchronous Rollout-Learner Decoupling (Phase 3)**:
   - True continuous asynchronous actor-learner pipelining (such as Ray RLlib / IMPALA style continuous background buffering) is deferred. The pipeline maintains the established epoch-level alternating synchronization barrier ("Explore All Tasks -> Update Network -> Sync Weights").
3. **Stage II / Stage III VAE and Routing Refactoring**:
   - Changes are strictly isolated to Stage I low-level reinforcement learning exploration and do not alter VAE training or routing pipelines.

---

## Further Notes

- **Linux FD Limits**: In shell execution environments, ensure `ulimit -n 65536` is applied to eliminate any container-level descriptor limits.
- **Backwards Compatibility**: Existing test fixtures that mock `start_parallel_workers` and `shutdown_exploration_workers` will be updated to align with the persistent pool lifecycle interface.
