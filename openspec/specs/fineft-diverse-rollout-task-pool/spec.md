# Spec: Diverse Rollout Task Pool with Decoupled Workers for High-Concurrency Low-Level Exploration

Related Documents:
- Domain Glossary: [CONTEXT.md](/home/lanceliang/opt/aiwork/FineFT_code_space2026/CONTEXT.md:328)
- Architectural Decision Record: [docs/adr/0015-rollout-task-pool-for-diverse-exploration.md](/home/lanceliang/opt/aiwork/FineFT_code_space2026/docs/adr/0015-rollout-task-pool-for-diverse-exploration.md:1)

Triage Label: `ready-for-agent`

---

## Problem Statement

During Stage I low-level reinforcement learning diverse training, the exploration phase is intended to collect diverse policy trajectories across combinations of sub-agent context indices, initial actions, and market data slice files. On modern high-core multi-socket servers (such as 192-thread NUMA machines), reinforcement learning practitioners expect to scale up CPU exploration concurrency to 64, 96, or 128 parallel worker processes to drastically accelerate training throughput.

However, the existing diverse training implementation severely throttles parallelism through two fundamental architectural constraints:
1. **Slice-Bound Worker Ceiling**: Each exploration worker process is hard-bound to a disjoint subset of training data slice indices (`df_index`), and the number of workers is capped by `min(len(effective_df_indices), MAX_EXPLORATION_WORKERS)`. Because sliced training datasets typically contain only around 16 files, at most 16 worker processes can ever be spawned, leaving over 90% of available CPU cores completely idle regardless of user intent.
2. **Synchronous Barrier Straggler Bottleneck**: The orchestrator iterates through sub-agent contexts and initial actions in sequential nested loops. Within each step, 16 tasks are dispatched across slice-bound workers, followed by a blocking barrier that stalls the entire pipeline until the slowest slice finishes. Because slice lengths vary up to sevenfold (e.g. 5,000 steps vs 35,000 steps), fast workers sit idle for seconds at every barrier, accumulating severe multi-hour delays across training epochs.

From the user's perspective, running diverse training on high-performance hardware suffers from low CPU utilization, excessive epoch exploration time, and an inability to configure higher worker counts.

## Solution

Deconstruct the static slice-bound worker topology into a **Diverse Rollout Task Pool** powered by **Decoupled Exploration Workers**.

All training data slice files (which total only ~34 MB across 16 slices) are pre-loaded by every worker process in the pool. The exploration workload for each epoch—consisting of all $N \times |A_{\text{pos}}| \times |DF|$ independent trajectory episodes—is flattened into a shared task queue of atomic task descriptors. A configurable pool of general-purpose worker processes (defaulting to 96 workers to match physical core counts) continuously draws tasks from the queue and executes episodes to completion without any intermediate synchronization barriers.

Worker inference is strictly restricted to single-threaded CPU execution (`torch.set_num_threads(1)`) to eliminate multi-process thread contention. As completed rollout results stream back through a unified result queue, the main process inserts transitions directly into the regime-stratified replay buffer and checks early-termination capacity thresholds. Model parameters are synchronized once per epoch at worker startup rather than duplicated across task messages, maintaining microsecond queue latencies.

## User Stories

1. As an RL practitioner, I want to configure the number of parallel exploration workers via a dedicated command-line flag `--diverse_num_workers`, so that I can tailor exploration concurrency to my host machine's core capacity.
2. As a systems engineer, I want `--diverse_num_workers` to default to 96 on high-core servers, so that exploration saturates physical compute cores out of the box without manual tuning.
3. As a systems engineer, I want `--diverse_num_workers` to enforce strict positive integer validation, so that non-positive or invalid worker counts fail fast with clear errors during trainer initialization.
4. As an RL research engineer, I want the exploration worker count to be completely decoupled from the number of dataset slice files, so that I can run 64, 96, or 128 workers even when only 16 slice files exist.
5. As an RL research engineer, I want every exploration worker to have access to all cached training slices, so that any worker can execute an episode on any dataset slice dynamically.
6. As a performance engineer, I want all rollout episodes in an epoch ($N \times |A_{\text{pos}}| \times |DF|$) to be dispatched into a central shared task queue, so that fast workers immediately pull new tasks without waiting for slow workers.
7. As a quantitative researcher, I want all synchronization barriers between different initial actions and sub-agent contexts within an epoch exploration phase to be eliminated, so that the straggler problem from uneven slice lengths is completely resolved.
8. As a low-level RL developer, I want exploration tasks to be represented as atomic messages containing all necessary execution parameters, so that environment reset and trajectory simulation execute atomically within a single worker dispatch without race conditions.
9. As an RL system operator, I want policy model weights to be loaded once per worker at epoch initialization rather than serialized into every task message, so that IPC queue bandwidth and memory overhead remain minimal.
10. As a performance engineer, I want each worker process to enforce single-threaded PyTorch CPU execution (`torch.set_num_threads(1)`), so that dozens of concurrent worker processes do not thrash CPU caches or contend on thread pools.
11. As an RL researcher, I want trajectory results to be streamed from the worker pool directly into the regime-stratified replay buffer as they complete, so that buffer capacity limits are monitored in real time.
12. As a strategy developer, I want all completed trajectory transitions to retain valid N-step discounted returns and initial-step regime grid tags, so that downstream regime curriculum sampling functions identically to the sequential baseline.
13. As an RL researcher, I want context-level and epoch-level rollout metrics and TensorBoard scalar summaries to be computed and recorded upon epoch collection completion, so that existing experiment diagnostics and logging contracts remain fully preserved.
14. As a system operator, I want all worker processes in the task pool to be cleanly shut down and confirmed terminated before the neural network training phase begins, so that child processes never compete with GPU training for system resources.
15. As a system operator, I want the orchestrator to guarantee thorough worker termination even if an unexpected exception or worker error occurs, so that orphan processes are never leaked into the background.
16. As a quantitative researcher, I want in-flight tasks to terminate gracefully and extra transitions to be absorbed by regime grid capacities when the replay buffer reaches maximum capacity, so that early stopping does not cause deadlocks or corrupt trajectory accumulation.
17. As an engineer maintaining unit tests, I want the underlying worker runner methods for task reset and step exploration to remain accessible, so that existing low-level mock and unit test suites continue to pass.

## Implementation Decisions

### Architectural Topology
- **Diverse Training Orchestration Component**: Decompose the nested loops over contexts, initial actions, and slice indices in the epoch exploration orchestrator into a single task generator that populates a shared multiprocessing `task_queue`. Collect results from a shared `result_queue` until all expected tasks are accounted for or early termination triggers.
- **Worker Pool Lifecycle**: Launch $W$ worker processes (where $W$ is controlled by `--diverse_num_workers`) at the start of each epoch's exploration phase. Pass the epoch's frozen model state dictionary and the full training slice cache within the worker startup configuration. Terminate all workers cleanly via shutdown sentinel messages and process joins before the neural network parameter update phase commences.
- **Atomic Task Protocol**: Define an atomic task data structure encapsulating target slice index, epoch index, context index, initial action, round counter, and exploration rate epsilon. The worker consumes this message, instantiates the demonstration environment for the specified slice and initial action, executes the rollout policy until `episode.done`, performs N-step accumulation, and returns the completed trajectory record.

### Interface & Configuration Changes
- **Trainer Initialization & CLI Parsing**: Introduce the `--diverse_num_workers` command-line argument to the low-level argument parser. Store this attribute directly on the trainer instance. Validate that `--diverse_num_workers` is an integer strictly greater than zero.
- **Hardcoded Limit Removal**: Eliminate the static constant ceiling `MAX_EXPLORATION_WORKERS = 20` from the diverse training module. The number of spawned processes directly reflects the trainer's configured diverse worker count.
- **CPU Threading Guard**: Inject `torch.set_num_threads(1)` at the entry point of worker initialization to suppress PyTorch's default multi-threaded CPU backend during parallel environment rollouts.

### Result Streaming & Early Stopping Semantics
- **Stream Ingestion**: The main orchestrator reads round results from the shared result queue as they become available, directly writing accumulated transitions into the regime-stratified replay buffer.
- **Buffer Full Handling**: When `is_buffer_full` evaluates to true during result ingestion, the orchestrator sets a termination flag, drains remaining in-flight worker results, applies capacity constraints through the replay buffer's regime retention rules, and proceeds directly to worker shutdown.
- **Metric Aggregation**: Retain rollout metric records in memory indexed by context index and slice index. After all tasks complete or early termination is reached, aggregate the metrics to write context rollout scalars and epoch rollout scalars matching existing TensorBoard schemas.

## Testing Decisions

- **Focus on External Observable Behavior**: Tests must verify external observable behaviors (such as the number of active child processes matching the configured parameter, all expected task combinations being explored, transitions populating the buffer, CPU threads set to 1, and clean shutdown on completion or failure) rather than asserting against private intermediate helper calls.
- **Primary Seam**: The highest testing seam is the epoch exploration entrypoint (`run_epoch_exploration`), which drives the complete worker lifecycle, queue dispatch, result ingestion, and process termination.
- **Secondary Seam**: The atomic task execution interface on the worker runner (`DfRolloutWorkerRunner`), verifying that processing a single atomic task produces a valid result with proper transition counts, returns, and done flags.
- **Prior Art**: 
  - `FineFT/tests/rl/test_parallel_diverse_cpu_exploration.py` (verifies CPU device enforcement and runner execution).
  - `FineFT/tests/rl/test_regime_curriculum_diverse_train.py` (verifies curriculum phase transitions and buffer interactions).
  - `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py` (verifies worker shutdown and parameter parsing).

## Out of Scope

- Modifying the Stage I pretrain warmup collection pipeline (`parallel_pretrain.py`). Pretrain warmup retains its existing slice-bound worker infrastructure.
- Modifying the neural network parameter update routines (`run_diverse_training_phase` and `update`).
- Altering the serial training baseline (`weight_advantage_pretrain.py`).
- Changing the 9-grid regime classification thresholds, turning-point slicing logic, or `RegimeStratifiedReplayBuffer` storage invariants.
- Modifying Stage II agent evaluation or Stage III VAE routing pipelines.

## Further Notes

- Memory analysis confirms that pre-loading all 16 slice files (~34 MB total) across 96 worker processes requires approximately 3.3 GB of host RAM, well within the host machine's 49 GB available memory capacity.
- The 96-worker default aligns with the physical core count of the dual AMD EPYC 7K62 processor configuration, preventing SMT hyperthreading cache collisions while fully saturating hardware throughput.
