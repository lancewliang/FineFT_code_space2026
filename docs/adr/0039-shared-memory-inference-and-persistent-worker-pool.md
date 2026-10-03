# 0039. 基于共享内存的只读推断模型与跨 Epoch 常驻探索进程池架构

为彻底解决大规模并发子进程（如 90 个 Worker）在低层多样化强化学习探索中的物理内存爆炸（90×200MB 模型膨胀至 18GB）与频繁创建销毁损耗，我们决定将模型推断架构重构为基于 POSIX 共享内存的单副本 CPU 共享模型（`SharedInferenceModel`），并通过跨 Epoch 常驻探索进程池（`PersistentRolloutPool`）消除每轮训练的 `spawn`/`destroy` 系统颠簸与串行 `join(timeout=10)` 超时累加挂死隐患。

## Status

accepted

## Context

在现有低层强化学习探索中，历史代码因担忧直接通过 `torch.multiprocessing.Queue` 传递张量会耗尽系统文件描述符（`Errno 24 Too many open files`），采用了向 `/dev/shm` 写入 pickle 序列化文件、由各子进程独立反序列化并实例化独立模型的折中方案。当并发子进程规模扩充至 90 个时，该方案引发了严重的物理内存成倍冗余（仅模型权重即占用 18GB 内存）、每个 Epoch 频繁启动/销毁 90 个 `spawn` 解释器进程的严重性能开销，以及停机时 `for p in processes: p.join(timeout=10)` 在异常情况下长达 900 秒的线性超时累加风险。

## Considered Options

- **选项 1：维持现状，仅依靠调大宿主机内存与超时参数**：无法从根源解决 90 个进程的内存膨胀与频繁进程抖动，容易在更大并发或多任务并发时触发系统 OOM。
- **选项 2：统一全局进程池跨越预训练与多样化训练**：虽然能实现全局复用，但因预训练阶段（`PretrainCollectRunner`）与多样化训练阶段（`DfRolloutWorkerRunner`）的环境初始化和任务协议不同，引入了过高的运行时状态切换复杂度。
- **选项 3：仅重构多样化训练为常驻进程池，并采用 PyTorch `file_system` 共享策略与 `model.share_memory_()`（选中）**：
  1. 通过 `torch.multiprocessing.set_sharing_strategy('file_system')` 与提升 `ulimit -n` 彻底根除文件描述符耗尽问题；
  2. 主进程在 CPU 上创建唯一的 `SharedInferenceModel` 并执行 `share_memory_()`，90 个子进程通过 `mmap` 零拷贝共享同一物理内存页；
  3. 主进程在 GPU 训练完毕后，通过 `param.copy_()` 原位覆写共享内存，耗时 < 5ms 完成无锁权重热同步；
  4. 子进程跨 Epoch 常驻运行，停机时采用全局截止时间统一轮询并发 Join + 批量 Terminate/Kill，并在队列上配置 `cancel_join_thread()` 彻底消除 Python 队列 Feeder 线程卡死。

## Consequences

- **正面收益**：
  - 90 个子进程的模型权重内存开销由 18GB 骤降至 200MB（降幅 98.9%）；
  - 消除每个 Epoch 3~8 秒的进程重复冷启动与环境重新初始化时间；
  - 停机回收总耗时有界收敛在 10~12 秒内，彻底杜绝最坏 900 秒串行累加超时。
- **权衡与约束**：
  - 子进程前向推断必须严格处于 `model.eval()` 和 `with torch.no_grad():` 上下文中，禁止在共享张量上触发任何反向传播计算图跟踪或原地写操作；
  - 多样化训练各阶段保持严格的“全量探索完成 -> GPU 训练 -> 原位覆写共享内存 -> 下一轮探索派发”的串行步调，确保无锁同步的绝对安全。
