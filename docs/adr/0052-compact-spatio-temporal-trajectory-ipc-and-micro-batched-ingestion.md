# 0052. 紧凑时空轨迹回传与微批次聚合入池架构

为彻底解决低层多样化探索（Stage I）中“40 个探索子进程快速完成推演，主进程却因重型轨迹序列化、海量逐步 Blake2b 哈希计算与碎步 PCIe 同步耗费数十秒回传与入池”的性能假瓶颈，我们决定确立**紧凑时空轨迹回传与微批次聚合入池架构**：探索子进程彻底剔除 148 维静态市场状态与嵌套字典的 IPC 传输，改用紧凑时空数值序列；去重键由时空坐标与离散动作决策直接映射为整数键；主进程采用微批次聚合机制统一写入 GPU 驻留经验池。

## Status

accepted

## Context

在 Stage I 多样化探索训练的实际压测中（如 `fu/10min_parallel` Trial 342，40 个 Worker 执行 462 个探索任务）：
1. **IPC 传输量严重膨胀**：旧实现中每个 Worker 逐步打包包含 148 维浮点状态 `state`、`next_state`、25 档盘口量价切片及庞大嵌套 `info` 字典的 Python 对象元组。单轮 462 任务通过 `multiprocessing.Queue` 传输的序列化数据量高达 7~11 GB，引发剧烈的管道拥塞与内存垃圾回收（GC）停顿；
2. **时空状态重复哈希计算**：主进程入池时对每条轨迹的每一步调用 `hashlib.blake2b(state.tobytes())` 计算语义决策键。单轮逾 13 万步重复执行浮点数组序列化与加密哈希，耗费大量 CPU 算力；
3. **单任务碎步显存写入与同步**：主进程每从队列拉取一个任务结果，就立即调用 `add_round_records_bulk([result])`。462 个任务触发 462 次小粒度 CPU/GPU 张量转换与 PCIe 写入，引发严重的 CUDA 调度开销与主机/设备同步延迟；
4. **去重权责边界需厘清**：关于是否应在 Worker 端提前去重的疑问，经深入分析确认：单任务内部时间单调递增重复率为 0；跨 Worker 去重受限于只读共享内存无法无锁感知，且破坏 FineFT 核心的“$td\_error_{new} > td\_error_{old}$ 择优覆盖”逻辑；在传输量压缩 99.8% 后传输已非瓶颈，Worker 应保持零锁纯并行推演。

## Considered Options

- **选项 1：Worker 本地哈希去重或共享内存锁表去重**：
  在各 Worker 端建立去重表，提前剔除重复样本。
  *缺陷*：单任务内部状态天然不重；跨 Worker 共享去重表必须引入进程间互斥锁（IPC Lock），40 个高频推演进程产生严重锁争用；Worker 无法感知主进程 GPU 经验池内已存样本的 `old_td_error`，破坏择优覆盖语义。
- **选项 2：共享内存环形缓冲区 (Ring Buffer)**：
  在 POSIX 共享内存中开辟全局大环形数组，Worker 写入偏移指针。
  *缺陷*：需要复杂的原子游标管理与多进程并发安全控制，开发维护成本高，且未解决主进程入池时的 Blake2b 哈希瓶颈。
- **选项 3：紧凑时空数值序列回传 + 纳秒级时空去重键 + 微批次聚合入池（选中）**：
  1. **紧凑时空轨迹回传 (Compact Spatio-Temporal Trajectory IPC)**：
     - 鉴于 `SharedMarketDataPack` 已在全进程共享内存常驻，Worker 彻底不再通过 Queue 回传 148 维 `state`、`next_state` 数组及冗余 `info` 键；
     - Worker 仅回传轻量一维/二维数值连续数组（`actions`、`rewards`、`td_errors`、`dones`、`step_indices`、`next_step_indices`、`trading_infos`、`avaliable_actions` 等），单轮传输量由 7~11 GB 骤降至 14~25 MB（压缩率超 99.7%）；
  2. **时空决策去重键 (Spatio-Temporal Semantic Deduplication Key)**：
     - 同一切片内的行情状态由 `(df_index, step_index)` 严格唯一确定，去重键重构为整型时空元组 `(df_index, step_index, previous_action, action, pos_dir, dur_bucket)`；
     - 彻底废除 148 维浮点数组的 `blake2b` 密集加密哈希计算，去重键提取耗时从微秒级降至纳秒级；
  3. **微批次聚合入池 (Micro-Batched Buffer Ingestion)**：
     - 主进程改变 462 次单任务小碎步入池模式，引入微批次收集缓冲机制（攒满 16 任务或队列暂空即触发批量写入）；
     - 批量聚合数千步经验，直接从常驻宿主内存的 `SharedMarketDataPack` 零拷贝切片还原状态向量，单次批量写入 GPU 经验池，消除 95% 以上的 PCIe 搬运与同步开销；
  4. **零锁只读共享安全**：
     - Worker 对 `SharedMarketDataPack` 和 `SharedInferenceModel` 的访问均为纯只读无状态推演，无并发写冲突，严格杜绝任何进程间锁机制。

## Consequences

- **正面收益**：
  - **回传与入池延迟骤降**：单轮 462 任务的回传与入池总耗时由 85 秒压降至 1 秒以内，性能提升逾 80 倍；
  - **内存与 GC 压力彻底释放**：消除每轮数 GB 的短生命周期 Python 字典与对象反序列化，主进程常驻内存平稳无毛刺；
  - **语义完全无损**：严格保持 $td\_error_{new} > td\_error_{old}$ 的择优覆盖规则与 N-Step 贴现回报逻辑；
  - **零锁高并发吞吐**：Worker 进程池维持绝对无锁，纯 CPU 并行推演算力得以 100% 释放。
- **约束与代价**：
  - 依赖 `SharedMarketDataPack` 在共享内存的存在；对于未挂载共享数据包的非生产测试环境，保留兼容的懒加载/降级链路。
