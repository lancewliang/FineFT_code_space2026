# 0040. 基于共享内存的行情数据张量包与常驻交易环境原位重置架构

为彻底解决大规模多进程探索（如 90 个 Worker）中，每个子进程独立反序列化全量市场行情 DataFrame（`train_df_cache`）导致的物理内存指数级膨胀（例如 500MB 数据膨胀至 45GB 物理 RAM），以及每个探索任务频繁抽取 DataFrame 列与重新实例化交易环境带来的性能瓶颈，我们决定将行情数据预先提纯为基于 POSIX 共享内存的只读纯数值张量包（`SharedMarketDataPack`），并升级交易环境支持原位无损重置（`PersistentTradingEnv`）。

## Status

accepted

## Context

在目前的低层多样化强化学习训练中，主进程加载各时间切片与合约的原始行情（`train_df_cache: dict[int, pd.DataFrame]`）后，将其 pickle 导出至 `/dev/shm` 临时文件。90 个探索子进程启动时，各自在独立的堆内存中完整 `pickle.load()` 载入全量 DataFrame 字典，导致单份数据集被复制 90 份，物理内存消耗高达 $90 \times \text{Data Size}$（常态下达 40GB~90GB），面临极高的宿主机 OOM 风险。

此外，当子进程收到 `ExploreTask` 时，每次均通过 `df["mark_price"].values`、`df[feature_list].values` 现场切片提取 NumPy 数组并调用 `create_demo_env(...)` 重新构建 `Demo_Env` 对象。这种高频对象的反复创建不仅带来高额的 CPU 字符串哈希与 Pandas 索引开销，还加剧了 Python 运行时的垃圾回收抖动。

## Considered Options

- **选项 1：维持现状，仅做 DataFrame 子集切片分发**：主进程按子进程编号仅分发其负责的 `df_index` 子集。虽然单进程内存有所降低，但各子进程依然持有独立的 DataFrame 副本，整体物理内存依然存在多倍冗余，且限制了任务跨 worker 的动态弹性负载均衡。
- **选项 2：基于 PyArrow / Feather 的内存映射表（mmap）**：将 DataFrame 导出为 Feather 格式并通过 PyArrow 内存映射加载。保留了高级列式结构，但引入外部依赖，且在进入交易环境时仍需经过一层 Arrow 到 NumPy 的指针适配开销。
- **选项 3：基于 PyTorch CPU 共享张量提纯数据包 + 常驻交易环境原位重置（选中）**：
  1. 主进程在多样化训练启动时，一次性将所有 `df_index` 对应的 DataFrame 字段剥离为纯数值连续数组，转换为 PyTorch CPU 张量（`float32` / `int64`）并调用 `tensor.share_memory_()` 映射至 POSIX 共享内存；
  2. 封装为全局唯一的只读 `SharedMarketDataPack`，所有 90 个子进程均可零拷贝直接通过 `tensor.numpy()` 视图访问任意 `df_index`；
  3. 子进程内部按 `df_index` 缓存 `Demo_Env` 单例，收到新任务时仅重新计算该动作对应的 `initial_state` 并执行原位重置（In-place Reset），终身复用底层共享内存数组；
  4. 数据包生命周期与 `PersistentRolloutPool` 绑定，训练结束或进程池销毁时自动解绑释放，无任何残留文件。

## Consequences

- **正面收益**：
  - 彻底消除 90 个子进程的市场数据内存冗余，物理 RAM 占用从 $90 \times N$ 骤降至 $1 \times N$（90 个 worker 时内存节约 98.8%）；
  - 探索任务开始时无需再进行 DataFrame 列索引查找、字符串切片与内存重新分配；
  - 交易环境对象跨回合常驻复用，仅重置内部状态机游标与账户变量，达到 0 次内存拷贝与 0 次环境析构。
- **权衡与约束**：
  - 共享张量必须严格为只读视图，交易环境在推进步长（`step()`）时严禁在传入的行情数组上进行原地写入或切片篡改；
  - 涨跌停扩展列（如 `UpperLimitPrice` 等）在构建张量包时执行严格自适应提纯与 Fail-fast 校验，缺列但启用了限价奖励时直接阻断启动。
