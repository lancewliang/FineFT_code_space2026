# 0046. 基于路由分位数预计算与轻量调优回测的高层 VAE Optuna 内存优化架构

为彻底解决高层 VAE 路由超参数搜索（`vae_routing_optuna.py`、`vae_optuna_fu_10.sh`）在大规模子进程并发（如 40 个 Worker）下出现的物理内存耗尽（40×1.38GB 暴增至 55.2GB 内存，触发 5.3GB Swap 磁盘颠簸与 OOM 风险），以及在 100 个 Trial 中对静态行情重复执行 1,800 万次 CPU 深度推断与 14,400+ 次磁盘落盘的算力严重空转，我们决定在高层路由中引入基于**“路由分位数预计算缓存”**的**“缓存感知路由（Cache-Aware Routing）”**架构，并配套**“轻量调优回测（Lean Tuning Rollout）”**与**“最优超参回放（Best-Trial Replay）”**机制。

## Status

accepted

## Context

在目前的 FineFT 高层 VAE 门控路由体系中：
1. **静态模型多进程重复拷贝**：每个 Worker 进程独立实例化 `vae_risk_aware_routing`，无条件加载 6 个庞大的 `MLP_VAE` 模型（`vae_hidden_dims=[4096, 2048, 1024, 1024]`，单模型 95MB，6 个模型占 570MB 物理内存）。当配置 `--n_workers 40` 时，仅完全相同的只读 VAE 神经网络权重即在堆中冗余拷贝 40 份，独占 **22.8GB** 物理内存；
2. **算力巨大浪费**：VAE 模型的输入特征完全来自行情的历史技术指标（`vae_slope_indicators` 与 `vae_volatility_indicators`），与强化学习的交易动作及 Optuna 采样的路由超参数（`gamma`, `window_length`, `rule_base_threshold`, `action_persistence` 等）100% 解耦。验证集 12 个合约的全部历史数据累计仅为 **30,363 行**，全量分位数矩阵仅 **728 KB**。但历史代码在每个进程的每一步仿真中均实时调用 6 个 VAE 模型的单样本推断，导致 100 个 Trial 累计在 CPU 上重复计算了 $100 \times 30,363 \times 6 = \mathbf{1,821.78 \text{ 万次}}$ 深度前向传播；
3. **高频落盘挤爆 Dirty Page Cache**：每个 Trial 针对每个合约写入 12 个 `.npy` 历史轨迹（如 `reward_history.npy`、`total_asset_history.npy` 等），100 次 Trial 触发 14,400+ 次磁盘写入，剧烈消耗操作系统 Page Cache 并引发系统级 Swap 换页；
4. **公共工具类多场景共用约束**：`vae_routing_util.py` 不仅服务于多进程 Optuna 调优，还作为单进程最终评测（`vae_routing_final_result_macro_action.py`）以及多个单元测试的公共基础设施。重构必须保证对单进程 `final_result` 和单元测试的 100% 功能兼容。

## Considered Options

- **选项 1：仅依靠 PyTorch `model.share_memory_()` 共享 6 个 VAE 模型**：
  主进程加载 6 个 VAE 模型并映射至 POSIX 共享内存。虽然能将 40 个进程的模型内存由 22.8GB 降至 570MB，但无法消除 1,800 万次 CPU 神经网络推断计算瓶颈，也无法解决 14,400+ 次磁盘写盘引起的 Page Cache 颠簸。
- **选项 2：彻底移除 `vae_routing_util` 中的 6 个 VAE 模型加载能力**：
  将 `vae_risk_aware_routing` 强行改为纯查表，完全删除模型加载。
  *缺陷*：直接破坏单进程 `final_result` 评测工具与现有单元测试对真实模型推断的依赖，属于不合理的破坏性破坏。
- **选项 3：分位数预计算 + 缓存感知路由 + 调优期免落盘 + 最优超参回放（选中）**：
  1. **路由分位数预计算缓存（Routing Quantile Cache）**：在启动 40 个 Worker 之前，主进程 JIT 自动检查并对验证集 12 个合约执行一次批量批处理推断，将 6 个分位数持久化为 `(T, 6)` float32 格式的 `quantiles.npy`（全量仅 728 KB）；
  2. **缓存感知路由（Cache-Aware Routing）**：
     - 若检测到预计算分位数文件存在（Optuna 场景）：跳过 6 个 VAE 模型加载（`self.vae_models = None`），通过 `np.load(path, mmap_mode="r")` 零拷贝共享内核页，仿真步进直接微秒级数组切片查表；
     - 若缓存不存在（单进程 `final_result` 或 Mock 测试场景）：完全回退至原位加载 6 个 VAE 深度模型并走原生推断；
  3. **轻量调优回测（Lean Tuning Rollout）**：Optuna 的 `objective(trial)` 内部传入 `save_artifacts=False`，跳过中间 Trial 的任何 `.npy`、CSV 磁盘写入与目录创建，纯在内存中累计 reward 并返回标量 `return_rate`；
  4. **最优超参回放（Best-Trial Replay）**：Optuna 调优收敛后，主进程提取 `study.best_trial.params` 单跑一次 `save_artifacts=True` 回测，集中留存最优参数的全量诊断报告与历史轨迹。

## Consequences

- **正面收益**：
  - **单 Worker 物理内存暴降 84%**：从 ~1.38 GB 降至 ~220 MB，40 个 Worker 总内存由 55.2 GB 骤降至 ~9.1 GB（仅占 62GB 宿主机的 14.6%），彻底根治多进程爆内存与 Swap 颠簸；
  - **并发容量提升 4~5 倍**：可在 62GB 宿主机上平稳启动 80~120 个 Worker 跑满多核 CPU；
  - **搜索耗时缩短 80%~90%**：单次 Optuna 100 Trial 运行从 35~50 分钟大幅压缩至 3~5 分钟；
  - **零磁盘颠簸**：消除 14,400+ 次中间 `.npy` 写入，彻底释放操作系统 Dirty Page Cache；
  - **100% 向后兼容**：`final_result_fu_10.sh` 与现有自动化单元测试行为与接口严格保持不变。
- **权衡与约束**：
  - 预计算分位数矩阵固定保存在 `analysis_result/DiHFT/high_level/{dataset_name}/{experiment_name}/vae_quantiles/{eval_stage}/{contract}.npy`，当底层 VAE 模型重新训练后需清空该目录重新生成；
  - Optuna 调优过程中不落盘中间淘汰 Trial 的 `.npy` 历史，符合调优惯例，但如需事后全量 Trial 诊断需单独开启开关。
