# 04: 接通并行训练与紧凑经验传输

**What to build:** 让真实交易观测通过 worker、compact IPC、微批次、CPU/GPU replay 到达 learner，并行预训练和多样化训练各完成短程更新。

**Blocked by:** 03 — 接通模型与串行训练闭环

**Status:** ready-for-agent

**Implementation:** 未实施。此前代码改动已全部回退；本轮只重切任务，不启动代码修改。ready-for-agent 表示规格可实施，不表示依赖已完成。

## 文件范围

parallel_pretrain.py、parallel_weight_advantage_pretrain.py、parallel_diverse_train.py、regime_stratified_replay_buffer.py 及 worker/shared-data 入口。

## 验收标准

- [ ] 所有并行训练、warmup 和周期评估传递 --exit_price_scale_floor、辅助列/有效位/坐标及 01 数据契约；共享行情只存行情，交易记忆来自连续环境。
- [ ] 删除漏传 trading_info 时补零的分支；当前与 next 均严格十一维，compact 声明、空数组、存储分配、批量导入导出和采样全部覆盖。
- [ ] 完整与 compact 当前/n-step 终点数值一致，覆盖完整窗、短窗、终止、空批次、尾部微批次；不以紧邻 next 替代多步终点。
- [ ] 不同退出记忆的经验不合并；保留组上限四、种子轮换、FIFO、索引回收和满池继续探索，不改经验优先级算法。
- [ ] 复用 03 的模型/replay 元数据保存与加载契约，不另建并行兼容协议；旧宽度/缺列/非法配置在边界明确失败。
- [ ] 并行预训练与多样化训练分别有实际采集、传输、采样及真实 learner 参数更新证据；CPU 与可用 GPU 路径分别记录，设备不可用则标明未验证。

## 职责边界

不依赖 05，不更改环境事件定义或数据计算；不以单纯拼造十一维向量代替环境轨迹，不启动正式训练。

共同遵循 [spec](../spec.md) 第 4–6 节与 [实施依赖总览](../implementation-plan.md)。Python/pytest 使用 `conda activate finetf`。不更改已确认范围，不启动正式重训。

## Comments

- 2026-11-01：按确认后的行情统计预计算、Q8 A 与 exit_price_scale_floor 命名重新切分职责；当前验收项全部未完成。

- 2026-11-01（回退后状态校正）：用户确认本功能所有代码改动已回退；重新打开本票，全部验收项恢复未完成。下方记录只描述回退前工作，不能证明当前源码已实现或验证通过。

### 回退前历史记录（不作为当前验收证据）

- 2026-11-01：并行预训练、完整多样化采集、compact IPC、微批量导入和 CPU/GPU replay 均复用共享 11 维校验；缺少交易观测或旧宽度会在 worker/replay 边界立即失败。
- 2026-11-01：并行产物保存 `trading_observation.json`，并行预训练加载和周期评估检查 tick 配置与交易编码层宽度；worker 不再以零向量补齐漏传字段。
- 验证：`python -m pytest tests/rl/test_parallel_diverse_cpu_exploration.py tests/rl/test_compact_trajectory_and_micro_batched_ingestion.py tests/rl/test_gpu_resident_buffer_and_pipeline.py -q`（conda 环境 `finetf`）通过，`26 passed`；CUDA 可用，GPU 路径按测试选择器执行。
