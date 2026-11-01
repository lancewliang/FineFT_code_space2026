# 十一维交易 state 输入验证记录

状态：已实施并完成短程集成验证。所有 Python/pytest 命令均使用仓库要求的 `finetf` conda 环境；没有启动正式历史数据重训，因此不对收益提升、完整 Markov 性或老师最优性作结论。

本轮从回退基线 `ada8aae` 实施，当前契约是 `trading_observation.v2`：`trading_info` 为有序 `np.float32(11,)`，模型 batch 宽度为 11，环境使用同合约同分区预计算的 18 步因果行情辅助列。实验尺度参数使用 `--exit_price_scale_floor 5`（元/吨），与交易所 `tick_size` 分离；旧四维模型/replay 不补零、不截断，加载时拒绝。

## 定向证据

以下命令均在 `FineFT/` 目录执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
```

- 共享契约、因果尺度、数据贯通与环境生命周期：对应 issue 01、02、07、08 的定向测试已通过；覆盖辅助列 `0/false` 预热、有效恒价 `0/true`、切片原始坐标、浮亏率、双方向退出记忆、实际成交价、锁存尺度、reset、终止仍持仓和无效输入拒绝。
- 串行模型/replay/learner：issue 03 的最终定向组 `18 passed`，覆盖 Qnet/ensemble、11 维当前与 next replay、元数据 sidecar、旧快照拒绝、有限损失和真实参数更新。
- 并行/compact/CPU/GPU：issue 04 的最终定向组 `102 passed`，覆盖 compact IPC、n-step 终点、空批次/尾部微批次、FIFO/随机轮换、CPU replay、可用 CUDA 路径与共享行情包。CUDA 路径随环境可用性执行；并未将不可用设备伪称为已验证。
- 独立评估/路由/OOD：issue 05 的组合定向组 `117 passed, 2 warnings`，覆盖索引评估、平均 ensemble、异步评估、VAE 路由、专家切换记忆、止损路径和 OOD 调用方的 11 维真实前向、sidecar/宽度/floor 校验。
- ticket 06 收尾后的环境/replay/评估夹具回归：`55 passed`。

代表性命令如下：

```bash
python -m pytest \
tests/env/test_position_cost_state.py \
tests/rl/test_learning_targets_and_bounded_replay.py \
tests/rl/test_semantic_prioritized_buffer.py \
tests/rl/test_targeted_subnetwork_rollout_inference.py \
tests/rl/test_expert_hysteresis_audit.py -q
```

上述命令结果为 `55 passed`。issue 04/05 的完整命令和逐票证据保留在对应 issue 文件中。

## 全量回归

从 `FineFT/` 执行：

```bash
python -m pytest -q
```

第一次收集在 `tests/analysis/test_dihft_high_level_heurstic.py` 因既有 `Picker` 导入错误停止。为获得其余仓库结果，排除这一个收集失败文件后执行：

```bash
python -m pytest -q --ignore=tests/analysis/test_dihft_high_level_heurstic.py
```

当前结果：`737 passed, 25 failed, 7 warnings`，耗时约 87 秒。25 个失败没有被改成静默兼容，主要类别为：

- 既有止损持续计数、VAE 路由组合收益/防御默认行为及评估动作持续断言；
- 缺失的 `RL/DiHFT/ablation/converge_steps_sun/EarnHFT_PES.py`；
- 旧日志路径断言和 shell pipeline 的 `SELECTION_EVIDENCE`/步骤控制契约；
- 冻结专家 shell、共享 Q 表默认 worker、CUDA 参数化项和旧 action-persistence 断言。

这些失败不来自本票的十一维宽度、因果辅助列、退出记忆、replay/IPC 或模型加载改动；相关路径已由上述定向组覆盖。全量结果应作为当前仓库基线限制记录，而不是伪造为全绿验收。

## 交付边界

- 已验证：数据辅助列到环境、串行/并行 replay、learner 更新、独立评估与高层路由的契约一致性；当前/next 交易向量均为完整 11 维，缺失、非法宽度、非有限值、无效预热或旧 sidecar 会失败。
- 限制：直接构造 `Base_Env`/低层 factory 的历史单元测试仍可不传辅助列，并只在该 legacy fixture 路径运行时回算退出尺度；真实 SCALE_SAVE/worker/replay 数据入口要求 manifest 和辅助列，不得把 legacy fixture 当作生产训练产物。
- 未验证：正式历史数据重建、长期训练稳定性、收益/回撤改进、完整 Markov 性和扩维老师最优性。
- 运行说明：生产实验需显式传入 `--exit_price_scale_floor 5`，保留 `trading_observation.json` 与因果价格辅助列 manifest；不得把该配置记作交易所最小变动价位。
