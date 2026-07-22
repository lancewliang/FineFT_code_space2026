# refactor-rl-diagnostics-dataclasses

## 背景与目标

`FineFT/RL/DiHFT/low_level` 中的训练诊断链路目前大量使用匿名 dict 表达跨函数边界数据、
worker queue 消息、诊断结果、日志摘要和 JSON/CSV 写入记录。典型位置包括
`loss_nan_diagnostics.py`、`pretrain_qtable_diagnostics.py` 和
`parallel_weight_advantage_pretrain.py`。

这些 dict 的字段契约散落在函数体和测试中，调用方通过字符串 key 访问字段，容易出现字段
遗漏、拼写错误或返回结构漂移。本次重构目标是将这三个文件中的业务记录、临时数据对象和
落盘前 JSON/CSV 数据生成侧尽量改为 dataclass 表达，同时保持现有训练算法和落盘文件格式
兼容。

## 用户场景

- 开发者排查 loss NaN 时，可以通过 `LossNanDiagnostics`、
  `NumericValueSummary` 和 `NonfiniteLocation` 理解诊断结构，而不是在嵌套 dict 中推断字段。
- 开发者维护 qtable 预训练诊断时，可以通过 `SamplePlanItem`、
  `SampleDiagnostic`、`QTableDiagnosticsManifest` 和
  `PretrainQTableDiagnosticsResult` 理解 sample plan、manifest、CSV 明细和准备结果。
- 开发者维护 parallel rollout 训练时，可以通过 `ParallelRolloutTask`、
  `WorkerRoundResult`、`WorkerErrorMessage`、`RolloutMetrics` 和
  `ParallelRoundSummary` 理解 worker queue 与主进程之间的数据契约。
- 现有训练产物消费者继续读取原有 `manifest.json` 和
  `df_*_initial_action_*.csv` 字段结构，不需要迁移落盘格式。

## 设计方向

采用局部 dataclass 契约重构方案。在三个目标模块内定义贴近当前流程的 dataclass，不新增
大型框架，也不改变训练算法。跨函数边界、worker 消息/结果、diagnostics、manifest、
metrics、summary 和 CSV row 等业务记录改为 dataclass；日志输出和训练调用点改为属性访问。

`loss_nan_diagnostics.py` 定义 `NumericValueSummary`、`NonfiniteLocation` 和
`LossNanDiagnostics`。`build_loss_nan_diagnostics()` 返回 `LossNanDiagnostics`，
`log_loss_nan_diagnostics()` 使用属性访问。对象提供 `to_dict()`，用于保持测试和未来 JSON
输出的结构兼容。

`pretrain_qtable_diagnostics.py` 定义 `SamplePlanItem`、`QTableDiagnosticsManifest`、
`DiagnosticCsvRow`、`SampleDiagnostic`、`QTableWorkerResult` 和
`PretrainQTableDiagnosticsResult`。`prepare_pretrain_qtable_diagnostics()` 返回
`PretrainQTableDiagnosticsResult`，调用方通过 `result.sample_plan`、
`result.q_table_cache`、`result.train_df_cache`、`result.diagnostics` 和
`result.sample_action_cache` 访问数据。manifest 和 CSV 写入前通过 `to_dict()` 转回现有
字段结构。

`parallel_weight_advantage_pretrain.py` 定义 `RolloutMetrics`、
`RolloutMetricsSummary`、`RolloutDiagnosticsSummary`、`ParallelRolloutTask`、
`EpochTrainingParams`、`WorkerTransitionRecord`、`WorkerRoundResult`、
`WorkerErrorMessage` 和 `ParallelRoundSummary`。multiprocessing queue 边界传递模块顶层
dataclass 实例，满足 pickle 要求；主进程按对象类型判断 worker result/error。

允许保留自然的索引容器，例如 `dict[int, pd.DataFrame]`、`dict[int, q_table]` 和
`dict[SamplePlanItem, list[int]]`。这些是 cache/map 容器，不是匿名业务记录；value 或
record 应尽量对象化。

## 关键决策

- 覆盖 `loss_nan_diagnostics.py`、`pretrain_qtable_diagnostics.py` 和
  `parallel_weight_advantage_pretrain.py` 中的 dict 业务记录、临时数据对象、worker payload
  和 JSON/CSV 生成结构。
- 允许公开函数返回值从 dict/tuple 改为 dataclass，调用方和测试同步改为属性访问。
- 使用标准库 `dataclass`，不引入 pydantic 或其他运行时 schema 依赖。
- 落盘 `manifest.json` 和 `df_*_initial_action_*.csv` 的字段名、层级和主要数值类型保持兼容。
- `to_dict()` 只用于 JSON/CSV 写入边界、日志兼容和测试中与旧结构对比。
- multiprocessing queue 中传递的 dataclass 必须定义在模块顶层，避免 pickle 问题。
- 保留 cache/map 形态的 dict 容器，但不再用匿名 dict 表达业务记录字段。
- 不改变 qtable 构建、DP action path 回放、rollout exploration、loss 更新和训练参数调度逻辑。

## 范围边界

**包含：**

- 将 `build_loss_nan_diagnostics()` 返回值改为 `LossNanDiagnostics`。
- 将 loss NaN numeric summary 和 info nonfinite location 改为 dataclass。
- 将 qtable sample plan item、manifest、CSV row、sample diagnostic、worker result 和
  prepare result 改为 dataclass。
- 将 `prepare_pretrain_qtable_diagnostics()` 的调用方改为使用
  `PretrainQTableDiagnosticsResult` 属性访问。
- 将 parallel rollout task、epoch training params、worker reset/explore/shutdown payload、
  worker result/error、transition record、rollout metrics 和 round summary 改为 dataclass。
- 更新 focused tests，覆盖对象返回值、属性访问、`.to_dict()` 兼容性和 worker 边界。

**不包含（本次）：**

- 不改变 `manifest.json` 外部 JSON 结构。
- 不改变 qtable diagnostics CSV 文件名、字段名或字段含义。
- 不改变训练算法、模型结构、CLI 参数、日志语义或目录布局。
- 不引入第三方数据校验库。
- 不重构目标三个文件之外的模块，除非调用接口变更要求最小适配。
- 不强行包装纯索引缓存容器，例如 `df_index -> DataFrame` 和 `df_index -> q_table`。

## 验收标准

- [ ] `build_loss_nan_diagnostics()` 返回 `LossNanDiagnostics`，调用方可通过
  `diagnostics.numeric` 和 `diagnostics.info_nonfinite` 属性访问。
- [ ] `LossNanDiagnostics.to_dict()` 保持旧的 `{"numeric": ..., "info_nonfinite": ...}` 结构。
- [ ] `build_sample_plan()` 返回 `list[SamplePlanItem]`，`select_sample_from_plan()` 返回
  `SamplePlanItem`。
- [ ] `evaluate_and_export_sample()` 返回 `SampleDiagnostic`，CSV 写出字段与现有格式一致。
- [ ] `QTableDiagnosticsManifest.to_dict()` 写出的 `manifest.json` 与现有字段结构兼容。
- [ ] `prepare_pretrain_qtable_diagnostics()` 返回 `PretrainQTableDiagnosticsResult`，
  `weight_advantage_pretrain.py` 和 `parallel_weight_advantage_pretrain.py` 调用方使用属性访问。
- [ ] `parallel_weight_advantage_pretrain.py` 的 rollout task、worker result/error、rollout
  metrics 和 round summary 通过 dataclass 表达，测试不再依赖这些业务记录的字符串 key 访问。
- [ ] worker 异常仍通过 result queue 返回，并由 `raise_for_worker_error()` 抛出包含
  `df_index`、`epoch_index`、`context_index`、`initial_action` 和 `round_counter` 的
  `RuntimeError`。
- [ ] 现有 qtable diagnostics CSV 缓存读取逻辑在 manifest 匹配时仍可复用旧 CSV，manifest
  不匹配或 CSV 字段缺失时仍重新计算。
- [ ] 使用 `conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py FineFT/tests/rl/test_pretrain_qtable_diagnostics.py FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q`
  通过 focused test 验证。
- [ ] 使用 `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/loss_nan_diagnostics.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`
  通过语法验证。
