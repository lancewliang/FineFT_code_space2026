# add-agent-pattern-dual-classifier

> 双轴形态语义见 [ADR-0006](../../../docs/adr/0006-agent-pattern-dual-classifier.md)，旁路采集边界见 [ADR-0007](../../../docs/adr/0007-isolate-agent-pattern-data-collection.md)，术语见 [CONTEXT.md](../../../CONTEXT.md) 的 Evaluation And Diagnostics 段。

## Why

现有低层链路会选择和组合 Agent，但它的目标不是生成可供研究的完整行为明细。策略研究员需要横向比较不同 epoch 中的全部子 Agent，观察它们在不同 Label、数据文件、Initial-action 和形态识别窗口中的动作、策略二阶形态与盈亏，同时不能改变任何既有脚本、产物格式或下游逻辑。

## What Changes

新增单一入口 `test_agents_indexs.py`。一次运行面向一个模型参数目录，扫描其直接子目录中的全部 `epoch_<N>/trained_model.pkl`，使用一套共享动作空间，执行完整的 `epoch × bin_index × label × df_path × initial_action` 评估全集。

该入口自行生成按 epoch 分区的英文逐步 Detail CSV，并在同一次运行中生成窗口明细、形态展开、覆盖率、分类诊断、六个聚合视图和分析 manifest。所有新文件只写入一个必需且隔离的 `--output_dir`；已有同名输出时失败，避免混合分析运行。

K 线形态、策略二阶形态、窗口划分、PnL 归因、展开与聚合规则沿用原双分类器 spec。需求中的“不同场景”规范为不同 `window_id`；`initial_action` 是窗口身份的一部分。

## Capabilities

### New Capabilities

- `fineft-agent-pattern-analysis`：独立执行完整 Agent 形态评估，生成可追溯的逐步、窗口、展开、诊断和聚合数据。

### Modified Capabilities

- 无。既有数据预处理、单 Agent 测试、Agent 选择、potential model 和下游路由能力均不修改。

## Impact

- 新增一个自包含的评估与分析入口及其测试。
- 新增独立输出目录中的分析数据契约。
- 不修改 `muti_contract_scale_save.py`、`test_agent_index.py` 或 `FineFT_single_agent_with_different_position.py`。
- 不读取既有 Detail CSV 或 Selection Manifest，不生成 `is_selected`，不写模型 checkpoint 目录或既有选择结果目录。
- `contract` 与原始 `volume` 是现有 Reward/Execution 数据中的必需字段；缺失时新入口立即失败。

## Success Criteria

- 完整覆盖指定模型参数目录下所有可用 epoch、全部 bin、全部 Label、全部数据文件和全部 Initial-action。
- 每个行为轨迹的 timestep 从 0 开始、唯一、连续，所有完整窗口均有唯一 `window_id`。
- 逐步 Detail 按 epoch 分文件，其余分析表跨 epoch 汇总，全部使用英文机器列名。
- 每个窗口只保存一次 PnL；展开和 Initial-action 聚合不放大账户盈亏。
- 全部新产物写入隔离目录且有指纹记录；既有代码、产物和下游行为保持不变。
