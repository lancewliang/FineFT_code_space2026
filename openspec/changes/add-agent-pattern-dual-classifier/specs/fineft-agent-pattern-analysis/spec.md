# fineft-agent-pattern-analysis Delta

## ADDED Requirements

### Requirement: Pattern analysis SHALL be isolated from every existing production and selection artifact

系统 SHALL 通过一个全新的自包含入口采集 Agent 行为并生成全部形态分析产物，不得修改、导入或消费既有 Scale Save、单 Agent 测试、Agent 选择或 Selection Manifest 专用链路。

#### Scenario: Existing code and outputs remain unchanged

- **WHEN** 实施和运行 Agent 形态分析
- **THEN** 既有 Python 文件和既有输出 schema SHALL 保持不变
- **AND** 新入口 SHALL NOT 读取既有 Detail CSV、Aggregate CSV 或 Selection Manifest
- **AND** 新入口 SHALL NOT 生成 `is_selected`
- **AND** 新入口 SHALL NOT 写入模型 checkpoint 目录或既有选择结果目录

#### Scenario: New outputs use one isolated directory

- **GIVEN** 调用方提供独立 `output_dir`
- **WHEN** 新入口开始运行
- **THEN** 全部新产物 SHALL 只写入该目录
- **AND** 任一目标文件已经存在时 SHALL 在正式评估开始前 fail fast
- **AND** 不同分析运行 SHALL 使用不同输出目录

### Requirement: One analysis run SHALL evaluate one complete model-version universe

系统 SHALL 在单个模型参数目录内执行全部可用 epoch、全部 bin index、全部 Label、全部数据文件和全部 Initial-action 的完整评估组合。

#### Scenario: One parameter directory defines version identity

- **GIVEN** `model_root` 指向单个模型参数目录
- **WHEN** 系统发现模型版本
- **THEN** 系统 SHALL 只扫描直接子目录 `epoch_<N>`
- **AND** `epoch` SHALL 作为该次运行中的模型版本标识
- **AND** 不得在一次运行中混合多个参数目录

#### Scenario: Missing and invalid models are distinguished

- **WHEN** `epoch_<N>` 缺少模型文件
- **THEN** 系统 SHALL 在 Coverage Report 中记录 `missing_model` 并跳过该 epoch
- **WHEN** 模型文件存在但加载失败
- **THEN** 系统 SHALL fail fast，不得把该版本记为已分析

#### Scenario: Complete candidate universe is executed

- **WHEN** 系统评估一个可用 epoch
- **THEN** 它 SHALL 执行该 epoch 的 `全部 bin_index × 全部 label × 全部 df_path × 全部 initial_action`
- **AND** 候选发现和执行 SHALL 与 Selection Manifest 无关
- **AND** 任一预期行为轨迹缺失 SHALL fail fast 并报告完整身份

### Requirement: All epochs SHALL share one authoritative action space

系统 SHALL 从本次运行的 CLI 配置构造共享动作空间，并用它确定全部 Initial-action 和仓位相关策略语义。

#### Scenario: Shared action configuration is validated

- **WHEN** 系统创建任一 epoch 的评估环境
- **THEN** 环境动作空间 SHALL 与 CLI 的 `max_holding_number`, `position_choices`, `leverage_choices` 一致
- **AND** 系统 SHALL 按环境公式生成完整有序 `position_levels`
- **AND** 任一 epoch 的环境动作空间不一致时 SHALL fail fast
- **AND** 完整动作空间 SHALL 记录在 Analysis Manifest

#### Scenario: All Initial-actions are evaluated

- **WHEN** 系统生成一个 `(epoch, label, bin_index, contract, df_path)` 的行为轨迹集合
- **THEN** 每个共享动作空间中的 Initial-action SHALL 恰好执行一次
- **AND** 期望 Initial-action 集合 SHALL NOT 从已观测结果反推

### Requirement: The new evaluator SHALL write complete English step Detail facts per epoch

系统 SHALL 自行执行回测并按 epoch 写出逐时间步 Detail CSV，不得依赖旧 Detail 产物。

#### Scenario: Detail rows contain analysis facts and identity

- **WHEN** 新入口执行一个 validation step
- **THEN** Detail row SHALL 使用英文机器列名
- **AND** row SHALL 包含 `epoch, label, bin_index, contract, df_path, initial_action, timestep`
- **AND** row SHALL 包含原始 `volume`, `mark_price`, 动作、目标仓位、执行前后仓位、奖励、手续费、滑点、已实现 PnL 和浮动 PnL
- **AND** `contract` 与原始 `volume` 缺失、空值或非有限时 SHALL fail fast
- **AND** 系统 SHALL NOT 使用 `log_volume_origin` 替代原始 `volume`

#### Scenario: Step Detail is partitioned by epoch

- **WHEN** epoch `N` 的评估完成
- **THEN** 系统 SHALL 写出 `step_detail/agent_pattern_step_detail_epoch_<N>.csv`
- **AND** 每个已分析 epoch SHALL 恰好对应一个 Detail 文件
- **AND** 其他分析表 SHALL 跨全部已分析 epoch 汇总

#### Scenario: Timestep is authoritative

- **WHEN** 系统构建 `(epoch, label, bin_index, contract, df_path, initial_action)` 行为轨迹
- **THEN** timestep SHALL 从 0 开始、唯一、连续且为非负整数
- **AND** 任一重复、缺口或负值 SHALL fail fast
- **AND** 全局处理顺序 SHALL NOT 改变窗口、分类或 PnL 结果

### Requirement: Kline Pattern classifier SHALL return one deterministic market pattern per window

系统 SHALL 对 mark price 和原始 volume 序列返回单一 K 线形态，并按 `KX1 → KM1 → KT2 → KM3 → KT1 → KT3 → KM2 → 未分类` 解决多规则命中。

#### Scenario: Ordinary labels use complete non-overlapping windows

- **GIVEN** Label 属于 Label 1 至 Label 5
- **WHEN** 系统切分一条行为轨迹
- **THEN** 系统 SHALL 只产生长度 20、步长 20 的完整窗口
- **AND** 尾部不足 20 步 SHALL NOT 产生形态行
- **AND** Coverage Report SHALL 记录 `dropped_tail_steps`, `dropped_tail_gross_pnl`, `dropped_tail_net_pnl`

#### Scenario: Limit labels use one whole-trajectory event window

- **GIVEN** Label 为 Label 0 或 Label 6
- **WHEN** 系统分类该行为轨迹
- **THEN** 整条轨迹 SHALL 产生恰好一个事件窗口
- **AND** `kline_patterns` SHALL 等于 `["KX1"]`
- **AND** 达到策略规则最小样本数时 SHALL 扫描整条轨迹

#### Scenario: Kline priority preserves specific patterns

- **GIVEN** 同一窗口同时满足 KT1 与 KT2 或 KM3
- **WHEN** 分类器返回结果
- **THEN** KT2 SHALL 优先于 KM3 和 KT1
- **AND** KM3 SHALL 优先于 KT1
- **AND** 输出 SHALL 始终只有一个 K 线形态

#### Scenario: Z-score convention is stable

- **WHEN** 任一规则计算 `z_price`
- **THEN** 它 SHALL 使用完整窗口均值和 `ddof=0` 标准差
- **AND** 标准差为零时所有依赖 z-price 的规则 SHALL NOT 命中

### Requirement: Strategy Pattern classifier SHALL use executed positions and allow compatible multi-label results

系统 SHALL 使用共享 Position Level、实际执行仓位、浮动 PnL、mark price 和原始 volume 识别 ST1、ST2、ST3、SM1、SM2 和 SM3。

#### Scenario: Compatible patterns co-exist without duplicating window PnL

- **GIVEN** Agent 在突破回踩时同向加仓且加仓前浮动 PnL 大于 0
- **WHEN** 策略分类器返回结果
- **THEN** `strategy_patterns` SHALL 同时包含 ST2 和 ST3
- **AND** Window Table 中该窗口 PnL SHALL 仍只保存一次

#### Scenario: Position rules use the shared grid

- **WHEN** 分类器判断近满仓、相邻档位或同向加仓
- **THEN** 判定 SHALL 从共享 `position_levels` 推导
- **AND** 判定 SHALL NOT 写死仓位数量或从单条轨迹反推档位

#### Scenario: Event conflicts remain deterministic

- **GIVEN** 同一低量背离突破事件满足 ST1 候选
- **WHEN** 该事件发生顺突破方向开仓
- **THEN** 同一事件 SHALL NOT 同时命中 SM3
- **AND** 只有两个不同合格事件才允许同窗出现 ST1 和 SM3

#### Scenario: Unmatched windows remain observable

- **GIVEN** 所有具备最小样本数的策略规则均未命中
- **WHEN** 系统生成 Window row
- **THEN** `strategy_patterns` SHALL 等于 `["策略未分类"]`
- **AND** 该窗口 SHALL NOT 被丢弃

### Requirement: Window identity SHALL distinguish Initial-action and remain independent of analysis results

系统 SHALL 为每个形态识别窗口生成稳定且唯一的 window id。

#### Scenario: Window identity uses the fixed identity tuple

- **WHEN** 系统生成 window id
- **THEN** 它 SHALL 对 `label, epoch, bin_index, contract, df_path, initial_action, window_index, start_timestep, end_timestep` 的规范 JSON 计算 SHA-256
- **AND** 相同市场区间的不同 Initial-action SHALL 生成不同 window id
- **AND** 形态结果、PnL、阈值、绝对目录和输出位置 SHALL NOT 进入哈希
- **AND** `df_path` SHALL 使用规范相对 POSIX 路径

### Requirement: Window PnL SHALL be conserved once before any expansion

系统 SHALL 用执行账本计算每个窗口唯一的毛盈亏和净盈亏。

#### Scenario: Window PnL uses realized and boundary unrealized values

- **WHEN** 系统计算闭区间 `[start, end]`
- **THEN** `gross_pnl` SHALL 等于 `realized_pnl_sum + unrealized_pnl_end - unrealized_pnl_before_start`
- **AND** `net_pnl` SHALL 等于 `gross_pnl - commission_fee_sum`
- **AND** 首窗口的浮动 PnL 基线 SHALL 为 0，后续窗口 SHALL 取前一 timestep 的浮动 PnL
- **AND** slippage SHALL 只作诊断，不得再次扣除

#### Scenario: Expanded rows do not define account total

- **GIVEN** 一个窗口命中多个策略形态
- **WHEN** 系统生成 Window 和 Expanded tables
- **THEN** Window Table SHALL 对 window id 只保存一行 PnL
- **AND** Expanded Table SHALL 允许为条件分析复制 PnL
- **AND** Expanded Table SHALL NOT 被跨策略形态求和为账户总 PnL

### Requirement: New analysis artifacts SHALL use the fixed English schemas

系统 SHALL 生成固定文件清单、英文表头、列顺序和唯一键。

#### Scenario: Window and Expanded schemas are fixed

- **WHEN** 系统写出 `agent_pattern_window_table.csv`
- **THEN** 列 SHALL 依次为 `label, epoch, bin_index, contract, df_path, initial_action, window_index, start_timestep, end_timestep, start_timestamp, end_timestamp, step_count, window_id, kline_patterns, strategy_patterns, realized_pnl_sum, unrealized_pnl_before_start, unrealized_pnl_end, commission_fee_sum, slippage_sum, gross_pnl, net_pnl`
- **AND** 每个 window id SHALL 恰好一行
- **AND** 两个 patterns 字段 SHALL 是合法 JSON 数组
- **WHEN** 系统写出 `agent_pattern_expanded_table.csv`
- **THEN** 列 SHALL 依次为 `label, epoch, bin_index, contract, df_path, initial_action, window_index, start_timestep, end_timestep, window_id, kline_pattern, strategy_pattern, gross_pnl, net_pnl`
- **AND** `(window_id, kline_pattern, strategy_pattern)` SHALL 唯一

#### Scenario: Coverage schema is fixed

- **WHEN** 系统写出 `agent_pattern_coverage_report.csv`
- **THEN** 列 SHALL 依次为 `record_type, epoch, label, bin_index, contract, df_path, initial_action, expected_count, observed_count, coverage_ratio, status, window_count, dropped_tail_steps, dropped_tail_gross_pnl, dropped_tail_net_pnl, message`
- **AND** record type SHALL 支持 epoch 和 trajectory
- **AND** status SHALL 至少支持 complete、missing_model 和 failed

#### Scenario: Diagnostics schema is fixed

- **WHEN** 系统写出 `agent_pattern_classifier_diagnostics.csv`
- **THEN** 列 SHALL 依次为 `scope, label, epoch, bin_index, pattern_axis, kline_pattern, strategy_pattern, is_unclassified, window_count, window_ratio, total_net_pnl, pnl_p25, pnl_p50, pnl_p75, pnl_median_range, warning_code, warning_message`
- **AND** scope SHALL 支持 overall、label、epoch 和 triple
- **AND** pattern axis SHALL 支持 kline、strategy 和 cross
- **AND** 策略多选导致 strategy 或 cross 的 window ratio 合计大于 1 SHALL 被允许

### Requirement: Pattern summaries SHALL preserve Initial-action scenario semantics

系统 SHALL 先生成 Scenario 级统计，再对实际命中目标形态的 Scenario 等权平均为 Agent triple 统计。

#### Scenario: Six summary files have fixed schemas

- **WHEN** 系统生成 Kline Scenario summary
- **THEN** 列 SHALL 为 `label, epoch, bin_index, contract, df_path, initial_action, kline_pattern, total_net_pnl, window_count, pnl_p25, pnl_p50, pnl_p75`
- **WHEN** 系统生成 Strategy Scenario summary
- **THEN** 列 SHALL 为 `label, epoch, bin_index, contract, df_path, initial_action, strategy_pattern, total_net_pnl, window_count, pnl_p25, pnl_p50, pnl_p75`
- **WHEN** 系统生成 Cross Scenario summary
- **THEN** 列 SHALL 为 `label, epoch, bin_index, contract, df_path, initial_action, kline_pattern, strategy_pattern, total_net_pnl, window_count, pnl_p25, pnl_p50, pnl_p75`
- **WHEN** 系统生成对应 triple summary
- **THEN** 每个文件 SHALL 移除 contract、df path 和 Initial-action，保留对应形态键
- **AND** 指标列 SHALL 为 `mean_initial_action_total_net_pnl, mean_initial_action_window_count, mean_initial_action_pnl_p25, mean_initial_action_pnl_p50, mean_initial_action_pnl_p75, observed_initial_action_count, expected_initial_action_count, initial_action_coverage_ratio`

#### Scenario: Triple aggregation does not invent or add scenarios

- **GIVEN** 全部预期 Initial-action 行为轨迹均已执行
- **WHEN** 某形态只在部分 Scenario 命中
- **THEN** `mean_initial_action_*` SHALL 只对命中该形态的 Scenario 做算术平均
- **AND** 未命中 Scenario SHALL NOT 被补为零 PnL
- **AND** 不同 Initial-action Scenario 的 PnL SHALL NOT 被直接相加
- **AND** observed count、expected count 和 coverage ratio SHALL 被显式报告

#### Scenario: Unclassified sentinels remain diagnostic only

- **WHEN** Window Table 包含未分类哨兵
- **THEN** 哨兵 SHALL 保留在 Window、Expanded 和 Diagnostics
- **AND** 哨兵 SHALL NOT 进入任何正式 Kline、Strategy 或 Cross summary

### Requirement: Analysis Manifest SHALL make the isolated run reproducible

系统 SHALL 记录运行配置、候选全集和输入输出身份，但不得记录 Selection Manifest。

#### Scenario: Manifest keys and fingerprints are fixed

- **WHEN** 分析成功完成
- **THEN** `analysis_manifest.json` 顶层键 SHALL 为 `schema_version, dataset_name, experiment_name, model_root, data_root, evaluation_config, action_space, window_config, classifier_thresholds, epochs_discovered, epochs_analyzed, epochs_missing_model, candidate_universe, input_files, output_files, warnings`
- **AND** 每个文件指纹项 SHALL 包含 `logical_path, size_bytes, sha256`
- **AND** 模型、验证数据和全部生成 CSV SHALL 被记录
- **AND** manifest SHALL NOT 记录 Selection Manifest、is selected 或自身指纹
- **AND** 绝对输出目录 SHALL NOT 构成窗口或数据身份

### Requirement: Calibration SHALL remain diagnostic rather than quota-driven

系统 SHALL 报告真实数据形态分布和 PnL 区分度，但不得为满足固定类别占比自动调参。

#### Scenario: Distribution warnings do not fail hard acceptance

- **WHEN** 未分类率至少 30%、正式类别零命中或 PnL 区分度弱
- **THEN** 系统 SHALL 写入诊断告警而不是失败
- **AND** PnL 区分度 SHALL 定义为同一视图正式形态 `pnl_p50` 的最大值减最小值
- **AND** 系统 SHALL NOT 自动调整阈值以满足类别配额
