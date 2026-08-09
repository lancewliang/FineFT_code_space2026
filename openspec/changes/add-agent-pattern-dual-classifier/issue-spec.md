# Agent 形态分析：独立全量采集与双分类器窗口数据

Label: `ready-for-agent`

## Problem Statement

现有低层 Agent 测试和选择链路服务于单 Agent 评估、Agent 选择与 potential model 组合，不提供一个隔离、完整、可重现的研究数据采集入口。策略研究员需要比较同一实验参数目录中不同 epoch 版本的全部子 Agent，观察它们在不同 Label、数据文件、Initial-action 和形态识别窗口中的行为、策略二阶形态与盈亏。

旧方案依赖既有 Detail CSV，并要求修改 Scale Save、单 Agent 测试和 Selection Manifest 相关链路。这会改变既有产物契约并可能影响下游。新需求必须保持既有 Python、文件格式、选择逻辑、模型组合和下游路由完全不变，只通过一个新的旁路入口采集事实并生成新的分析数据。

## Solution

提供一个自包含的 `test_agents_indexs.py` 入口。一次运行只分析一个模型参数目录，扫描其直接 `epoch_<N>` 子目录中的可用模型，并使用一套共享动作空间执行完整的 `epoch × bin_index × label × df_path × initial_action` 组合。入口自行回测并按 epoch 写出新的逐步 Detail CSV，再从这些内存中或本次运行生成的事实计算形态识别窗口、K 线形态、策略二阶形态、窗口 PnL、展开数据、覆盖统计、分类诊断、六个聚合视图和可重现 manifest。

所有新产物使用英文机器列名，只写入必需的独立输出目录。入口不修改、导入或消费既有单 Agent 测试专用代码及其产物，不读取 Selection Manifest，不写 `is_selected`，也不接入任何下游选择或路由逻辑。

## User Stories

1. As a 策略研究员, I want 比较一个实验中全部 epoch 版本, so that 我能观察 Agent 策略随训练版本如何变化。
2. As a 策略研究员, I want 测试每个 epoch 的全部 bin index, so that 分析不会只覆盖当前被选择的子 Agent。
3. As a 策略研究员, I want 覆盖全部 Label, so that 每个子 Agent 在不同市场动态分片上的行为都可比较。
4. As a 策略研究员, I want 覆盖每个 Label 下的全部数据文件, so that 分析不会被单个 Market Dynamic Segment 偏置。
5. As a 策略研究员, I want 覆盖全部 Initial-action 情景, so that 起始仓位对 Agent 行为和盈亏的影响能够被观测。
6. As a 策略研究员, I want 让相同市场区间在不同 Initial-action 下拥有不同 window id, so that 不同反事实情景不会被错误合并。
7. As a 策略研究员, I want 每个 epoch 都有独立的逐步 Detail 文件, so that 大规模数据可以按版本检查和处理。
8. As a 策略研究员, I want 逐步 Detail 包含版本、子 Agent、Label、合约、数据文件、Initial-action 和 timestep, so that 任意交易行为都能追溯到完整评估身份。
9. As a 策略研究员, I want Detail 同时记录市场、动作、执行前后仓位、奖励、手续费、滑点及已实现和浮动盈亏, so that 后续窗口归因可由逐步事实复核。
10. As a 策略研究员, I want 新产物统一使用英文机器列名, so that 列名稳定且不会出现中文映射碰撞。
11. As a 策略研究员, I want Label 1 至 Label 5 按连续、不重叠的 20 步窗口分析, so that 不同 Agent 使用相同的普通形态归因单位。
12. As a 策略研究员, I want Label 0 和 Label 6 的完整行为轨迹各自成为一个涨跌停事件窗口, so that 极短涨跌停样本不会被不合理切分。
13. As a 策略研究员, I want 普通轨迹不足 20 步的尾部不产生形态窗口, so that 分类器只消费满足定义的完整窗口。
14. As a 策略研究员, I want 被丢弃尾部的步数和 PnL 被记录, so that 数据损失不是静默的。
15. As a 策略研究员, I want 每个形态识别窗口具有确定且唯一的 window id, so that 明细、展开和聚合数据可以稳定关联。
16. As a 策略研究员, I want window id 只由评估身份、数据来源、Initial-action 和窗口边界决定, so that 调阈值或重算 PnL 不会改变窗口身份。
17. As a 策略研究员, I want 每个窗口获得一个单选 K 线形态, so that 行情侧分类可以互斥统计。
18. As a 策略研究员, I want 每个窗口可以命中多个策略二阶形态, so that 兼容的 Agent 行为模式不会被强制互斥。
19. As a 策略研究员, I want 未命中规则的窗口保留明确的未分类哨兵, so that 分类覆盖率可以被诊断。
20. As a 策略研究员, I want 每个窗口同时输出 gross PnL 和 net PnL, so that 手续费前策略表现与手续费后账户表现均可分析。
21. As a 策略研究员, I want 滑点只作为诊断字段而不从 net PnL 重复扣除, so that 账户盈亏不会被重复惩罚。
22. As a 策略研究员, I want window table 每个 window id 只出现一行, so that 策略多选不会复制窗口 PnL。
23. As a 策略研究员, I want expanded table 展开 K 线和策略形态组合, so that 后续可以直接构建 7×6 条件分析。
24. As a 策略研究员, I want expanded table 明确禁止用于账户总 PnL 求和, so that 多选展开不会放大收益或亏损。
25. As a 策略研究员, I want coverage report 显示预期和实际评估组合, so that 全量采集承诺可以被验证。
26. As a 策略研究员, I want classifier diagnostics 报告形态分布、未分类率和 PnL 区分度, so that 分类阈值可以基于真实数据人工标定。
27. As a 策略研究员, I want 类别零命中或未分类率偏高只产生诊断告警, so that 分类分布不会被固定配额驱动。
28. As a 策略研究员, I want 获得 K 线、策略和交叉形态的 Scenario 级汇总, so that 每个 Initial-action 反事实情景能够独立分析。
29. As a 策略研究员, I want 获得 K 线、策略和交叉形态的 triple 级汇总, so that 不同版本子 Agent 可以横向比较。
30. As a 策略研究员, I want triple 汇总对实际命中形态的 Initial-action 情景等权平均, so that 长轨迹或多窗口情景不会天然获得更大权重。
31. As a 策略研究员, I want 未命中某形态的情景不被伪造成零 PnL, so that形态条件表现不会被人为稀释。
32. As a 策略研究员, I want 不同 Initial-action 情景的 PnL 不被直接相加, so that 反事实回测不会被误解为多个独立账户。
33. As a 审计者, I want manifest 记录模型、数据、动作空间、窗口和阈值配置, so that 每次分析可以被解释和复现。
34. As a 审计者, I want manifest 记录全部输入和输出文件的大小及 SHA-256, so that 分析数据身份可以验证。
35. As a 维护者, I want 一次运行只对应一个模型参数目录, so that 相同 epoch 编号不会跨实验发生身份碰撞。
36. As a 维护者, I want 所有 epoch 共享一套显式动作空间配置, so that仓位相关策略规则有权威且一致的档位语义。
37. As a 维护者, I want 缺少模型的 epoch 被记录而不是伪装成已分析, so that版本覆盖情况透明。
38. As a 维护者, I want 模型加载、必需数据列或行为轨迹连续性错误立即失败, so that 部分错误数据不会进入正式分析表。
39. As a 维护者, I want 全部新文件只写入显式的隔离输出目录, so that模型 checkpoint 和既有分析结果不会被污染。
40. As a 维护者, I want 输出目录已有同名产物时运行失败, so that 两次分析不会被静默混合。
41. As a 下游维护者, I want 既有 Scale Save、单 Agent 测试、Agent 选择和 Selection Manifest 完全不变, so that 当前下游逻辑继续使用原有契约。
42. As a 下游维护者, I want 新分析不读取 Selection Manifest 或生成 is_selected, so that研究数据采集与部署选择保持解耦。
43. As a 后续实现者, I want 所有新逻辑位于一个入口文件, so that 本次变更严格遵守单文件边界。
44. As a 后续实现者, I want 单文件内部的分类和聚合函数仍保持无 I/O 的纯函数接口, so that复杂规则可以被稳定测试。

## Implementation Decisions

- 只新增一个自包含入口文件；不修改任何既有 Python 文件，也不新增辅助分析模块。
- 新入口不得导入既有单 Agent 测试入口的专用函数，不得读取其 Detail 或 Aggregate 产物。项目通用的模型、环境和数据模块可以复用。
- 一次运行只接受一个模型参数目录；只扫描直接 `epoch_<N>` 子目录。存在模型文件的 epoch 进入候选全集，缺少模型文件的 epoch 进入覆盖报告，模型加载失败则运行失败。
- 候选全集为 `可用 epoch × 全部 bin_index × 全部 label × 全部 df_path × 全部 initial_action`。Selection Manifest 不参与发现、过滤、标记或输出。
- 一次运行内所有 epoch 共享 CLI 提供的动作空间参数。入口按环境公式构造完整 Position Level，并在创建环境后校验一致性；动作空间写入分析 manifest。
- `contract` 和原始 `volume` 是现有 Reward/Execution 数据的必需字段。缺失、空值或非有限值立即失败；不修改 Scale Save，不使用 `log_volume_origin` 回退。
- 行为轨迹由 `(epoch, label, bin_index, contract, df_path, initial_action)` 唯一确定，按 timestep 排序。timestep 必须从 0 开始、唯一、连续且为非负整数。
- Label 1 至 Label 5 使用 20 步连续、不重叠窗口，只生成完整窗口。Label 0 和 Label 6 的每条完整行为轨迹各生成一个涨跌停事件窗口。
- window id 由 `label, epoch, bin_index, contract, df_path, initial_action, window_index, start_timestep, end_timestep` 的规范 JSON 计算 SHA-256。形态、PnL、阈值、输出目录和绝对路径不进入哈希。
- K 线形态继续使用单选分类：KX1、KM1、KT2、KM3、KT1、KT3、KM2 和未分类哨兵。KX1 只用于涨跌停事件窗口；普通窗口按 `KM1 → KT2 → KM3 → KT1 → KT3 → KM2 → 未分类` 判定。
- 策略二阶形态继续使用多选分类：ST1、ST2、ST3、SM1、SM2、SM3。兼容规则可同时命中；全部未命中时保留策略未分类哨兵。
- 分类阈值、严格事件顺序、多空对称、最小样本数、ST1/SM3 同事件冲突和各形态定义沿用双分类器窗口模型的既有决策。
- `gross_pnl = realized_pnl_sum + unrealized_pnl_end - unrealized_pnl_before_start`；`net_pnl = gross_pnl - commission_fee_sum`。首窗口浮动 PnL 基线为 0，后续窗口基线取前一 timestep 的浮动 PnL。slippage 仅诊断，不重复扣除。
- 新入口强制接受独立输出目录。所有输出只写入该目录；不写模型目录和既有选择结果目录。发现同名目标文件时在写入前失败。
- 逐步 Detail 按 epoch 分区，路径形态为 `step_detail/agent_pattern_step_detail_epoch_<N>.csv`。其余 CSV 跨全部 epoch 汇总。
- 所有新增 CSV 使用英文机器列名和稳定列顺序。
- 跨 epoch 固定输出文件为 `agent_pattern_window_table.csv`, `agent_pattern_expanded_table.csv`, `agent_pattern_coverage_report.csv`, `agent_pattern_classifier_diagnostics.csv`, `agent_pattern_kline_scenario_summary.csv`, `agent_pattern_kline_triple_summary.csv`, `agent_pattern_strategy_scenario_summary.csv`, `agent_pattern_strategy_triple_summary.csv`, `agent_pattern_cross_scenario_summary.csv`, `agent_pattern_cross_triple_summary.csv`, `analysis_manifest.json`。

### Output Schemas

- Window table：`label, epoch, bin_index, contract, df_path, initial_action, window_index, start_timestep, end_timestep, start_timestamp, end_timestamp, step_count, window_id, kline_patterns, strategy_patterns, realized_pnl_sum, unrealized_pnl_before_start, unrealized_pnl_end, commission_fee_sum, slippage_sum, gross_pnl, net_pnl`。
- Expanded table：`label, epoch, bin_index, contract, df_path, initial_action, window_index, start_timestep, end_timestep, window_id, kline_pattern, strategy_pattern, gross_pnl, net_pnl`。
- Coverage report：`record_type, epoch, label, bin_index, contract, df_path, initial_action, expected_count, observed_count, coverage_ratio, status, window_count, dropped_tail_steps, dropped_tail_gross_pnl, dropped_tail_net_pnl, message`。
- Classifier diagnostics：`scope, label, epoch, bin_index, pattern_axis, kline_pattern, strategy_pattern, is_unclassified, window_count, window_ratio, total_net_pnl, pnl_p25, pnl_p50, pnl_p75, pnl_median_range, warning_code, warning_message`。
- Kline Scenario summary：`label, epoch, bin_index, contract, df_path, initial_action, kline_pattern, total_net_pnl, window_count, pnl_p25, pnl_p50, pnl_p75`。
- Kline triple summary：`label, epoch, bin_index, kline_pattern, mean_initial_action_total_net_pnl, mean_initial_action_window_count, mean_initial_action_pnl_p25, mean_initial_action_pnl_p50, mean_initial_action_pnl_p75, observed_initial_action_count, expected_initial_action_count, initial_action_coverage_ratio`。
- Strategy Scenario summary：`label, epoch, bin_index, contract, df_path, initial_action, strategy_pattern, total_net_pnl, window_count, pnl_p25, pnl_p50, pnl_p75`。
- Strategy triple summary：`label, epoch, bin_index, strategy_pattern, mean_initial_action_total_net_pnl, mean_initial_action_window_count, mean_initial_action_pnl_p25, mean_initial_action_pnl_p50, mean_initial_action_pnl_p75, observed_initial_action_count, expected_initial_action_count, initial_action_coverage_ratio`。
- Cross Scenario summary：`label, epoch, bin_index, contract, df_path, initial_action, kline_pattern, strategy_pattern, total_net_pnl, window_count, pnl_p25, pnl_p50, pnl_p75`。
- Cross triple summary：`label, epoch, bin_index, kline_pattern, strategy_pattern, mean_initial_action_total_net_pnl, mean_initial_action_window_count, mean_initial_action_pnl_p25, mean_initial_action_pnl_p50, mean_initial_action_pnl_p75, observed_initial_action_count, expected_initial_action_count, initial_action_coverage_ratio`。
- Analysis manifest 顶层键：`schema_version, dataset_name, experiment_name, model_root, data_root, evaluation_config, action_space, window_config, classifier_thresholds, epochs_discovered, epochs_analyzed, epochs_missing_model, candidate_universe, input_files, output_files, warnings`。文件指纹项固定为 `logical_path, size_bytes, sha256`；manifest 不记录自身指纹。

- `kline_patterns` 是 JSON 单元素数组，`strategy_patterns` 是 JSON 多选数组。Expanded table 对数组做笛卡尔展开，唯一键是 `(window_id, kline_pattern, strategy_pattern)`。
- Coverage report 的 `record_type` 支持 epoch 和 trajectory；status 至少支持 complete、missing_model 和 failed。
- Classifier diagnostics 使用长表；pattern axis 支持 kline、strategy 和 cross，scope 支持 overall、label、epoch 和 triple。策略多选导致 strategy/cross 的 window ratio 合计可超过 1。
- Scenario 汇总键包含 contract、df path 和 Initial-action。每组输出 total net PnL、window count 以及窗口 net PnL 的 25/50/75 分位数。
- Triple 汇总只对实际命中目标形态的 Scenario 统计做算术平均，输出 `mean_initial_action_*`。未命中的 Scenario 不伪造零值；同时报告 observed count、expected count 和 coverage ratio。
- 未分类哨兵保留在 Window、Expanded 和 diagnostics 中，但不进入六个正式 summary。
- Analysis manifest 记录模型、验证数据和全部输出的逻辑相对路径、字节数与 SHA-256；绝对输出目录不构成数据身份。

## Testing Decisions

- 主测试 seam 是单文件入口内部的两个纯分类函数：给定完整窗口序列、动作空间与阈值，断言单一 K 线形态和策略形态集合。测试只观察输入输出，不断言中间特征实现。
- 纯函数测试覆盖全部 7 类正式 K 线形态、6 类正式策略形态、两个未分类哨兵、多选、优先级、严格事件时序、多空对称、最小样本数和阈值上下边界。
- K 线测试重点覆盖 KT2/KM3 优先于 KT1、KX1 事件窗口、KT3 加速对称性、KM1 端点排除、KM2 往返计数和 KM3 量价背离。
- 策略测试重点覆盖 ST2+ST3 兼容多选、ST1/SM3 同事件冲突、不同事件共存、仓位档位参数化、SM1 涨跌停对称性、SM2 相邻档位和 SM3 背离过滤。
- 窗口与 PnL 纯转换测试覆盖普通 20 步非重叠切窗、涨跌停完整轨迹窗口、尾部统计、window id 稳定性、首/后续窗口浮动 PnL 边界及手续费和滑点语义。
- 聚合纯转换测试覆盖 JSON 数组展开、Window 唯一性、Expanded PnL 不作为账户总额、Scenario 不相加、命中 Scenario 等权平均、未命中不补零及覆盖率计算。
- 一个端到端 CLI smoke test 使用多个小型 epoch、多个 bin、多个 Label、多个数据文件和全部 Initial-action，验证完整评估组合、按 epoch Detail 分区、跨 epoch 分析表、固定 schema、所有文件生成和 manifest 指纹。
- 失败测试覆盖输出文件碰撞、缺失 contract/volume、非有限市场值、非法 timestep、模型加载失败、动作空间不一致、行为轨迹缺失和文件指纹不一致。
- 硬验收只覆盖可证明不变量：schema、唯一键、完整候选与 Initial-action 覆盖、PnL 守恒、确定性、指纹和合成边界测试。真实数据类别占比、未分类率和 PnL 区分度只作诊断。
- 测试风格沿用仓库中既有 Agent 选择纯转换测试、低层测试入口 smoke test 和数据切片测试使用的 pytest、合成输入与临时目录模式。

## Out of Scope

- 修改 Scale Save、Reward/Execution 列、State Feature 或数据切片逻辑。
- 修改或复用既有单 Agent 测试入口及其 Detail/Aggregate 产物。
- 修改 Agent 选择入口、Selection Manifest、potential model 或任何既有输出 schema。
- 使用 Selection Manifest 过滤候选或生成 is selected 标记。
- 将形态结果接入 Meta Router、Stage III 或任何下游执行逻辑。
- 修改训练、Q 网络输入、已训练权重、动作空间或 Labeling Method。
- 跨多个模型参数目录合并一次分析运行；不同实验应使用独立运行和独立输出目录。
- 把不同 Initial-action 情景当作独立账户直接相加。
- 从 Expanded table 跨策略形态汇总账户总 PnL。
- 为真实数据强制类别配额或以类别占比作为硬验收条件。

## Further Notes

- 需求中的“不同场景”是不同 `window_id`，规范术语为形态识别窗口；不要与 Initial-action 情景混用。
- 单文件要求是明确的交付边界。实现可在同一文件中定义小型纯函数，但不得拆出新的分析模块。
- 当前工作区中的旧 Detail 可能缺少 volume；它们不属于本能力的数据来源。新入口只消费当前评估数据，且把 contract 和原始 volume 作为硬前置条件。
- Detail 按 epoch 分区是数据规模决策；Window、Expanded、diagnostics 和 summary 跨 epoch 输出是版本比较决策。
- 新产物没有现有下游消费者；稳定 schema 的目的在于形成可供未来聚合研究使用的明细事实契约，而不是改变当前系统行为。
