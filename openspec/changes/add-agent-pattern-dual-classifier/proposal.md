# add-agent-pattern-dual-classifier

> 架构决策见 [ADR-0006：Agent 形态分类采用双轴窗口模型](../../../docs/adr/0006-agent-pattern-dual-classifier.md)。术语见 [CONTEXT.md](../../../CONTEXT.md) 的 Evaluation And Diagnostics 段。本 spec 是该 ADR 的落地规格。

## Why

当前低层 agent 选择链路（`FineFT_single_agent_with_different_position.py`）只能按 `trans_reward_mean` 等 reward 类指标给每个 `(label, epoch, bin_index)` agent triple 排序，无法回答一个对策略研究和 agent 调度都很关键的问题：

**"哪个 agent 在哪个 label 的哪个 K 线形态下、用了什么二阶策略、盈利如何？"**

用户（策略研究员 / agent 调度设计者）面对一堆候选 agent triple，只有"平均回报"这一个粗粒度信号，无法区分：

- 同样在 `label_4`（持续上涨）下盈利的两个 agent，一个是"突破即时重仓"（大赚大亏、滑点敏感），另一个是"回调加仓"（胜率高、易踏空单边）——它们适用行情和风险特征完全不同，但 reward 均值可能相同。
- 一个 agent 是否同时擅长多种形态（"一个 agent 可以属于多种类型"），以便在 Meta Router 调度时按行情形态匹配，而不是只按 label 匹配。

缺乏这层"形态 × 策略 × 盈亏"的明细，用户就无法做基于形态的 agent 归类、无法验证 Semantic Guard 是否真的把 agent 限制在合规方向、也无法为 Stage III 路由提供比 label 更细的匹配维度。

## What Changes

构建**两个正交的分类器**，在现有 Detail CSV（`trading_action_detail_epoch_*.csv`）上离线计算，产出窗口级与展开级两份 Agent 形态数据文件，读法为"哪个 agent (epoch, bin) 在哪个 label 的哪个 K 线形态下用哪种策略盈利如何"。

1. **K 线形态分类器**（行情侧）：读 `mark_price` / `volume` 序列，识别 7 类行情二阶形态。
2. **策略形态分类器**（agent 侧）：读行为轨迹整体（含 `position_before` / `position_after` / `unrealized_pnl` / `mark_price` / `volume`），识别 6 类 agent 动作与行情的关系模式。
3. **两种数据粒度**：`agent_pattern_window_table.csv` 每个形态识别窗口恰好一行，`kline_patterns` 存 JSON 单元素数组（如 `["KT2"]`），`strategy_patterns` 存 JSON 多选数组（如 `["ST2", "ST3"]`），窗口盈亏只存一次；`agent_pattern_expanded_table.csv` 将两个数组展开为每个 `(window_id, kline_pattern, strategy_pattern)` 一行，供 7×6 形态分析使用。
4. **全候选覆盖**：扫描全部 `trading_action_detail_epoch_*.csv` 并分类其中所有 `(label, epoch, bin_index)` 候选；`selection_manifest.json` 只用于生成 `is_selected` 标记，不过滤候选。

两个分类器是纯函数，输入一个形态识别窗口的序列，输出形态标签。label_1~label_5 使用 N=20 的标准窗口；label_0/label_6 使用整条短轨迹构成的涨跌停事件窗口。明细表生成与聚合是薄 orchestrator 层。

## Capabilities

### New Capabilities

- `fineft-agent-pattern-analysis`：保留形态分析所需的评估数据与动作空间来源，生成 K 线/策略双轴窗口分类、盈亏归因、聚合视图和标定诊断。

### Modified Capabilities

- 无。

## Impact

- 数据链：commodity Scale Save 额外 passthrough 原始 `volume`，label 切片自然保留该列。
- 评估链：`test_agent_index.py` 的 Detail 产物增加 `contract`、原始 `volume` 和 epoch 动作空间 sidecar。
- 分析链：新增纯分类器、窗口/展开明细、聚合视图、覆盖率报告和可重现 manifest。
- 模型边界：不改变 `state_features.npy`、Q 网络输入、动作空间或已训练权重；需重生成数据集/切片并重跑评估，无需重新训练。

## User Stories

### 核心查询

1. 作为策略研究员，我想查看任意一个 `(label, epoch, bin_index)` agent triple 在 7 种 K 线形态下的盈亏分布，以便判断它擅长哪种行情。
2. 作为策略研究员，我想查看任意一个 agent triple 在 6 种策略形态下的盈亏分布，以便判断它用的是突破即时、回调加仓、还是金字塔递增等哪种二阶策略。
3. 作为策略研究员，我想把"K 线形态"和"策略形态"交叉成 7×6 表，看同一个 agent 在不同 (K 线形态, 策略形态) 组合上的盈亏，以便发现"突破行情下用回调加仓策略盈利"这类细粒度信号。
4. 作为策略研究员，我想知道一个 agent 是否同时命中多种策略形态（如回踩同向加仓且加仓前已有浮盈时同时命中 ST2 + ST3），以便确认"一个 agent 可以属于多种类型"。
5. 作为策略研究员，我想按 label 过滤明细表，看该 label 下所有 agent 的形态分布，以便对比同 label 内不同 agent 的策略差异。

### 形态识别

6. 作为策略研究员，我想让系统在 `label_1`~`label_5` 的 segment 内按 N=20 步不重叠窗口自动识别 K 线形态，以便把一段行情切成若干个形态同质的区间分别归因盈亏。
7. 作为策略研究员，我想让系统对 `label_0` / `label_6`（涨跌停）的 trajectory 直接标记为 KX1 涨跌停型，不在其上跑滑窗识别，以便尊重其极短长度（median=3 步）。
8. 作为策略研究员，我想让 `label_0` / `label_6` 的整条轨迹作为涨跌停事件窗口参与策略形态判定；样本不足的规则不得命中，全部规则均未命中时保留 `strategy_patterns=["策略未分类"]` 的窗口行。
9. 作为策略研究员，我想让 K 线形态单窗口单选、命中顺序为 `KX1→KM1→KT2→KM3→KT1→KT3→KM2→未分类`，以便回调和低量背离不会被普通突破吞没。
10. 作为策略研究员，我想让策略形态单窗口多选，以便一个窗口能同时命中 ST2 与 ST3 这类兼容组合。
11. 作为策略研究员，我想看到无法命中任何类的窗口被明确标为 K 线“未分类”或“策略未分类”，以便知道识别覆盖率。

### 盈亏归因

12. 作为策略研究员，我想让每行同时给出手续费前 `gross_pnl` 和手续费后 `net_pnl`，两者都覆盖窗口内全部执行步及窗口起止之间的浮动 PnL 变化，以便既能诊断策略本身，也能按真实账户净收益比较形态。
13. 作为策略研究员，我想让每个窗口只产生一行，K 线形态和策略形态都存为数组，窗口盈亏只保存一次，以便明细层不因策略多选而复制盈亏。
14. 作为策略研究员，我想在聚合时按需展开形态数组，但不跨策略形态汇总盈亏，以便比较各形态表现而不把同一窗口的盈亏误当成组合后的账户总盈亏。
15. 作为策略研究员，我想先查看每个 initial-action 情景在各形态组合下的总盈亏、窗口数和盈亏分位数，再查看实际命中该形态的情景等权平均及情景覆盖率，以便在不伪造零 PnL 的前提下做粗粒度对比。

### 与现有产物对齐

16. 作为 agent 调度设计者，我想让分类键 `(label, epoch, bin_index)` 与 `selection_manifest.json` 的部署产物对齐，以便明细表能直接 join 到已被部署的 agent。
16a. 作为 agent 调度设计者，我想保留全部候选 agent 的分类结果，并用 `is_selected` 区分当前部署 agent，以便既能分析当前选择，也能比较和构建候选池。
17. 作为 agent 调度设计者，我想把 `initial_action` 排除在分类键外（只作诊断切片轴），以便分类输出描述的是被部署的 agent 而非某种起步条件。
18. 作为 agent 调度设计者，我想把 segment（df_0..n）排除在分类维度外（只作 K 线数据来源文件），以便分类键不被文件管理维度污染。
19. 作为 agent 调度设计者，我想让明细表能与 `result_all.csv`（按 `df_path` 列）join 回原始 trajectory，以便追溯任意一行的原始逐 step 数据。

### 阈值标定与诊断

20. 作为策略研究员，我想用真实 Detail CSV 跑一遍分类器，看 7 类 K 线形态和 6 类策略形态各自的命中分布与诊断告警，以便人工判断阈值是否合理，而不是为满足固定类别占比而制造命中。
21. 作为策略研究员，我想在标定阶段查看每个仍参与判定的特征（如突破幅度、前后半窗 log-price 斜率比、回撤幅度、成交量下降比例）在真实窗口上的分布，以便基于证据调整阈值。
22. 作为策略研究员，我想在标定后用回归测试锁住阈值，以便后续重构不破坏分类语义。

### 可读性与可维护性

23. 作为后续维护者，我想让两个分类器是纯函数（输入序列 → 输出标签），不依赖 I/O，以便单独测试和复用。
24. 作为后续维护者，我想让明细表生成脚本只是薄 orchestrator（读 Detail CSV → 切窗口 → 调分类器 → 写表），不在其中塞判别逻辑，以便判别逻辑集中在分类器内可独立演进。
25. 作为后续维护者，我想让所有形态标签用前缀命名（KT/KM/KX + ST/SM），以便在表里短而不混淆。
26. 作为后续维护者，我想让分类器读行为轨迹整体（含 price/vol/pnl），而非只读 position_after，以便能判别"动作与行情的关系模式"而非纯动作形状。

## Implementation Decisions

### 架构与分类键

- 采用 [ADR-0006](../../../docs/adr/0006-agent-pattern-dual-classifier.md) 定义的双独立分类器架构。K 线形态（行情侧）与策略形态（agent 侧）正交，分别由两个分类器独立判定，构成 7×6 组合空间。
- **分类键** = `(label, epoch, bin_index)`，对齐 `selection_manifest.json` 的部署产物。`initial_action` 不进键（部署产物不带它，且 3 条起步强相关投票会重复计数），降级为诊断切片轴。`segment`（df_0..n）是 label 内文件切片，不构成分类维度，仅作为 K 线数据的来源文件。
- **分类范围**覆盖 `--model_root` 下全部 `epoch_*/trading_action_detail_epoch_*.csv` 的所有候选 triple，不以 `selection_manifest.json` 预先过滤。对 `(label, epoch, bin_index)` 与 manifest 条目精确匹配的窗口设置 `is_selected=true`，其余设置为 `false`；展开级文件继承该字段。
- **Selection manifest 逻辑归属**：在进行 triple 匹配前，必须校验 manifest 的 `dataset_name` / `experiment_name` 与 `model_root` 所表示的逻辑数据集/实验一致，`label_0`~`label_6` 各恰好一条，且每条 `epoch_path` 与 `model_path` 解析出的 epoch 一致。`model_path` 的逻辑末尾必须是该 epoch 下的 `trained_model.pkl`，对应 `model_root/epoch_<N>` 必须存在。校验比较数据集、实验、epoch 和文件末尾这些逻辑身份，不依赖机器绝对路径前缀；任何错配、重复、缺失或格式错误均立即失败。
- **Detail 覆盖完整性**：候选全集以实际存在的 Detail CSV 为准，不要求每个 checkpoint epoch 都已有明细。缺少未选 epoch 的 Detail CSV 只进入 `agent_pattern_coverage_report.csv` 并告警；manifest 中任一已选 triple 在 Detail rows 中不存在时立即失败。Detail 文件名中的 epoch、父目录 `epoch_<N>` 和 sidecar epoch 必须一致，发现重复或冲突时失败。
- **initial-action 聚合语义**：窗口级文件完整保留每个 `initial_action`。形态聚合先生成包含 `initial_action` 维度的情景统计，triple 级的每个形态组只对至少有一个窗口命中该形态的情景做算术平均，报告 `mean_initial_action_*`，不得将未命中形态的情景伪造为零 PnL。triple 级还必须输出 `observed_initial_action_count`、`expected_initial_action_count` 和 `initial_action_coverage_ratio`。每个 epoch 的期望 initial-action 集合由 sidecar 中的 `position_choices` 和 `leverage_choices` 按环境动作数公式生成，不从已观测 Detail rows 推断；每个 `(epoch, label, bin_index, contract, df_path)` 必须具有该 epoch 全部期望 Initial-action 的 Detail 行为轨迹，任一轨迹缺失时立即失败。
- 每个重新生成的 Detail CSV 伴随 `trading_action_detail_epoch_<N>.manifest.json`，至少记录 `epoch, max_holding_number, position_choices, leverage_choices, position_levels`。明细表生成器以 sidecar 为权威动作空间；可选 CLI override 只能用于校验，与 sidecar 不一致时失败。纯策略分类器显式接收 sidecar 中的完整有序仓位档位，不得写死或从单条轨迹反推动作空间。
- 输入 Detail CSV 允许现有中文表头及对应英文机器名，但读取后必须规范化为唯一的英文内部 schema；同一语义的中英文列同时存在且值冲突时失败。
- Detail CSV 的全局行序不构成输入契约。生成器按行为轨迹分组后必须以 `timestep` 升序排序，并要求每组 `timestep` 是从 0 开始、唯一且连续的整数序列；负值、重复或缺口均立即失败，不得在切窗前静默修复或重新编号。
- 窗口级文件 schema 必含 `label, epoch, bin_index, is_selected, contract, df_path, initial_action, window_index, start_timestep, end_timestep, window_id, kline_patterns, strategy_patterns, gross_pnl, net_pnl`。`window_id` 的 SHA-256 输入固定为 `(label, epoch, bin_index, contract, df_path, initial_action, window_index, start_timestep, end_timestep)` 的规范 JSON 表示；`df_path` 必须是相对 `valid/` 的规范 POSIX 路径。`is_selected`、形态结果、PnL、阈值配置和任何绝对目录不得进入哈希，使同一逻辑窗口在阈值、selection manifest 或数据根目录变化后仍保持同一 ID。
- 展开级文件使用标量列 `kline_pattern` 和 `strategy_pattern`，并以 `(window_id, kline_pattern, strategy_pattern)` 作为唯一键。窗口级文件用于账户盈亏计算，展开级文件用于单形态和 7×6 组合分析。

### 形态识别窗口

- `label_1`~`label_5` segment 内按 **N=20 步不重叠窗口**切分（步长 = N，无重叠）。不重叠设计避免盈亏重复计入。
- `label_1`~`label_5` segment 尾部不足 20 步的余数不产出形态明细；覆盖率报告必须按轨迹记录 `dropped_tail_steps`, `dropped_tail_gross_pnl`, `dropped_tail_net_pnl`，不得静默丢弃。
- `label_0` / `label_6` 不切 N=20 标准窗口；每条完整行为轨迹构成一个**涨跌停事件窗口**，K 线形态固定为 KX1。当该轨迹长度达到策略规则的最小样本数时，规则在整条轨迹上扫描，仍只产生一个事件窗口。
- 选 N=20 的依据：实测 label_1~5 trajectory 中位数 102~185 步，N=20 覆盖 KM2 箱体最小往返（~15~20 步）与 KM3 背离（2 个新高 ~10~20 步）；label_3/4/5 在 N=20 时无法识别率仅 8.3%。

### K 线形态分类器（7 类，单选）

- 7 类：KT1 突破即时型 / KT2 回调型 / KT3 加速型 / KM1 V/倒V反转型 / KM2 箱体型 / KM3 背离型 / KX1 涨跌停型。
- 命中顺序（互斥单选，命中即止）：`KX1 → KM1 → KT2 → KM3 → KT1 → KT3 → KM2 → 未分类`。KT2 与 KM3 均优先于 KT1，以避免回调和低量背离被普通突破吞没。
- KX1 仅由 label 决定（label ∈ {0,6} → KX1），不进窗口识别。
- 各类判别公式与提议阈值（需标定，见 Testing Decisions）：
  - **Z-score 通用语义**：`z_price` 用完整形态识别窗口的 `mark_price` 均值和总体标准差（`ddof=0`）计算。标准差为零时，所有依赖 `z_price` 的规则均不命中。
  - **KM1 V/倒V反转**：极值点 `|z_price|≥2.0`，且极值点前后各至少保留 `min_leg_steps` 步（提议 3 步）；窗口起点到极值点的单边移动、极值点后的反向移动均至少达到 `min_leg_return`（提议 0.5%）。V 型识别下跌→上涨，倒 V 识别上涨→下跌，两方向对称；Z-score、双腿最小收益和最小边长均参数化。
  - **KT1 突破即时**：第 1~5 步建立基准区间；第 6~10 步内 `mark_price` 突破基准区间 max/min ±0.3%。突破后，窗口最终价格须从对应基准边界沿突破方向延伸至少 0.5%，且突破后的观测点中至少 80% 保持在基准边界的突破侧。基准区间自身不参与突破判定。
  - **KT2 回调**：第 1~5 步建立基准区间，第 6~10 步触发与 KT1 相同的突破；突破后必须先形成同向新极值，再反向回撤至少 0.3% 并回到基准突破边界 ±0.5% 范围，最后在窗口结束时从该边界沿突破方向再次延伸至少 0.5%。触发突破的观测点不能同时计作回踩。
  - **KT3 加速**：分别对前 10 步和后 10 步的 `log(mark_price)` 做线性拟合；两段斜率方向一致，后半段绝对斜率至少为前半段的 `acceleration_ratio` 倍（提议 `1.5`），且整个窗口 `|cum_ret|≥0.5%`。上涨与下跌加速对称判定，斜率倍数和累计收益阈值均参数化。
  - **KM2 箱体**：窗口 `std(log_return)≤return_std_threshold`（提议 0.3%），且所有 `mark_price` 位于窗口均价 ±`outer_band_ratio`（提议 0.5%）。下沿区定义为 `price≤mean×(1-touch_band_ratio)`，上沿区定义为 `price≥mean×(1+touch_band_ratio)`（`touch_band_ratio` 提议 0.25%）；忽略中间区域并压缩连续同区状态后，上下沿至少交替转换 `min_band_transitions` 次（提议 4 次，例如下→上→下→上→下）。波动率、内外沿和最小转换次数均参数化。
  - **KM3 背离**：与 SM3 共用价格—成交量背离检测器。第 1~5 步建立价格与成交量基准；第 6 步以后价格突破基准高/低点至少 `breakout_ratio`（提议 0.3%），突破时点及之前两步的成交量中位数相对基准期成交量中位数至少下降 `volume_drop_ratio`（提议 20%），且相对基准均价的价格移动至少达到 `min_price_move`（提议 0.5%）。新高与新低对称，三个阈值均参数化。
- 阈值均为提议值，落地后用真实数据标定（见 Testing Decisions）。

### 策略形态分类器（6 类，多选）

- 6 类：ST1 突破即时型 / ST2 回调加仓型 / ST3 金字塔递增型 / SM1 硬边界反转型 / SM2 离散网格调仓型 / SM3 背离过滤型。
- 单窗口**多选**：一个窗口可命中多类（如回踩同向加仓且加仓前已有浮盈时命中 ST2+ST3），结果作为一个数组写入该窗口唯一的明细行。
- 读行为轨迹整体（含 `position_before` / `position_after` / `unrealized_pnl` / `mark_price` / `volume`），判别 agent 动作与行情的关系模式，而非纯动作形状。
- 各类判别公式与提议阈值：
  - **ST1 突破即时**：价格按 KT1 的固定时序在第 6~10 步触发突破，agent 在突破当步或下一步从空仓直接开到近满仓（`position_before == 0` 且 `|position_after| / max_abs_position ≥ near_full_ratio`，提议 `near_full_ratio=0.8`），开仓方向与突破方向一致。持仓计数包含开仓当步，连续至少 10 个观测的执行后仓位必须同号；期间可减仓，但不要求持续近满仓。
  - **ST3 金字塔递增**：窗口内至少出现一次**盈利后同向加仓**。同向加仓定义为执行前后仓位同号且绝对仓位增加，同时要求执行加仓前的 `unrealized_pnl_before > 0`；多头和空头按绝对仓位对称判定，不写死具体档位。`0→非 0` 是开仓、仓位变号是反手，均不算同向加仓。`unrealized_pnl_before` 取同一行为轨迹前一行的执行后浮动盈亏，轨迹首步取初始值 0。
  - **ST2 回调加仓**：价格先按 KT2 的固定时序突破，随后回踩至突破位 ±0.5%；agent 在回踩当步或下一步执行与原突破方向一致的同向加仓，且窗口结束前价格再次沿突破方向延续。ST2 不要求加仓前盈利；若加仓前浮动盈亏同时大于 0，则同一窗口可同时命中 ST2 与 ST3。加仓判定从输入仓位档位集合计算，不绑定档位数量。
  - **SM1 硬边界反转**：普通 N=20 窗口中，`z_price≤-2.0` 时逆向开多/加多，或 `z_price≥2.0` 时逆向开空/加空；涨跌停事件窗口中，`label_0` 跌停时开多/加多，或 `label_6` 涨停时开空/加空。动作后要求 `|position_after| / max_abs_position ≥ near_full_ratio` 且绝对仓位较执行前增加。仅维持已有仓位、顺势开仓/加仓、减仓、平仓和反手均不命中。
  - **SM2 离散网格调仓**：仓位档位来自 Detail sidecar。`z_price≥0.5` 时仓位向空头方向移动一个相邻档位，`z_price≤-0.5` 时仓位向多头方向移动一个相邻档位；命中要求时间顺序上存在一对方向相反的高低侧调仓事件，不要求最终仓位回到窗口初始档位。允许经过 0 的相邻开平仓调节；跨多档跳仓和直接反手不计入。
  - **SM3 背离过滤**：窗口内存在价格创新高/低但成交量趋势下降的背离段，且该段 `|cum_ret_price|≥0.5%`。背离段内不得沿价格突破方向开仓或加仓；允许保持空仓、减少/平掉顺势仓位或建立逆向仓位。窗口的非背离时段必须至少发生一次有效仓位变化，以排除全程不交易的 agent。不使用背离段与全局 `|Δpos|` 均值比率。
- 多选下无命中顺序；各类独立判定，命中几个算几个。
- 冲突按具体事件而非整个窗口处理：同一个突破事件上，ST1 要求沿突破方向开仓，而 SM3 禁止在背离段沿突破方向开仓/加仓，因此二者不能归因于同一事件；只有窗口内存在两个不同的合格事件时，才允许同一窗口同时保存 ST1 与 SM3。
- 每条策略规则必须声明并检查自身所需的最小样本数。普通 N=20 窗口运行全部 6 类规则；涨跌停事件窗口中，ST1、ST2、SM3 最少需要 20 步，ST3 与涨跌停语义下的 SM1 最少需要 1 条执行记录，SM2 最少需要 2 条执行记录且价格方差非零。样本不足时不得外推或补齐；所有规则均未命中时，`strategy_patterns=["策略未分类"]`。该值是诊断哨兵，不计入 6 类策略形态。

### 盈亏归因

- `gross_pnl = sum(realized_pnl_step[start:end+1]) + unrealized_pnl[end] - unrealized_pnl_before_start`，表示手续费前盈亏。
- `net_pnl = gross_pnl - sum(commission_fee_step[start:end+1])`，表示手续费后账户净盈亏，也是所有形态绩效聚合的默认口径。环境的实际成交价值已经反映在 `realized_pnl_step` 中，`slippage_step` 只保留为诊断字段，不得再次从 `net_pnl` 扣除。
- 非首窗口的 `unrealized_pnl_before_start` 取同一行为轨迹的前一行；首窗口取测试配置的初始浮动 PnL `0`。涨跌停事件窗口使用相同边界。每个 `window_id` 只保存一行及一组 `gross_pnl`/`net_pnl`；策略多选不在明细层复制盈亏。

### 模块划分

- 新建分类器模块目录（建议在 `FineFT/analysis/` 下新建 `classify_agent/` 子包），包含：
  - K 线形态分类器（纯函数：窗口序列 → 单一 K 线形态标签）。
  - 策略形态分类器（纯函数：窗口序列 → 策略形态标签集合）。
  - 窗口级文件生成脚本（薄 orchestrator：读 Detail CSV → 按 triple 遍历 → label_0/6 以整条轨迹建立涨跌停事件窗口并标 KX1 / label_1~5 切 N=20 不重叠窗口 → 调两个分类器 → 写窗口级文件）。
  - 展开与聚合脚本（从窗口级文件展开两个形态数组，写展开级文件；再分别按 K 线形态、策略形态或 7×6 组合聚合）。
- 修改 commodity Scale Save，在处理同一份 split DataFrame 时将原始 `volume` 直接作为非模型 passthrough 列写出；不做二次 join，不将 `volume` 加入 `state_features.npy`。`contract` 已是 Reward/Execution 列，`slice_model.py` 无需修改。
- 修改 `test_agent_index.py` 的 Detail 行生成，写出 `contract`、原始 `volume` 及 epoch 级动作空间 sidecar。重新生成数据集/切片并重跑已有评估 epoch 以更新 Detail 产物，不涉及模型重新训练。
- 不修改 `FineFT_single_agent_with_different_position.py`（selection_manifest 不变），不修改形态分类键或模型动作空间。
- 不修改现有 `labeling_method` 配置（`slope` 已在 ADR-0006 落实，与本 spec 无关）。

### 数据来源

- CLI 使用三个独立路径参数：
  - `--model_root`：指向 `result/DiHFT/low_level/<dataset>/<experiment>/weights_advantage_pretrain`，递归读取其下重新生成的 `epoch_*/trading_action_detail_epoch_*.csv`（必含 `contract` / 原始 `volume` / `position_after` / `mark_price` / `realized_pnl_step` / `commission_fee_step` / `unrealized_pnl` / `action` / `标签` / `分箱索引` / `初始动作` / `数据文件` 等列）。旧 Detail 缺少 `contract` 或原始 `volume` 时硬失败并提示重新生成，不得回退到处理后的 `log_volume_origin`。
  - `--selection_manifest`：指向 `analysis_result/DiHFT/low_level/<dataset>/<experiment>/selection_manifest.json`，仅用于生成 `is_selected`。
  - `--output_dir`：指向 `analysis_result/DiHFT/low_level/<dataset>/<experiment>`，承载本功能生成的数据与报告。
- 输入发现阶段同时扫描 `--model_root/epoch_*` checkpoint 目录并生成 Detail 覆盖率报告；缺少未选 epoch 只告警，缺少 manifest 已选 epoch、epoch 标识不一致或重复文件则失败。
- `--selection_manifest` 虽不过滤候选，但必须先通过数据集/实验归属、七个 label 唯一完整性、epoch 路径一致性和 checkpoint 存在性校验；只有校验后的逻辑 triple 可用于 `is_selected` 标记。
- 分类器的计算分组键（内部，不进分类输出）：`(label, epoch, bin_index, contract, df_seq, initial_action)`，对应 Detail CSV 中按 `(标签, 分箱索引, 初始动作, 数据文件)` 分组的一组连续行。
- 同一行为轨迹的行以 `timestep` 为唯一权威顺序；全局 CSV 行序以及不同轨迹之间的交错方式不影响结果。
- 输出明细：`agent_pattern_window_table.csv` 和 `agent_pattern_expanded_table.csv`。覆盖诊断写入 `agent_pattern_coverage_report.csv`。聚合固定输出 `agent_pattern_{kline,strategy,cross}_{scenario,triple}_summary.csv`；情景级统计为 `total_net_pnl, window_count, pnl_p25, pnl_p50, pnl_p75`，triple 级对实际命中该形态的 initial-action 情景同名指标等权平均，使用 `mean_initial_action_*` 列名并输出命中/期望情景计数与覆盖率。
- `analysis_manifest.json` 记录阈值配置、窗口配置和输入/输出文件指纹，作为结果可重现契约。每个实际读取的 Detail CSV、epoch sidecar 和 `selection_manifest.json`，以及每个生成的 CSV/JSON 输出（`analysis_manifest.json` 自身除外），都必须记录逻辑相对路径、字节数和 SHA-256。Detail/sidecar 路径相对 `model_root`，输出路径相对 `output_dir`，selection manifest 路径相对其父目录。manifest 还必须记录扫描到但缺少 Detail 的 checkpoint epoch 清单。
- `未分类` 和 `策略未分类` 保留在明细与展开表，但不进入任何正式 kline、strategy 或 cross summary；六个 summary 只包含正式的 7 类 K 线形态和 6 类策略形态。标定报告单独输出未分类哨兵的窗口数、比率和 PnL 分布。“盈亏区分度”定义为同一视图内各正式形态 `pnl_p50` 的最大值减最小值，仅作诊断。

## Testing Decisions

### 测试 seam（已与用户确认）

- **主 seam = 纯函数 seam**：两个分类器都是纯函数（给定一个 N=20 步窗口的序列 → 返回形态标签），这是最高且最稳的 seam。**新增 seam**（形态判别是新的关注点，与现有 `test_pick_agent.py` 的 reward 聚合转换不同），但测试风格沿用现有约定。
- 理想 seam 数 = 1：所有判别逻辑（6/7 类判别、命中顺序、互斥/多选、阈值边界、未分类）都通过纯函数 seam 覆盖。
- 不单独测的（低价值，靠主 seam 间接覆盖 + 一个 smoke test）：
  - 明细表生成脚本（薄 orchestrator，I/O 胶水）：一个端到端 smoke test 验证跑通。
  - 聚合脚本（数组展开 + groupby）：一个 smoke test 验证每窗口明细唯一、展开后不跨策略形态汇总盈亏、initial-action 情景不被直接相加、形态未命中情景不被伪造为零 PnL，且期望 Initial-action 的 Detail 行为轨迹缺失时报错。

### 好测试的标准

- 只测外部行为（给定序列 → 期望标签），不测内部实现（如不 assert 中间特征值的计算步骤）。
- 合成输入精确构造每类形态的边界 case：完美阶跃序列测 KT1/ST1、盈利后同向加仓序列测 ST3、箱体往返序列测 KM2、Z≥2 反转序列测 KM1、价格-量背离序列测 KM3。
- 测 KT1 延续边界：最终价格须从基准边界同向延伸至少 0.5%，且突破后至少 80% 的观测点保持在突破侧；分别覆盖阈值内外、上涨和下跌方向。
- 测 KT2 的严格事件顺序：突破后先形成同向新极值，再反向回撤至少 0.3% 并进入基准边界 ±0.5%，最后重新同向延伸至少 0.5%；断言突破触发点本身不能充当回踩点。
- 测 ST3 对称性与排除条件：任意输入仓位档位下，多头和空头的同向加仓在加仓前浮盈为正时均命中；浮盈为 0/负数、开仓、减仓和反手均不命中。
- 测 ST1 时序与动作语义：突破当步或下一步必须从空仓开到按 `near_full_ratio` 计算出的同向近满仓档位，不能由加仓、减仓、平仓或反手触发；持仓计数包含开仓当步，后续同号减仓仍计入，平仓或变号中断。
- 测 ST2 事件链：突破后回踩当步或下一步发生任意同向加仓，在方向一致且随后再延续时命中；缺少突破、回踩、同向加仓或再延续任一环节均不命中。加仓前浮盈为正时可同时命中 ST2 与 ST3。
- 用不同的 sidecar `position_levels` 参数化运行策略分类测试，验证相邻关系、近满仓档位和加仓事件均随权威档位集合变化。
- 测 SM1 多空与涨跌停对称性：低位极端/跌停时逆向开多或加多至近满仓、高位极端/涨停时逆向开空或加空至近满仓均命中；仅持有、顺势、减仓、平仓和反手均不命中。
- 测 SM2 参数驱动的相邻档位语义：在至少两组 `max_holding_number` / `position_choices` 下，高位向空头方向移动一个相邻档位、低位向多头方向移动一个相邻档位并形成往返时均命中；只有单侧调节、跨多档跳仓或直接反手均不命中。
- 测 SM3 过滤语义：背离段内没有顺突破方向开仓/加仓且非背离时段存在有效调仓时命中；背离段顺势增加风险或全窗口无交易时均不命中。
- 测命中顺序：构造"突破+回调"序列，断言命中 KT2 而非 KT1。
- 测突破时序：前 5 步只建立基准区间、不参与突破判定；突破只能在第 6~10 步触发。
- 测 KT3 加速与对称性：上涨和下跌序列在后半窗绝对斜率达到阈值倍数时均命中；匀速趋势、前后斜率反向、倍数不足或累计收益不足均不命中。
- 测 KM1 V/倒V 对称性与端点排除：双腿方向相反、幅度和最小边长均达标时命中；极值落在窗口端部、任一腿过短/幅度不足或两腿同向均不命中。
- 测 KM2 箱体边界与往返计数：价格全部位于外沿、log-return 波动率达标且压缩后的上下沿状态至少转换 4 次时命中；中间区域不计作触沿且不打断等待，越过外沿、波动率超限或转换不足均不命中。
- 测 KM3/SM3 共享背离检测器：新高与新低在价格突破、成交量下降和价格移动阈值均达标时对称命中；任一条件不足均不构成背离。
- 测背离优先级：同时满足 KM3 与 KT1 的低量突破窗口归类为 KM3；满足 KT2 的突破回调窗口仍优先归类为 KT2。
- 测多选：构造回踩同向加仓且加仓前浮盈为正的序列，断言策略形态命中 {ST2, ST3}。
- 测事件冲突：同一低量背离突破上的顺突破开仓可命中 ST1，但必须使该事件的 SM3 不命中；只有构造两个不同合格事件时才允许同窗命中 {ST1, SM3}。
- 测短轨迹：ST1/ST2/SM3 少于 20 步不运行；ST3 和涨跌停语义的 SM1 在 1 条执行记录上可判定；SM2 仅在至少 2 条执行记录且价格方差非零时运行。样本不足时不得命中；涨跌停事件窗口全部规则未命中时输出 `strategy_patterns=["策略未分类"]`。长度达到 20 的 label_0/6 轨迹必须在整条轨迹上扫描，且仍只产生一个窗口。
- 测标准窗口切分：只产出完整的 N=20 非重叠窗口；尾部不产出形态行，但覆盖率报告的丢弃步数与丢弃 PnL 必须正确。
- 测盈亏边界：窗口 `gross_pnl` 的已实现 PnL 汇总包含首尾两端全部执行步；非首窗口的浮动 PnL 基线取前一行，首窗口基线取 0。断言 `net_pnl = gross_pnl - 窗口手续费总和`，且不重复扣除 `slippage_step`。
- 测窗口行唯一性：每个 `window_id` 恰好一行；`kline_patterns` 为单元素 JSON 数组，策略多选如 `["ST2", "ST3"]` 保存在同一行，一组 PnL 只出现一次。
- 测未分类：构造随机游走序列，断言 K 线形态 = 未分类。
- 测阈值边界：构造刚好在阈值上下的序列，断言分类翻转。
- 阈值标定后会变，测试应参数化阈值（fixture 注入），标定后只改 fixture 不改断言结构。

### 测试模块

- K 线形态分类器：`FineFT/tests/analysis/test_kline_pattern_classifier.py`。
- 策略形态分类器：`FineFT/tests/analysis/test_strategy_pattern_classifier.py`。
- 明细表生成 + 聚合：各一个 smoke test，可放 `FineFT/tests/analysis/test_pattern_detail_table.py`。

### Prior art

- [test_pick_agent.py](../../../FineFT/tests/analysis/test_pick_agent.py)：测 `picker` 的纯转换函数，合成 cross-contract record 输入 → 期望输出，`pytest` + `tmp_path` fixture，目录 `FineFT/tests/analysis/`。本 spec 沿用此风格。
- [test_slice_model.py](../../../FineFT/tests/datahandler/test_slice_model.py)：datahandler 测试 prior art。

### 阈值标定流程（goal-driven）

1. 用提议阈值实现分类器 → 验证：单测全部通过（合成输入）。
2. 跑真实 Detail CSV（如 `trading_action_detail_epoch_58.csv`）→ 输出未分类率、7 类 K 线形态分布、6 类策略形态分布和盈亏区分度；未分类率 ≥30%、某类零命中或区分度弱只产生诊断告警，不使程序或验收失败。
3. 看每个判别特征在真实窗口上的分布 → 经人工确认后调阈值 → 重跑并比较诊断报告；不得以满足固定类别占比为调参目标。
4. 锁定阈值 → 回归测试覆盖。

硬性验收只覆盖可证明的不变量：输入/输出 schema、唯一键、必需字段无非法空值、候选与 initial-action 情景覆盖、窗口盈亏守恒、同一输入与配置的结果确定性，以及合成边界测试。真实数据类别占比与盈亏区分度属于标定诊断，不属于硬性通过条件。

上游数据验收还必须证明：Scale Save passthrough 前后行数和行序不变，原始 `volume` 逐行完全一致，且 `volume` 不在 `state_features.npy` 中；重新生成的 Detail CSV 必含非空 `contract` 和原始 `volume`，并伴随可校验的动作空间 sidecar。

## Out of Scope

- **6 类与 12 原型的映射**：[label_agent_selection_logic.md](../../../docs/research/label_agent_selection_logic.md) 与 ADR-0005 已定义 12 个 Agent 策略原型（Archetype）。本 spec 的 6 类二阶形态与 12 原型的映射关系 deferred，本轮不建立。但术语层级关系已立住（Archetype = 12 类语义层，K 线/策略形态 = 6/7 类形状层），不会混淆。
- **per-triple 的 Sharpe / Calmar / MDD / win_rate**：这些风险/收益指标当前只在 high_level_heuristic 阶段对合并 ensemble 计算，per-triple 不存在。本 spec 只输出窗口级 `gross_pnl` / `net_pnl` 及已约定的聚合统计，不派生 Sharpe 等。如需，是后续 spec。
- **per-step `.npy` 轨迹转储**：[test_agent_index.py](../../../FineFT/RL/DiHFT/low_level/test_agent_index.py) 中转储逐 step 轨迹的代码被注释掉。本 spec 直接读 Detail CSV，不需要重新启用 `.npy` 转储。
- **把分类结果接入 Meta Router 调度**：本 spec 只产出明细表与聚合视图，不改 Stage III 路由逻辑。如何把形态标签用于调度是后续 spec。
- **跨数据集泛化**：本 spec 只在 `fu/30min_multi` 上落地。其他品种/频率的标定是后续工作。
- **修改模型训练输入或重新训练**：保留原始 `volume` 仅用于 Detail 输出与离线形态分析，不加入 `state_features.npy`，因此不改变 Q 网络输入，也不触发重新训练。
- **`labeling_method` 切换**：`slope` 已在 ADR-0006 落实，不在本 spec 范围。
- **训练或改变动作空间**：本 spec 只消费调用方提供的仓位档位配置，不修改训练/测试脚本的 `position_choices` 或 `max_holding_number`，也不负责重新训练模型。

## Further Notes

- **label_0/6 的特殊性**：实测 label_0/6 trajectory 中位数 3 步、p90=10 步，本就是涨跌停瞬间切片，没有"形态"可言。KX1 是诚实处理——不为凑 6 类硬塞，也不留 null 稀疏行。
- **7×6 组合空间的稀疏**：K 线形态受 label 约束（label_4 上涨段不会有 KM2 箱体），实际有效组合远小于 7×6=42。这是设计使然，不是缺陷——稀疏本身是信息（哪些组合在该 label 下不可能）。
- **策略多选的语义**：同一窗口可在策略形态数组中同时保存兼容规则，例如 ST2 和 ST3；ST1 与 SM3 只有归因于不同事件时才能同窗出现。明细表只保留一行和一组 `gross_pnl`/`net_pnl`。形态分析按需展开数组；各策略形态的盈亏独立解释，不跨策略形态求和为账户总盈亏。
- **`initial_action` 的诊断与汇总价值**：虽然不进分类键，但实现时仍按 `initial_action` 分组记录行为轨迹并先生成情景级统计，便于诊断起步仓位对形态和盈亏的影响。主 triple 的每个形态组只对命中该形态的情景等权平均，并显式报告情景覆盖率；不同 initial-action 的总盈亏不得直接相加。
- **与 Semantic Guard 的关系**：明细表的"策略形态"列可作为 Semantic Guard 是否把 agent 限制在合规方向的离线验证材料（如 label_4 上涨的 agent 是否真的没出现 SM1 硬边界反转的逆向押注）。这是后续分析，不在本 spec 落地范围。
- **阈值是提议值**：所有阈值（如 2.0、0.3%、0.5%、20%、80%、1.5）都是基于当前规则的提议值，尚未在真实 trajectory 数据上标定。落地必须先实现 → 跑真实分布 → 人工确认调阈值 → 锁定。这是工程量的一部分，不是可选项。
