# add-agent-pattern-dual-classifier

> 架构决策见 [ADR-0006：Agent 形态分类：K 线形态与策略形态双独立分类器](../../../docs/adr/0006-agent-pattern-dual-classifier.md)。术语见 [CONTEXT.md](../../../CONTEXT.md) 的 Evaluation And Diagnostics 段。本 spec 是该 ADR 的落地规格。

## Problem Statement

当前低层 agent 选择链路（`FineFT_single_agent_with_different_position.py`）只能按 `trans_reward_mean` 等 reward 类指标给每个 `(label, epoch, bin_index)` agent triple 排序，无法回答一个对策略研究和 agent 调度都很关键的问题：

**"哪个 agent 在哪个 label 的哪个 K 线形态下、用了什么二阶策略、盈利如何？"**

用户（策略研究员 / agent 调度设计者）面对一堆候选 agent triple，只有"平均回报"这一个粗粒度信号，无法区分：

- 同样在 `label_4`（持续上涨）下盈利的两个 agent，一个是"突破即时重仓"（大赚大亏、滑点敏感），另一个是"回调加仓"（胜率高、易踏空单边）——它们适用行情和风险特征完全不同，但 reward 均值可能相同。
- 一个 agent 是否同时擅长多种形态（"一个 agent 可以属于多种类型"），以便在 Meta Router 调度时按行情形态匹配，而不是只按 label 匹配。

缺乏这层"形态 × 策略 × 盈亏"的明细，用户就无法做基于形态的 agent 归类、无法验证 Semantic Guard 是否真的把 agent 限制在合规方向、也无法为 Stage III 路由提供比 label 更细的匹配维度。

## Solution

构建**两个正交的分类器**，在现有 Detail CSV（`trading_action_detail_epoch_*.csv`）上离线计算，产出一張 **Agent 形态明细表**，读法为"哪个 agent (epoch, bin) 在哪个 label 的哪个 K 线形态下用哪种策略盈利如何"。

1. **K 线形态分类器**（行情侧）：读 `mark_price` / `volume` 序列，识别 7 类行情二阶形态。
2. **策略形态分类器**（agent 侧）：读行为轨迹整体（`position_after` + `mark_price` + `volume` + `cumulative_realized_pnl`），识别 6 类 agent 动作与行情的关系模式。
3. **明细表**：按 `(label, epoch, bin_index, K 线形态, 策略形态, 盈亏)` 组织，构成 7×6 组合空间；一个 triple 可命中多行，天然支持"一个 agent 属于多种类型"。

两个分类器是纯函数，输入一个 N=20 步窗口的序列，输出形态标签。明细表生成与聚合是薄 orchestrator 层。

## User Stories

### 核心查询

1. 作为策略研究员，我想查看任意一个 `(label, epoch, bin_index)` agent triple 在 7 种 K 线形态下的盈亏分布，以便判断它擅长哪种行情。
2. 作为策略研究员，我想查看任意一个 agent triple 在 6 种策略形态下的盈亏分布，以便判断它用的是突破即时、回调加仓、还是金字塔递增等哪种二阶策略。
3. 作为策略研究员，我想把"K 线形态"和"策略形态"交叉成 7×6 表，看同一个 agent 在不同 (K 线形态, 策略形态) 组合上的盈亏，以便发现"突破行情下用回调加仓策略盈利"这类细粒度信号。
4. 作为策略研究员，我想知道一个 agent 是否同时命中多种策略形态（如同时是 ST1 突破即时 + SM3 背离增强），以便确认"一个 agent 可以属于多种类型"。
5. 作为策略研究员，我想按 label 过滤明细表，看该 label 下所有 agent 的形态分布，以便对比同 label 内不同 agent 的策略差异。

### 形态识别

6. 作为策略研究员，我想让系统在 `label_1`~`label_5` 的 segment 内按 N=20 步不重叠窗口自动识别 K 线形态，以便把一段行情切成若干个形态同质的区间分别归因盈亏。
7. 作为策略研究员，我想让系统对 `label_0` / `label_6`（涨跌停）的 trajectory 直接标记为 KX1 涨跌停型，不在其上跑滑窗识别，以便尊重其极短长度（median=3 步）。
8. 作为策略研究员，我想让 K 线形态单窗口单选、命中顺序为 `KX1→KM1→KT2→KT1→KT3→KM2→KM3→未分类`（KT2 优先于 KT1），以便回调形态不会被突破形态吞没。
9. 作为策略研究员，我想让策略形态单窗口多选，以便一个窗口能同时命中 ST1 与 SM3 这类不互斥的组合。
10. 作为策略研究员，我想看到无法命中任何类的窗口被标为"未分类"，以便知道识别覆盖率。

### 盈亏归因

11. 作为策略研究员，我想让每行盈亏 = 该窗口内已实现 PnL 变化 + 浮动 PnL 变化，以便盈亏反映窗口期真实持仓盈亏而非单步 reward 噪声。
12. 作为策略研究员，我想让 K 线形态单选不重复计入盈亏（一个窗口的盈亏只归到其唯一 K 线形态行），以便按 K 线形态聚合时盈亏不放大。
13. 作为策略研究员，我想让策略形态多选时盈亏重复计入各命中行，但聚合时按形态 distinct 计算总盈亏（不跨形态求和），以便既保留明细又不在聚合层放大。
14. 作为策略研究员，我想看到每个 triple 在每个 (K 线形态, 策略形态) 组合下的总盈亏、窗口数、盈亏分位数，以便做粗粒度对比。

### 与现有产物对齐

15. 作为 agent 调度设计者，我想让分类键 `(label, epoch, bin_index)` 与 `selection_manifest.json` 的部署产物对齐，以便明细表能直接 join 到已被部署的 agent。
16. 作为 agent 调度设计者，我想把 `initial_action` 排除在分类键外（只作诊断切片轴），以便分类输出描述的是被部署的 agent 而非某种起步条件。
17. 作为 agent 调度设计者，我想把 segment（df_0..n）排除在分类维度外（只作 K 线数据来源文件），以便分类键不被文件管理维度污染。
18. 作为 agent 调度设计者，我想让明细表能与 `result_all.csv`（按 `df_path` 列）join 回原始 trajectory，以便追溯任意一行的原始逐 step 数据。

### 阈值标定与诊断

19. 作为策略研究员，我想用真实 Detail CSV 跑一遍分类器，看 7 类 K 线形态和 6 类策略形态各自的命中分布，以便判断阈值是否合理（未分类率是否过高、某类是否从不命中）。
20. 作为策略研究员，我想在标定阶段查看每个判别特征（如阶跃比、指数拟合 R²、Z-score 线性 R²）在真实窗口上的分布，以便调整阈值而非拍脑袋。
21. 作为策略研究员，我想在标定后用回归测试锁住阈值，以便后续重构不破坏分类语义。

### 可读性与可维护性

22. 作为后续维护者，我想让两个分类器是纯函数（输入序列 → 输出标签），不依赖 I/O，以便单独测试和复用。
23. 作为后续维护者，我想让明细表生成脚本只是薄 orchestrator（读 Detail CSV → 切窗口 → 调分类器 → 写表），不在其中塞判别逻辑，以便判别逻辑集中在分类器内可独立演进。
24. 作为后续维护者，我想让所有形态标签用前缀命名（KT/KM/KX + ST/SM），以便在表里短而不混淆。
25. 作为后续维护者，我想让分类器读行为轨迹整体（含 price/vol/pnl），而非只读 position_after，以便能判别"动作与行情的关系模式"而非纯动作形状。

## Implementation Decisions

### 架构与分类键

- 采用 [ADR-0006](../../../docs/adr/0006-agent-pattern-dual-classifier.md) 定义的双独立分类器架构。K 线形态（行情侧）与策略形态（agent 侧）正交，分别由两个分类器独立判定，构成 7×6 组合空间。
- **分类键** = `(label, epoch, bin_index)`，对齐 `selection_manifest.json` 的部署产物。`initial_action` 不进键（部署产物不带它，且 3 条起步强相关投票会重复计数），降级为诊断切片轴。`segment`（df_0..n）是 label 内文件切片，不构成分类维度，仅作为 K 线数据的来源文件。
- 明细表 schema：`(label, epoch, bin_index, K 线形态, 策略形态, 盈亏)`。一个 agent triple 可命中多行（跨窗口 + 策略形态多选），天然支持"一个 agent 属于多种类型"。

### 形态识别窗口

- `label_1`~`label_5` segment 内按 **N=20 步不重叠窗口**切分（步长 = N，无重叠）。不重叠设计避免盈亏重复计入。
- `label_0` / `label_6` 不使用窗口（trajectory 中位数 3 步，实测 label_0/6 短于 10 步的占绝大多数），直接标记 KX1。
- 选 N=20 的依据：实测 label_1~5 trajectory 中位数 102~185 步，N=20 覆盖 KM2 箱体最小往返（~15~20 步）与 KM3 背离（2 个新高 ~10~20 步）；label_3/4/5 在 N=20 时无法识别率仅 8.3%。

### K 线形态分类器（7 类，单选）

- 7 类：KT1 突破即时型 / KT2 回调型 / KT3 加速型 / KM1 V 反转型 / KM2 箱体型 / KM3 背离型 / KX1 涨跌停型。
- 命中顺序（互斥单选，命中即止）：`KX1 → KM1 → KT2 → KT1 → KT3 → KM2 → KM3 → 未分类`。KT2 优先于 KT1 以保留回调形态（否则所有"突破+回调"都被 KT1 吞没）。
- KX1 仅由 label 决定（label ∈ {0,6} → KX1），不进窗口识别。
- 各类判别公式与提议阈值（需标定，见 Testing Decisions）：
  - **KM1 V 反转**：窗口内 Z-score 触及 `|z|≥2.0` 后反向，且反向后 `|cum_ret_from_trough|≥0.5%`。
  - **KT1 突破即时**：前 5 步内 `mark_price` 突破窗口前 5 步 max/min ±0.3%，且后 15 步 `|cum_ret|≥0.5%` 延续。
  - **KT2 回调**：含突破（同 KT1 触发）+ 之后回踩到突破点 ±0.5% 内 + 再延续。
  - **KT3 加速**：`mark_price` 对时间指数拟合 R²≥0.6 且斜率参数同号于趋势方向。
  - **KM2 箱体**：`roll_std<0.3%` 且窗口内 `mark_price` 在均价 ±0.5% 内往返 ≥2 次。
  - **KM3 背离**：`vol_trend<0` 且 `|cum_ret|≥0.5%`（价新高/低但量能不确认）。
- 阈值均为提议值，落地后用真实数据标定（见 Testing Decisions）。

### 策略形态分类器（6 类，多选）

- 6 类：ST1 突破即时型 / ST2 回调加仓型 / ST3 金字塔递增型 / SM1 硬边界抄底型 / SM2 网格微调型 / SM3 背离增强型。
- 单窗口**多选**：一个窗口可命中多类（如 ST1+SM3：阶跃动作 + 背离过滤不互斥）。盈亏重复计入各命中行，聚合时按形态 distinct 计算总盈亏（不跨形态求和，避免放大）。
- 读行为轨迹整体（`position_after` + `mark_price` + `volume` + `cumulative_realized_pnl`），判别 agent 动作与行情的关系模式，而非纯动作形状。
- 各类判别公式与提议阈值：
  - **ST1 突破即时**：`max(|Δpos|)/max_hold≥0.8`（单步跳到接近满仓），且阶跃后 `|pos|` 维持 ≥10 步。
  - **ST3 金字塔递增**：`pos[t]~a·exp(b·cum_pnl[t])`，b 与 label 方向同号，R²≥0.6。
  - **ST2 回调加仓**：`corr(pos[t], price[t-k])` 峰值在 k≥2（仓位滞后价格），且窗口内有价格回踩时 `Δpos` 才显著。
  - **SM1 硬边界抄底**：`|pos|` 在 `|z_price|≥2.0` 时的均值 ≥ 其他时均值 ×3.0。
  - **SM2 网格微调**：`pos[t]~-α·z_price[t]`，R²≥0.7，α 小。
  - **SM3 背离增强**：窗口内存在价格-量背离段（`vol_trend<0` 且 `|cum_ret_price|≥0.5%`），且该段内 `|Δpos|` 均值 ≤ 全局 `|Δpos|` 均值 ×0.5。
- 多选下无命中顺序；各类独立判定，命中几个算几个。

### 盈亏归因

- 每行盈亏 = 窗口内 `已实现 PnL 变化 + 浮动 PnL 变化`，即 `(cumulative_realized_pnl[end] - cumulative_realized_pnl[start]) + (unrealized_pnl[end] - unrealized_pnl[start])`。
- K 线形态单选 → 盈亏不重复计入；策略形态多选 → 盈亏重复计入各命中行；聚合时按形态 distinct（每个形态独立算总盈亏，不跨形态求和）。

### 模块划分

- 新建分类器模块目录（建议在 `FineFT/analysis/` 下新建 `classify_agent/` 子包），包含：
  - K 线形态分类器（纯函数：窗口序列 → 单一 K 线形态标签）。
  - 策略形态分类器（纯函数：窗口序列 → 策略形态标签集合）。
  - 明细表生成脚本（薄 orchestrator：读 Detail CSV → 按 triple 遍历 → label_0/6 标 KX1 / label_1~5 切 N=20 不重叠窗口 → 调两个分类器 → 写明细表）。
  - 聚合视图脚本（按 `(label, epoch, bin, K 线形态)` 与 `(label, epoch, bin, 策略形态)` distinct 聚合）。
- 不修改 `test_agent_index.py`（Detail CSV 已含所需列）、不修改 `FineFT_single_agent_with_different_position.py`（selection_manifest 不变）、不修改 `slice_model.py`（label 切分不变）。
- 不修改现有 `labeling_method` 配置（`slope` 已在 ADR-0006 落实，与本 spec 无关）。

### 数据来源

- 输入：`analysis_result/DiHFT/low_level/<dataset>/<experiment>/trading_action_detail_epoch_*.csv`（已含 `position_after` / `mark_price` / `volume` / `cumulative_realized_pnl` / `unrealized_pnl` / `action` / `标签` / `分箱索引` / `初始动作` / `数据文件` 等列）。
- 分类器的计算分组键（内部，不进分类输出）：`(label, epoch, bin_index, contract, df_seq, initial_action)`，对应 Detail CSV 中按 `(标签, 分箱索引, 初始动作, 数据文件)` 分组的一组连续行。
- 输出：明细表 CSV（建议路径在 `analysis_result/DiHFT/low_level/<dataset>/<experiment>/` 下，文件名如 `agent_pattern_detail_table.csv`）。

## Testing Decisions

### 测试 seam（已与用户确认）

- **主 seam = 纯函数 seam**：两个分类器都是纯函数（给定一个 N=20 步窗口的序列 → 返回形态标签），这是最高且最稳的 seam。**新增 seam**（形态判别是新的关注点，与现有 `test_pick_agent.py` 的 reward 聚合转换不同），但测试风格沿用现有约定。
- 理想 seam 数 = 1：所有判别逻辑（6/7 类判别、命中顺序、互斥/多选、阈值边界、未分类）都通过纯函数 seam 覆盖。
- 不单独测的（低价值，靠主 seam 间接覆盖 + 一个 smoke test）：
  - 明细表生成脚本（薄 orchestrator，I/O 胶水）：一个端到端 smoke test 验证跑通。
  - 聚合脚本（纯 groupby，逻辑极薄）：一个 smoke test 验证 distinct 不放大。

### 好测试的标准

- 只测外部行为（给定序列 → 期望标签），不测内部实现（如不 assert 中间特征值的计算步骤）。
- 合成输入精确构造每类形态的边界 case：完美阶跃序列测 KT1/ST1、指数增长序列测 ST3、箱体往返序列测 KM2、Z≥2 反转序列测 KM1、价格-量背离序列测 KM3。
- 测命中顺序：构造"突破+回调"序列，断言命中 KT2 而非 KT1。
- 测多选：构造"阶跃+背离"序列，断言策略形态命中 {ST1, SM3}。
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
2. 跑真实 Detail CSV（如 `trading_action_detail_epoch_58.csv`）→ 验证：未分类率 < 30%、7 类 K 线形态分布合理（无某类从不命中）、6 类策略形态分布合理。
3. 看每个判别特征在真实窗口上的分布 → 调阈值 → 重跑 → 验证分布改善。
4. 锁定阈值 → 回归测试覆盖。

### 数值稳定性验证

- 部分公式（ST2 的 `corr(pos[t], price[t-k])` 峰值、SM3 的背离段活跃度、ST3 的指数拟合）在 N=20 小窗口上的数值稳定性未验证。实现时可能发现某些特征在短窗口上方差太大无法用，需降级或换公式——这是实现期风险，已在 ADR-0006 后果段记录。

## Out of Scope

- **6 类与 12 原型的映射**：[label_agent_selection_logic.md](../../../docs/research/label_agent_selection_logic.md) 与 ADR-0005 已定义 12 个 Agent 策略原型（Archetype）。本 spec 的 6 类二阶形态与 12 原型的映射关系 deferred，本轮不建立。但术语层级关系已立住（Archetype = 12 类语义层，K 线/策略形态 = 6/7 类形状层），不会混淆。
- **per-triple 的 Sharpe / Calmar / MDD / win_rate**：这些风险/收益指标当前只在 high_level_heuristic 阶段对合并 ensemble 计算，per-triple 不存在。本 spec 的"盈亏值"仅用窗口内已实现+浮动 PnL 变化，不派生 Sharpe 等。如需，是后续 spec。
- **per-step `.npy` 轨迹转储**：[test_agent_index.py](../../../FineFT/RL/DiHFT/low_level/test_agent_index.py) 中转储逐 step 轨迹的代码被注释掉。本 spec 直接读 Detail CSV，不需要重新启用 `.npy` 转储。
- **把分类结果接入 Meta Router 调度**：本 spec 只产出明细表与聚合视图，不改 Stage III 路由逻辑。如何把形态标签用于调度是后续 spec。
- **跨数据集泛化**：本 spec 只在 `fu/30min_multi` 上落地。其他品种/频率的标定是后续工作。
- **`labeling_method` 切换**：`slope` 已在 ADR-0006 落实，不在本 spec 范围。

## Further Notes

- **label_0/6 的特殊性**：实测 label_0/6 trajectory 中位数 3 步、p90=10 步，本就是涨跌停瞬间切片，没有"形态"可言。KX1 是诚实处理——不为凑 6 类硬塞，也不留 null 稀疏行。
- **6×6 组合空间的稀疏**：K 线形态受 label 约束（label_4 上涨段不会有 KM2 箱体），实际有效组合远小于 7×6=42。这是设计使然，不是缺陷——稀疏本身是信息（哪些组合在该 label 下不可能）。
- **盈亏重复计入的语义**：策略形态多选导致同一窗口盈亏出现在多行。这不是 bug 而是设计——"这个窗口的盈亏同时支持 ST1 和 SM3 两个策略假设"。聚合时按形态 distinct（不跨形态求和）避免放大。明细层保持信息无损。
- **`initial_action` 的诊断价值**：虽然不进分类键，但实现时仍按 `initial_action` 分组记录行为轨迹，便于后续做"起步仓位是否影响形态"的诊断切片。这是诊断视图，不是主分类输出。
- **与 Semantic Guard 的关系**：明细表的"策略形态"列可作为 Semantic Guard 是否把 agent 限制在合规方向的离线验证材料（如 label_4 上涨的 agent 是否真的没出现 SM1 硬边界抄底的反向押注）。这是后续分析，不在本 spec 落地范围。
- **阈值是提议值**：所有阈值（2.0、0.3%、0.5%、0.6、0.7、0.8 等）都是基于原始描述的提议值，尚未在真实 trajectory 数据上标定。落地必须先实现 → 跑真实分布 → 调阈值 → 锁定。这是工程量的一部分，不是可选项。
