# FineFT / DiHFT：Today / Next TradingDay 多预测期限特征审计功能开发规范

> 文档类型：开发需求与实施设计（Markdown）  
> 版本：**提议版 v0.1.0** · 2026-10-13  
> 适用场景：国内商品期货；以 `fu / 10min` 为首个验证配置  
> 目标读者：数据工程、量化研究、模型训练与审计开发人员 / AI Coding Agent  
> 状态：**待实现**。本文提出的 Stage 6.5、文件接口、配置、指标和测试均为开发建议，**不代表仓库中已存在对应模块**。

## 0. 一页结论

**核心问题**：目前的三流特征筛选主要使用未来 `w=6` 根 10min Bar（约 60 分钟）的预测相关性，却希望策略利用多尺度行情信息判断**当前 TradingDay 收盘前**和**下一个 TradingDay 收盘前**的盈利机会。这存在需要实证核查的**预测期限错配风险**。

**推荐顺序**：

```text
Stage 1~5 生成候选特征
       ↓
Stage 6   清洗 + Train / Valid / Test 时间切分
       ↓
Stage 6.5 多预测期限特征审计                 ← 本期新增、只读、无需训练RL/VAE
       │   ├─ 生成 Today / Next TradingDay 离线研究标签
       │   ├─ 对全部合格候选特征评估分期限相关性
       │   ├─ 检查标签成熟、样本覆盖、跨合约稳定性
       │   └─ 生成预测期限错配及入选/遗漏审计报告
       ↓
Stage 7   现有三流特征选择                   ← 首期保持原行为
       ↓
Stage 8   特征缩放与训练数据物化
       ↓
Stage I/II RL、VAE 训练、选模
       ↓
Stage III 路由与模拟交易
       ↓
后续阶段  模型利用能力审计、策略级消融归因   ← 非 Stage 6.5 MVP
```

**功能与结论边界**：

1. **特征工程**：计算在决策时刻可得到的输入，不负责证明策略能赚钱。
2. **Stage 6.5 审计**：诊断各特征对 `Today`、`Next TradingDay` 等目标的**样本外预测关联**；无需 RL/VAE 训练。
3. **训练后模型审计**：判断网络是否使用了这些预测信息。
4. **完整模拟交易与消融**：在相同可执行性和成本假设下，判断特征或特征组是否真正改善了**净收益**。

**严格禁止**：把离线未来标签输入在线策略；把 IC 当成利润贡献率；把 Q 值直接当成盈利概率；使用最终 Test 进行特征选择。

---

## 1. 项目现状与实施依据

### 1.1 已有数据流水线及研究事实

据用户提供的 FineFT / DiHFT 项目文档，主要流程为：

| 既有阶段 | 当前职责 | 与新审计关系 |
|---|---|---|
| Stage 1 | L2/快照下采样 | 提供价格、盘口、时间基础数据 |
| Stage 2 | 截面特征 | 提供 K 线及盘口因子 |
| Stage 3 | 日/周/跨期扩展 | 提供跨尺度上下文 |
| Stage 4 | 合并、shift、历史拼接 | 确保因果可用与对齐 |
| Stage 5 | 时间滚动特征 | 输出候选特征矩阵 |
| Stage 6 | 清洗与数据集切分 | 给 Stage 6.5 提供 split 和数据基线 |
| **Stage 6.5（拟新增）** | **多预测期限特征审计** | **在三流筛选前发现期限错配** |
| Stage 7 | 三流统计门禁 | 第一版不修改现有筛选阈值 |
| Stage 8 | 缩放及保存 | 第一版不修改现有模型输入 |
| 后续 Stage I/II/III | RL、VAE、选模、路由、仿真 | 后续验收特征是否改善动作及盈利 |

参考：`2026-10-11-fineft-project-architecture-v2.zh_cn(2).md`（第 2、5 节）；`2026-10-12-commodity-futures-feature-engineering-and-3stream-selection.zh_cn.md`（第 1、3 节）。

### 1.2 为什么不能直接沿用现在的 RankIC

当前筛选规范的核心指标：

\[
R_{t,6}=\frac{P_{t+6}-P_t}{P_t},\qquad
IC_{60m}(f)=\operatorname{SpearmanCorr}(f_t,R_{t,6})
\]

相应三流侧重点：

| 流 | 当前统计方向 | 新审计需要增加什么 |
|---|---|---|
| Slope VAE | 有符号收益 RankIC + 趋势区制分化 | 分期限方向/趋势诊断，最终结合路由质量 |
| Volatility VAE | 未来绝对收益 VolRankIC + 波动区制分化 | 分期限波动与风险诊断，**不因方向 IC 低而直接删除** |
| RL Decision | 有符号收益 RankIC + 统计稳定性 | Today/Next 多空机会诊断，后续与动作和净盈亏核对 |

历史 `fu/10min` Manifest 记录：候选特征 `566`，强制状态特征 `17`，并集 `165`，其中 Slope `15`、Vol `12`、RL `148`。这些是**指定报告的历史数值**，不能写死为所有运行版本的常量。

**数据版本核对提醒**：另一份架构快照中的 `vae_volatility_state_features.npy` 长度为 **13**，与上述 Manifest 的 **12** 不一致。审计时必须输出特征列表来源、顺序、版本和差异，不可悄悄将其视为同一模型配置。

参考：`2026-10-12-fu-10min-feature-selection-manifest-attribution.zh_cn.md`；`2026-10-11-fineft-project-architecture-v2.zh_cn(2).md`（第 5.4 节）。

### 1.3 模型级预测与前置特征审计不是一回事

最新 Trial 229 逐日记录中标记：

```text
future_horizon_predictor: "none"
```

说明这份轨迹并未提供独立的 Today/Next 盈利概率预测器。**Stage 6.5 不得据此宣称现有 Q-Net 已经显式预测未来 1–2 TradingDay 盈利概率**。

同时 Trial 229 的状态为 `NEEDS_AUDIT`，存在运行元数据和执行对账问题，故不能用当前回测数字直接给特征排名赋予实盘盈利含义。

参考：`2026-10-12-dihft-latest-experiment-diagnostics-report.zh_cn(4).md`（第 1–2 节）。

---

## 2. 范围、优先级与非目标

### 2.1 本期 MVP（P0，必须交付）

1. 从现有 **Stage 6 后、Stage 7 前** 的候选特征数据读取输入。
2. 按品种交易日历生成 `Today` / `Next TradingDay` **标签定义和成熟时间**。
3. 为两个预测期限生成：未来有符号收益、未来绝对收益，以及在可执行价格和费用数据可用时的多空净盈利标签。
4. 在时间前推的 Train/开发验证窗口内计算每个候选特征、每个预测期限、每个合约的 Spearman RankIC/VolRankIC。
5. 计算跨合约一致性、样本覆盖率、有效期、同一时钟日的聚合结果；按剩余交易时间分层。
6. 比较 `60min`、`Today`、`NextTradingDay`，标出**仅短期有效 / 今日有效 / 次交易日有效 / 证据不足**等**描述性标签**，不得在未冻结规则前将其作为自动淘汰条件。
7. 把当前已入选特征（165 的历史清单）与全部合格候选特征（566 的历史清单）放在一起评估，防止只审已入选特征而忽视被现有 60min 门禁淘汰的候选。
8. 输出可复现的 CSV/Parquet、JSON Manifest 和 Markdown 报告。
9. **只读执行**：不修改当前三份 `.npy`、缩放参数、checkpoint、Stage 7 阈值和现有历史实验产物。

### 2.2 下一阶段（P1，MVP 可预留接口）

- 训练独立、简单的监督预测基线，在时间前推的样本外片段上比较单个特征与特征组合的**增量预测力**。
- 针对二分类多空净盈利事件计算 Brier score、log loss、可靠性图、样本覆盖率；校准器与基础模型训练数据分离。
- 对 Today/Next 的目标选择及特征族候选开展开发期 walk-forward 研究。
- 在 RL 及 VAE 训练后，核查 Q 值、动作、VAE 区制和路由槽位的响应。

### 2.3 后续（P2，不属于 Stage 6.5）

- 特征组消融后**重新训练受影响模型**，比较完整策略的净收益、回撤和成本。
- Slot 0/3/6 条件归因，核验做空机会是否存在以及模型为何未把握。
- 冻结模型及实验协议后，进行未见时段最终 Test。

### 2.4 本期明确不做

- 不强制把 Today/Next 的 IC 加入现有三流硬门禁。
- 不直接删除维度或修改交易策略。
- 不让未来收益标签进入任何在线状态特征表。
- 不用最终 Test 优化特征组合、阈值、成本、预测器或模型参数。
- 不把“单特征未来收益相关”解释为“该特征创造多少利润”。
- 不假定 `bar.close` 一定是可成交价格；若缺少执行条件，净收益指标必须标为**不可得**而非伪造。

---

## 3. 预测目标与时间语义：所有实现以此为准

### 3.1 基本术语

- `event_timestamp`：行情事件/Bar 结束时刻（含明确时区）。
- `feature_available_at`：该条特征真正能够用于策略决策的最早时刻。
- `decision_timestamp`：策略本次决策时刻；必须满足 `feature_available_at <= decision_timestamp`。
- `trading_day`：交易所业务交易日，**不等于自然日期**。
- `label_end`：未来观察目标的截止时刻。
- `label_available_at`：该目标结果真正成熟、能够用于离线训练/统计的时刻（考虑数据发布时间）。
- `entry_time` / `exit_time`：如果研究交易净盈利，实际假设的可执行进出场时刻；必须记录。
- `contract_id`：实际可交易合约代码，不能和连续合成主力序列混用而无映射说明。

### 3.2 两个独立期限

设当前决策为 `t`，业务交易日为 `D = TradingDay(t)`：

| horizon_id | 定义 | label_end | 重要提醒 |
|---|---|---|---|
| `today` | 从当前决策时刻到 **D 当日日盘结束** | `day_session_end(D)` | 夜盘可能属于第二天的 TradingDay |
| `next_td` | 从当前决策时刻到 **next_trading_day(D) 当日日盘结束** | `day_session_end(next_trading_day(D))` | 指从 **t 起累计**至下一交易日结束，而非仅次日单独时段 |
| `60min` | 既有 6 根 10min Bar 基线 | 按既有定义 `t+6 bars` | 只用于比较，不能当 Today 或 Next 的替身 |

**必须区分**“今天剩余时段”与“下一个交易日整个时段”，以及“当前至下个交易日结束”的累计窗口。首期 `next_td` 采用上表中的**累计窗口**，不能在代码内部悄然改定义。

使用品种专用、版本化的交易日历与 Session 表；遇节假日、周末、夜盘、临时休市、早收盘、异常缺失数据时按真实日历解析。严禁硬编码 `24h/48h`、`144/288` 根 Bar 或直接使用 `date()+1`。

### 3.3 标签定义与计算

基础研究目标（不等于可实现收益）：

\[
R_{t,h}=\frac{P_{\mathrm{end}(t,h)}-P_t}{P_t},\quad
V_{t,h}=|R_{t,h}|
\]

其中 `P_t` 和 `P_end` 的选价规则、数据来源和使用时点必须在 Manifest 说明。

用于交易机会分析的净盈亏：

\[
\Pi_{\mathrm{long},t,h}=\operatorname{ExecutablePnL}(\mathrm{long},t,h;\mathrm{entry},\mathrm{exit},\mathrm{fees},\mathrm{limits})
\]

\[
\Pi_{\mathrm{short},t,h}=\operatorname{ExecutablePnL}(\mathrm{short},t,h;\mathrm{entry},\mathrm{exit},\mathrm{fees},\mathrm{limits})
\]

再定义：

\[
Y_{\mathrm{long},t,h}=\mathbb{1}(\Pi_{\mathrm{long},t,h}>0),\qquad
Y_{\mathrm{short},t,h}=\mathbb{1}(\Pi_{\mathrm{short},t,h}>0)
\]

P0 建议至少有以下研究标签：

| 字段 | 含义 | 必需条件 |
|---|---|---|
| `return_today` | 当前 TradingDay 收盘前的方向收益 | 有规范化参考价格 |
| `return_next_td` | 截至下一 TradingDay 日盘结束的累计方向收益 | 有规范化参考价格 |
| `abs_return_today` / `abs_return_next_td` | 同期绝对收益 | 同上 |
| `long_net_pnl_today` / `short_net_pnl_today` | 固定期限持多/持空的扣费结果 | **可执行价格 + 成本 + 交易规则** |
| `long_net_pnl_next_td` / `short_net_pnl_next_td` | 持至下一交易日结束的扣费结果 | 同上 |
| `long_profit_today` / `short_profit_today` | 净盈利布尔标签 | 上述多空净盈亏有效 |
| `long_profit_next_td` / `short_profit_next_td` | 下一交易日累计净盈利布尔标签 | 同上 |

**净盈亏标签研究口径**：先明确为“此刻新开一单位标准化仓位，按预先固定退出规则离场”。实际策略的“继续持有/平仓”依赖已有仓位、持仓成本、合法动作和风险状态，**不能直接用这个新开仓标签替代**。

**交易执行约束**：

- 入场不得早于 `decision_timestamp`，应使用决策发出后合理的第一个可执行报价或明确的成交仿真规则。
- 退出必须使用事前声明的离场规则。不能事后为单条样本选择最有利报价。
- 分别处理多空交易方向的 bid/ask、手续费、滑点、涨跌停、不可成交、换月与交易暂停。
- 如果只有 Bar 最后成交价而没有可实现报价，仍可给方向收益，但 `net_pnl` 与 `profit` 标签应置空并写明 `label_invalid_reason`。
- 对跨交割、流动性不足、无法完成出入场的合约样本，遵守**明确的无效/不可成交策略**，不得静默按理想价格填充。
- 保存成本模型版本、合约乘数及资金口径。扣费收益不可与历史旧仿真口径未经对齐就直接比较。

### 3.4 Today 剩余时长的分层问题

Today 标签的预测长度是可变的：例如开盘与临近收盘的预测难度不相同。

必须输出：

```text
remaining_tradable_minutes
session_id
within_trading_day_bar_index
```

最少报告全样本及按 **剩余可交易时间** 分层的结果。初期分箱可配置，例如 `<=60min`, `(60,180]min`, `>180min`，**仅为设计示例，不能当作项目已验证的阈值**。

可另开“每日固定决策时点”分析，但要明确为不同抽样协议。

---

## 4. 数据治理与绝对防泄漏规则

### 4.1 特征与标签必须物理隔离

```text
market_features     ← 只含在决策时刻已知的候选输入
research_labels     ← 含未来结果，只能供离线审计和训练标签使用
horizon_audit_output← 统计/归因产物，不应进入在线状态
```

不得用列名排除名单代替隔离；DataFrame schema/文件路径应有明确分层。Stage 8 只能消费通过正常特征列表许可的状态列。

### 4.2 时间切分和标签成熟（最重要的 P0 门禁）

对于 fold 的评价起点 `T_eval`，参与该 fold 的训练/拟合样本必须满足：

\[
\operatorname{label\_available\_at}(i)<T_{\mathrm{eval}}
\]

保守实现还需确保：

\[
\operatorname{label\_end}(i)<T_{\mathrm{eval}}
\]

无效例子：训练样本决策在 `2024-07-22 10:00`，NextTradingDay 标签才在 `2024-07-23` 收盘成熟，而验证窗口从 `2024-07-23 09:00` 开始。该样本不得用于针对这个验证窗口的训练/拟合/参数选择。

实现要求：

1. **同一日历时间轴**给所有合约做 Train/Valid/Test 切分，不得把同一日不同合约分到训练和最终测试以制造伪独立性。
2. 不能仅凭样本起点划分，也不能用固定根数代替按标签真实 `label_end/available_at` 的净化。
3. 标准化、分箱边界、缺失值处理参数、筛选阈值、监督基线模型及概率校准必须明确 `fit_scope`，严禁使用最终 Test 拟合。
4. 对**滚动因果特征**，Valid/Test 可以回看此前已经公开的 Train 历史行情；这本身不是泄漏。
5. 如果新的 Stage 6.5 使用开发验证标签进行特征决策，该 Valid 就是**开发集**；最终 Test 保持封存。
6. 记录原有 Stage 7 使用 validation outpost **特征分布**做 PSI 的已知行为，不能把整个既有流水线冒称纯 Train-only。
7. 若还要再进行概率校准及交易阈值选择，需要在开发数据内独立时间前推切出相应子区间。

### 4.3 研究样本过滤必须可追踪

任何无效样本必须计数并输出原因，例如：

- `NO_SESSION_CALENDAR`
- `NO_NEXT_TRADING_DAY`
- `NO_CUTOFF_PRICE`
- `PRICE_NOT_EXECUTABLE`
- `FEATURE_NOT_YET_AVAILABLE`
- `LABEL_NOT_MATURED`
- `CONTRACT_EXPIRED_OR_UNTRADEABLE`
- `INVALID_PRICE_OR_SIZE`
- `DUPLICATE_TIMESTAMP`
- `CROSS_BOUNDARY_PURGED`

不得悄悄填零或剔除后只报告剩余样本数。

---

## 5. Stage 6.5 的评估算法

### 5.1 研究对象和样本分母

- 输入：Stage 6 完成的数据，以及完整候选特征名、强制列、目前三流入选列表和现有 Manifest（如有）。
- 评估**全部合格候选特征**，并以 `is_selected`, `selected_streams`, `rejected_by_legacy_gate` 标记与现有筛选结果的关系。
- 每个 `contract_id × feature × horizon` 独立统计；不能把 14 个合约在相同日期的样本假定为独立观察。
- 无法满足最少有效样本量/有效交易日时输出 `INSUFFICIENT_DATA` 和实际覆盖率，而不是 IC=0。

### 5.2 有符号/绝对收益 RankIC

对于特征 `f` 和期限 `h`：

\[
IC(f,h)=\operatorname{SpearmanCorr}(f_t,R_{t,h})
\]

\[
VolIC(f,h)=\operatorname{SpearmanCorr}(f_t,|R_{t,h}|)
\]

记录：

- `rank_ic`, `vol_rank_ic`：分合约统计和汇总统计。
- `ic_sign`：方向符号与置信程度；正负相关不自动等于好坏。
- `sample_count`, `trading_day_count`, `coverage_rate`, `missing_rate`。
- `time_to_close_bucket`、`market_regime`（如有可用的**事前**区制代理）条件统计。
- `ic_by_contract` 与 `ic_by_time_fold`；跨合约均值与方差。

原有 Slope/RL 的方向 IC 与 Vol 流的绝对收益 IC 都应保留；不同流不能用一个通用方向 IC 一票否决。

### 5.3 跨合约一致性与稳定性

按报告中的历史门禁思路，计算合同级 IC 后的：

\[
SignConsistency(f,h)=\frac{\#\{c:\operatorname{sign}(IC_c)=\operatorname{sign}(\overline{IC})\}}{\#\{c:IC_c\ \mathrm{valid}\}}
\]

\[
ICIR(f,h)=\frac{|\overline{IC}|}{\operatorname{Std}(IC_c)+\epsilon}
\]

实现应明确：合同级聚合采用等权/样本加权哪种方案；强烈建议主报告**合同级等权**并同时披露覆盖率和样本数。没有有效跨合约变异时记录 `UNDEFINED`，不能凭除零修补输出“极高稳定性”。

### 5.4 净盈利相关诊断

只在成本和执行模拟可信时计算：

- 多空收益在特征分位数（如十等分）上的平均值、胜率、有效覆盖率。
- 按正、负值及极端分位统计方向性净期望。
- `long_net_expectancy`, `short_net_expectancy` 的不确定性区间。
- 不要因为一个特征与多空收益的简单线性相关弱，就认定其无风险管理价值。

如果净盈亏执行证据不够：

```text
net_profit_audit_status: UNAVAILABLE_EXECUTION_EVIDENCE
```

可继续输出价格收益 RankIC，但报告要将“统计预测能力”与“可交易净盈利能力”分开。

### 5.5 报告重叠标签造成的不确定性

Today/Next 标签通常相互重叠；每 10 分钟采样并不提供独立的隔日预测实验。

- 建议以共同 `TradingDay` 时间轴聚合预测评分差，再采用区块 bootstrap 或等价、可审计的时间依赖稳健估计。
- 多合约同一交易日的向量作为整体参与重采样，保留同日横截面相关性。
- 报告区块长度、随机种子、置信区间和时段划分，检查不同依赖假设的敏感性。
- `95%` 区间是建议统计约定，不是自动盈利通过门槛。
- 需检查跨牛熊、震荡、波动状态的稳定性，不能依赖单一期满报告。

### 5.6 预测期限错配分类：只做“审计发现”，不做自动动作

推荐报告每个特征如下指标：

| feature | IC_60min | IC_today | IC_next_td | legacy_selected | audit_finding |
|---|---:|---:|---:|---|---|
| `ask1_price_trend_2` | 来自当前报告或重新计算 | 待测 | 待测 | 是/否 | 待审计 |
| `ema_divergence_20_120` | 待核 | 待测 | 待测 | 是/否 | 待审计 |
| `trend_to_noise_96` | 待核 | 待测 | 待测 | 是/否 | 待审计 |
| `prev_week_return_quantile_rank` | 待核 | 待测 | 待测 | 是/否 | 待审计 |

建议 `audit_finding` 枚举：

```text
SHORT_HORIZON_CANDIDATE
TODAY_HORIZON_CANDIDATE
NEXT_TD_HORIZON_CANDIDATE
MULTI_HORIZON_CANDIDATE
CONFLICTING_SIGNS
INCONCLUSIVE
INSUFFICIENT_DATA
```

**这些分类在 MVP 中应仅作解释性的审计建议**；正式分类阈值须在查看未来测试前约定，并要求时间前推、跨合约稳定性和置信区间支持。不能用 `|IC|>0.02` 之类任意阈值自动断言收益价值。

---

## 6. 三流差异化审计逻辑

| 流 | 多期限核心指标 | 不能做的事 | 后续验证 |
|---|---|---|---|
| `vae_slope` | Today/Next 有符号收益关联、方向区制条件表现 | 方向 IC 低就直接删区制特征 | 重训 VAE 后比较区制分离与路由净收益 |
| `vae_volatility` | Today/Next 绝对收益、波动状态和风险分层 | 用涨跌方向 IC 统一硬筛 | 重训 VAE 后比较风险识别、熔断及路由增量 |
| `rl_decision` | 有符号收益、多空条件净期望、Today/Next 稳定性 | 把 IC 当作 RL 净收益贡献率 | 模型动作归因及完整策略分组消融 |

**注意**：`trading_minute_progress` 等时间/强制上下文列本身可能不通过普通 IC，却是训练和环境契约的一部分，不能由新审计绕过架构约束擅自剔除。

---

## 7. 推荐软件模块与职责边界（待创建）

仓库当前并未确认存在下列模块；目录结构是推荐实现，不是代码现状：

```text
data_preprocess/operator_futures/
└── feature_audit/                    # 新增，独立于既有 feature_selection
    ├── __init__.py
    ├── cli.py                        # P0 CLI、配置、审计只读入口
    ├── config.py                     # Schema/版本化配置
    ├── data_contracts.py             # 输入/标签/输出类型
    ├── trading_calendar.py           # TradingDay 与 Session 解析适配
    ├── horizon_targets.py            # Today/Next 目标终点
    ├── label_builder.py              # 价格/净盈亏研究标签
    ├── time_leakage_guard.py         # 可用时点和跨界净化
    ├── score_rank_ic.py              # 分期限/分合约 RankIC
    ├── pnl_opportunity_audit.py      # 可选净盈亏条件统计
    ├── stability.py                  # 一致性、区块置信区间
    ├── report.py                     # CSV/Parquet/JSON/MD 输出
    └── orchestrator.py               # 只读审计任务编排

data_preprocess/tests/
└── feature_audit/                    # 新增单元/集成测试
    ├── test_trading_calendar.py
    ├── test_horizon_labels.py
    ├── test_time_leakage_guard.py
    ├── test_rank_ic.py
    ├── test_report_contracts.py
    └── test_integration_fu10.py
```

**边界原则**：

- 交易日历实现优先**复用**现有日历和 Session 规则；必要时只写 adapter，不独立维护第二套交易日真值。
- 不复制 Stage 7 现有统计门禁；可复用现有 `predictive_audit.py` 的安全、明确接口，但新增计算需有独立版本记录。
- 不侵入 `FineFT/env/` 成交/资金引擎；净盈利标签若需要环境撮合，调用既有受测量的执行适配器，并记录执行假设与版本。
- 不写入或覆盖 `dataset/10min/fu/` 下的训练模型输入。审核产物放在独立 `analysis_result` 路径。
- 路径、函数名、CLI 只是提议；落地前根据实际仓库现状对齐，保留现有对外兼容性。

---

## 8. 输入/输出数据契约

### 8.1 输入清单

| 数据源 | 必需字段/内容 | 必要性 | 出错策略 |
|---|---|---|---|
| Stage 6 候选特征表 | `contract_id`, `event_timestamp`, 特征列 | 必需 | FAIL |
| Split Manifest | Train/Valid/Test 全局时间边界 | 必需 | FAIL |
| 交易日历/Session | 交易日、夜盘/日盘、收盘时刻、节假日 | 必需 | FAIL |
| 特征列目录 | 特征名、可用时点、来源周期 | 必需 | 缺失因果语义即 FAIL |
| 品种行情与参考价格 | 决策时与截止时参考价 | 基础收益必须 | 无效样本统计 |
| 合约乘数/交易规则/执行价 | 买卖报价、费率、滑点、涨跌停 | 净盈亏标签必须 | 净盈亏 UNAVAILABLE |
| Stage 7 选择结果 | `.npy` 及 Manifest | 建议 | 允许审计独立运行，但关联字段标 unknown |
| git/config/data fingerprints | 版本与输入 SHA256 | 必需 | FAIL 或显式标记不可复现 |

P0 执行时必须对历史数据基准进行校验。若候选池不等于 566，应输出真实数量及版本，不可强制截成 566。

### 8.2 最小行级标签契约（推荐 Parquet）

```text
contract_id                   string
symbol                        string
event_timestamp               timestamp[tz]
feature_available_at          timestamp[tz]
decision_timestamp            timestamp[tz]
trading_day                   date
horizon_id                    enum(today, next_td, 60min)
label_end                     timestamp[tz]
label_available_at            timestamp[tz]
entry_time                    timestamp[tz] nullable
exit_time                     timestamp[tz] nullable
reference_entry_price         float nullable
reference_exit_price          float nullable
signed_return                 float nullable
absolute_return               float nullable
long_net_pnl                  float nullable
short_net_pnl                 float nullable
long_profitable               bool nullable
short_profitable              bool nullable
remaining_tradable_minutes    int nullable
label_valid                   bool
label_invalid_reason          string nullable
split                         enum(train, valid, test)
```

`60min` 是比较基线，不应重写现有 Stage 7 的计算结果；如需重算，必须记录是否严格复现原价口径与 Bar 对齐逻辑，避免“同名不同定义”。

### 8.3 输出目录（建议）

```text
analysis_result/DiHFT/feature_horizon_audit/fu/10min/<run_id>/
├── audit_manifest.json
├── input_inventory.json
├── label_quality_summary.csv
├── horizon_labels.parquet
├── feature_horizon_metrics.csv
├── feature_horizon_by_contract.csv
├── feature_horizon_by_time_bucket.csv
├── feature_horizon_by_fold.csv
├── horizon_mismatch_candidates.csv
├── sample_exclusion_reasons.csv
├── leakage_audit.json
└── feature_horizon_audit_report.md
```

可将大数据量行级标签设为可选导出，但计算过程应可复现；不应为了保留 CSV 而造成不必要的内存占用或混入在线训练数据。

### 8.4 `feature_horizon_metrics.csv` 核心字段

```text
feature_name
feature_source
feature_family
selected_streams
is_selected
horizon_id
sample_count
trading_day_count
contract_count
coverage_rate
rank_ic_mean
rank_ic_median
rank_ic_ir
vol_rank_ic_mean
sign_consistency
ic_ci_lower
ic_ci_upper
long_net_expectancy
short_net_expectancy
net_pnl_status
remaining_time_bucket
audit_finding
finding_reason
```

对不存在、无定义或样本不足的数值使用 `null/NaN + status`，**不能用 0 冒充真实统计结果**。

### 8.5 `audit_manifest.json` 必需记录

```json
{
  "schema_version": "0.1.0",
  "audit_name": "multi_horizon_feature_audit",
  "mode": "audit_only",
  "symbol": "fu",
  "frequency": "10min",
  "horizons": ["60min", "today", "next_td"],
  "label_semantics": {
    "today": "current_trading_day_day_session_end",
    "next_td": "next_trading_day_day_session_end_cumulative_from_decision"
  },
  "split_source": "<dataset_split_manifest_path>",
  "calendar_version": "<calendar_hash_or_version>",
  "feature_source_hash": "<sha256>",
  "existing_feature_manifest_hash": "<sha256_or_null>",
  "cost_model_version": "<version_or_not_available>",
  "reference_price_policy": "<name>",
  "execution_price_policy": "<name_or_not_available>",
  "git_commit": "<git_sha>",
  "config_sha256": "<sha256>",
  "random_seed": 42,
  "evaluation_split_used_for_fitting": "train_only_or_explicit_development_folds",
  "test_used_for_feature_selection": false,
  "created_at": "<utc_iso8601>",
  "quality_status": "<PASS|NEEDS_AUDIT|FAIL>",
  "artifacts": {}
}
```

上面字段及 `42` 为**示意接口**，不是项目既定配置。实现应根据真实数据和审计规则将占位符替换为完整值。

---

## 9. CLI、配置与调用行为（建议）

### 9.1 审计入口

```bash
python -m data_preprocess.operator_futures.feature_audit.cli \
    --config configs/feature_audit/fu_10min_multi_horizon.yaml \
    --mode audit-only
```

此命令为**拟新增接口**；当前仓库中不存在该命令的保证。

### 9.2 配置示例

```yaml
schema_version: "0.1.0"
audit_name: multi_horizon_feature_audit
mode: audit_only
symbol: fu
frequency: 10min

inputs:
  dataset_root: dataset/10min/fu
  source_stage: after_clean_and_split_before_feature_selection
  include_all_candidate_features: true
  include_current_selected_features: true

horizons:
  - id: 60min
    reference_only: true
    bars: 6
  - id: today
    cutoff: current_trading_day_day_session_end
  - id: next_td
    cutoff: next_trading_day_day_session_end
    from: decision_timestamp

time:
  timezone: Asia/Shanghai
  exchange_calendar_source: reuse_project_calendar
  enforce_feature_available_at: true
  purge_cross_boundary_labels: true
  remaining_time_buckets_minutes: [60, 180]

labels:
  signed_return: true
  absolute_return: true
  fixed_horizon_long_short_net_profit: auto_if_execution_data_available
  reference_price_policy: declare_and_version
  execution_policy: declare_and_version
  missing_label_policy: null_with_reason

statistics:
  spearman: true
  per_contract: true
  cross_contract_sign_consistency: true
  time_forward_folds: true
  bootstrap_on_trading_day_blocks: true
  report_confidence_level: 0.95  # 待研究协议冻结的项目建议

safety:
  allow_test_for_feature_selection: false
  overwrite_existing_feature_lists: false
  alter_stage7_selection: false
  alter_model_checkpoints: false

outputs:
  result_root: analysis_result/DiHFT/feature_horizon_audit
  export_parquet: true
  export_csv: true
  export_markdown: true
```

配置中的 `dataset_root` 是历史示例目录；开发时应以当前实际 Stage 6 产物位置与 Split Manifest 为准。

### 9.3 任务执行流程伪代码

```python
# 伪代码：接口名称均为建议，非已存在仓库实现

def run_multi_horizon_audit(config):
    inventory = collect_and_hash_inputs(config)
    split = load_global_time_split(inventory)
    calendar = load_existing_project_calendar(config.symbol)
    candidates = load_stage6_candidate_features(inventory)
    selected = load_optional_legacy_three_stream_manifest(inventory)

    assert_read_only_mode(config)
    validate_point_in_time_feature_availability(candidates)
    validate_global_split_and_calendar(split, calendar)

    targets = build_horizon_targets(
        candidates, calendar, horizons=["60min", "today", "next_td"]
    )
    labels = build_research_labels(
        targets, price_policy=config.labels.reference_price_policy,
        execution_policy=config.labels.execution_policy
    )
    leakage_report = audit_and_purge_boundary_labels(labels, split)

    metrics = calculate_per_contract_horizon_metrics(
        candidates, labels, leakage_report, fit_scope="development_only"
    )
    metrics = aggregate_and_estimate_uncertainty(metrics)
    findings = compare_horizon_patterns(metrics, selected, mode="descriptive")

    write_versioned_artifacts(
        inventory, targets, labels, leakage_report, metrics, findings
    )
    assert_original_model_inputs_unchanged()
```

如果数据很大，允许采用按合约/交易日分区、流式载入和增量归并统计；但必须确保与全量基准在定义上等价，并提供小样本一致性测试。

---

## 10. 审计报告必须回答的业务问题

生成的 `feature_horizon_audit_report.md` 至少回答：

### Q1. 现有特征中哪些偏向 60min？

- 在两个新预测期限中 IC 显著弱化的特征数量和名单。
- 哪些短线微观特征只应作为执行辅助的**候选假设**。
- 是否存在方向符号从 60min 到 NextTradingDay 反转的情况。

### Q2. 哪些特征可能适用于 Today 或 NextTradingDay？

- 分期限的方向 IC、VolIC、跨合约符号一致性、时间前推稳健性。
- 近收盘与开盘时的 Today 表现是否不同。
- 有统计相关但尚无可执行净盈利证据的特征必须明确标出。

### Q3. 现有 60min 门禁遗漏了什么？

- 在全部合格候选中，不在现有 165 并集、却可能在 Today/Next 表现较好的特征。
- 追踪每个特征原先被哪个阶段淘汰；若无淘汰原因，报告 `UNKNOWN_LEGACY_REASON`。
- 不在 MVP 中自动将这些特征写进三流 `.npy`。

### Q4. 三流各自应如何解读？

- Slope 的方向识别；Vol 的未来绝对收益/风险分层；RL 的可交易机会。
- 对强制上下文列、纯执行辅助列和波动风险特征给出不应统一使用方向 IC 的说明。

### Q5. 结果是否可以用于下一阶段？

- 审计资格：时间可用性、数据与模型版本、完整样本分母、执行可达性是否满足要求。
- 建议下一轮需要在训练后/完整模拟交易中验证的特征族，以及证据不足项。

---

## 11. 测试策略与验收标准（Definition of Done）

### 11.1 P0 必测用例

| 测试编号 | 场景 | 期望结果 |
|---|---|---|
| T01 | 夜盘周一 21:00 决策 | 映射到交易所定义的下一个 TradingDay，非周一自然日 |
| T02 | 周五夜盘/周末/法定节假日 | 通过交易日历确定 next_td，不按日期 `+1` |
| T03 | 临近当日收盘决策 | Today 剩余交易分钟数准确；无可执行退出时正确无效化 |
| T04 | 下一交易日无行情/临时休市 | 标签缺失并给出理由，绝不填未来价格 |
| T05 | 未来标签跨越 Valid 起点 | Train 中该样本被 purge，记录计数 |
| T06 | 某特征在决策之后才发布 | 特征被拒绝用于该时刻 |
| T07 | 同一日不同合约 | 同一全局时间窗口，区块统计保留共同日相关性 |
| T08 | 常数列/高度缺失列 | IC 为 `null + UNDEFINED/INSUFFICIENT_DATA`，非伪造 0 |
| T09 | 单特征全升序、目标升序 | Spearman IC 接近 +1；逆序接近 -1 |
| T10 | Today vs Next 累计目标 | 两个 cutoff 正确、标签不同，不被 60min 取代 |
| T11 | 参考价有效但缺盘口/执行成本 | 价格 IC 可得，净盈亏标签 `UNAVAILABLE` |
| T12 | 涨跌停/无法成交 | 不按参考价假设一定成交，保留无效原因 |
| T13 | 某合约重复或时间倒序 | 按规则去重或 FAIL，并保存证据 |
| T14 | Manifest 特征列表 `12` vs 实际 `.npy` `13` | 报告版本/维度不一致；不得静默覆写 |
| T15 | 默认 `audit_only` | 现有 `.npy`/scaler/checkpoint 哈希不变 |
| T16 | 对 Test 注入不同未来收益 | 开发期筛选/排名不能变化；Test 不参与任何拟合 |
| T17 | 相同数据、配置和种子重跑 | 指标与分类可复现，版本变化可定位 |
| T18 | 特征来自前一已完成交易日 | 正确允许历史回看；不能误判为未来泄漏 |

### 11.2 研究层面质量门禁

`PASS` 表示**数据/审计流程通过**，不表示策略盈利合格。建议使用：

- `PASS`：输入、标签、可用时点、split、计算和产物可审计。
- `NEEDS_AUDIT`：可给出部分描述结果，但执行成本、特征版本或某些拟合时间证据不完整。
- `FAIL`：严重前视泄漏、split 无法确认、关键交易日时钟错误、标签/数据无法可靠对齐。

每个指标还应有自己的 `AVAILABLE / INSUFFICIENT_DATA / UNDEFINED / UNAVAILABLE_EXECUTION_EVIDENCE` 状态。

**不定义**“RankIC 大于 X 就必须入选”这样的新硬门槛；如果需要，必须先形成独立研究协议，并在最终 Test 之前冻结。

### 11.3 文档与兼容性验收

- [ ] 可以在不训练 RL/VAE 的情况下独立运行 Stage 6.5。
- [ ] 能从现有 Stage 6 数据批量生成 Today / NextTradingDay 标签。
- [ ] 全部候选特征都进入审计，并标注历史入选/淘汰状态。
- [ ] 能对不同合约、期限、时段与剩余交易时间分组评分。
- [ ] 所有未来标签都具有 `label_end` 和 `label_available_at`。
- [ ] 开发期拟合过程没有使用最终 Test。
- [ ] 产生 `feature_horizon_metrics.csv` 和 `feature_horizon_audit_report.md`。
- [ ] 报告显示源数据哈希、配置/代码版本、交易日历及价格成本口径。
- [ ] 源码包括单元测试、集成测试、基本运行说明。
- [ ] 现有 Stage 7 和 Stage 8 的产物不受默认审计任务影响。
- [ ] 价格预测能力与可执行净利润能力被明确分开。

---

## 12. 推荐开发实施顺序（任务拆解）

| 顺序 | 工作包 | 交付物 | 验收重点 |
|---|---|---|---|
| 1 | 审计现有 Stage 6 数据契约、交易日历、三流清单及价格时点 | 输入清单与差异报告 | 无凭空假设的路径/字段 |
| 2 | 实现标签终点计算器 | `horizon_targets.py` + 测试 | 夜盘、节假日、Today/Next 正确 |
| 3 | 实现研究标签与有效性判断 | `label_builder.py` + 样本质量表 | 参考价格与成交口径分离 |
| 4 | 实现 PIT/标签成熟净化 | `time_leakage_guard.py` + 报告 | Train/Valid 边界无跨界标签 |
| 5 | 实现分合约 RankIC/VolIC | `score_rank_ic.py` | 单调数据、缺失、常数测试通过 |
| 6 | 实现稳定性/覆盖率/分层 | `stability.py` | 不把重叠 Bar 当独立样本 |
| 7 | 关联历史三流 Manifest，生成错配候选 | `report.py` | 可追踪为何入选/淘汰 |
| 8 | 实现 CLI/Manifest/只读防护 | `cli.py` + `audit_manifest.json` | 版本记录完备、原产物不变 |
| 9 | `fu/10min` 真实小样本集成测试 | 第一版 MD + CSV 报告 | 结果可复核、不误用 Test |
| 10 | 明确审计结果如何进入下一轮研究 | 后续 E1/E2 研究计划 | 不自动更改模型输入 |

**推荐首个可交付版本**：完成步骤 1–9 即可用来发现预测期限错配；不需要同步实现利润归因、Q-Net 解释器和完整特征消融。

---

## 13. Stage 6.5 之后如何衔接训练和模拟交易

### 13.1 训练前：预测相关性筛查

回答：

> 哪些特征对 Today/Next 有统计预测关联？哪些在 60min 强而隔日弱？哪些被原门禁遗漏？

**无需**训练 RL/VAE，阶段位置为 Stage 6 之后、Stage 7 之前。

### 13.2 训练后：模型是否利用预测信息

回答：

> 这些特征进入网络后，是否改善了有效预测或改变了决策？

当前 Q-Net 输出的是动作 Q 值，不是已校准的 Today/Next 盈利概率。若要直接测试未来盈利概率，需要另建显式预测头或监督基线，并用独立开发时段验证。可以观察条件下 Q 值排序、动作概率/分布和 VAE 路由变化，但不能把这些解释值冒称为盈利贡献。

### 13.3 模拟交易后：真实收益贡献

对特征组 \(G\) 做**重新训练式消融**：

\[
\Delta\operatorname{NetPnL}_G=
\operatorname{NetPnL}(\pi_{\mathrm{full}})-
\operatorname{NetPnL}(\pi_{-G})
\]

在相同时间外数据、交易成本、风险规则和可复现实验条件下比较净收益、回撤、换手、动作等。三流 VAE 的特征消融必须重训 VAE 并重新做评分/路由校准，不能只把一个输入特征置零。不同组的增量不可简单相加。

**结论**：Stage 6.5 负责**发现信息**；训练后审计负责**确认模型会用**；交易消融负责**确认使用它是否更赚钱**。三者必须分开。

---

## 14. 风险与已知边界

1. **时间语义风险**：夜盘属于下一 TradingDay；期货日盘终点不是自然日午夜。
2. **Stage 6/7 现状风险**：原筛选可能使用验证前哨的特征分布；新功能的可解释性不能掩盖原先依赖。
3. **执行证据风险**：历史 Trial 的参考价与盘口对齐、成交可达性、费用残差未完全审计，不得直接复用其“净收益”为精确标签。
4. **同一合约跨 split**：允许不同时间段重现，但必须避免标签跨界及拟合边界错误。
5. **大量多重检验**：566 个候选 × 多期限 × 多合约/时段会产生选择偏差；应记录全部尝试，不在最终 Test 上反复筛选。
6. **预测视界多义**：`next_td` 必须保留“从 t 累计到下个交易日结束”的语义，不等同“下一个交易日从开盘到收盘”的单日收益。
7. **分布与输入版本**：要核对 `.npy` 特征顺序、checkpoint 对应关系，以及 Manifest 12 vs 13 的差异。
8. **有限历史不保证盈利**：即使审计和消融通过，也只能表述特定品种、时段、执行假设下的研究证据。

---

## 15. 给开发人员的最终工作指令

> **请在不修改现有训练、三流选择和回测结果的前提下，实现可独立运行的 Stage 6.5 多预测期限特征审计模块。**
>
> 以当前 `fu/10min` Stage 6 候选特征与时间切分为输入，按商品交易所 TradingDay/Session 日历构造 `Today` 和 `Next TradingDay` 两类离线研究标签，并保留 `60min` 基线。要求严格的 point-in-time 可用性、未来标签成熟和时间切分净化，对全部候选特征按合约及预测期限计算 RankIC、VolRankIC、稳定性、覆盖率和必要的时间分层。可执行净盈利标签仅在执行价与费用证据完整时输出；缺失时必须明确标识。关联现有三流 Manifest，输出预测期限错配候选与未入选但可能对更长期目标有价值的特征清单。交付标准化 CSV/Parquet/JSON/Markdown、完整版本溯源、自动化测试和运行说明。MVP 只能审计，默认不得重写现有特征列表或使用最终 Test 来选择特征。

---

## 16. 本文依据的文件（用户提供）

以下路径用于回溯**现有项目记录**；本文没有声称代码已按新规范实现：

1. `2026-10-11-fineft-project-architecture-v2.zh_cn(2).md`：现有架构、Stage 流程、数据切分、三流列名清单、模型及验证边界。
2. `2026-10-12-commodity-futures-feature-engineering-and-3stream-selection.zh_cn.md`：候选特征工程及当前 `w=6` 的 RankIC/VolRankIC 筛选定义。
3. `2026-10-12-fu-10min-feature-selection-manifest-attribution.zh_cn.md`：历史 `fu / 10min` 的 566 候选、17 强制、165 并集及各流数量。
4. `2026-10-12-dihft-latest-experiment-diagnostics-report.zh_cn(4).md`：Trial 229、`future_horizon_predictor: none`、`NEEDS_AUDIT` 及执行/元数据缺口。
5. `2026-10-12-dihft-trial342-experiment-diagnostics-report.zh_cn(1).md`：旧验证结果审计、时间外独立性与对账限制。
6. `2026-10-12-multiscale-trading-validation-sources.zh_cn(1).md`：TradingDay 时间定义、跨界标签净化、概率校准、区块 bootstrap、多重选择偏差和冻结测试原则。

上述文档中，**Stage 6.5、拟新增的代码目录/CLI、文件 schema、质量状态、测试用例以及未来扩展方案**均属本次开发提案，不是已有实现的事实描述。
