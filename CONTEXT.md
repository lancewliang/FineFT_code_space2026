# FineFT

FineFT is the futures-trading research context for the three-stage risk-aware ensemble reinforcement learning pipeline described by "FineFT: Efficient and Risk-Aware Ensemble Reinforcement Learning for Futures Trading".

## Language

### Data And Preprocessing

**主力合约 (Main Contract)**:
按自然月成交量最高的前 2 个合约加上高成交量天数入选合约的并集，用于拼接连续主力数据。
_Avoid_: 主力连续、连续合约

**主力合约日文件 (Main Contract Daily File)**:
按 `TradingDay` 拆分的主力合约 CSV 文件，路径格式为 `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv`。
_Avoid_: 连续主力文件、日度连续数据

**主力合约 Summary (Main Contract Summary)**:
描述日期范围内入选合约集合、交易日窗口、源文件路径、每个合约最后交易日和完整交易日数量的 JSON 文件。
_Avoid_: summary JSON、主力合约 JSON

**交易日 (TradingDay)**:
交易所归属的交易日，决定源文件归属和日级别统计边界。
_Avoid_: 交易日期、business day

**事件时间戳 (Event Timestamp)**:
由 `ActionDay + UpdateTime` 构成的真实事件发生时间，用于下采样排序和窗口聚合。
_Avoid_: 时间戳（不明确时）、action time

**合约交易单位 (Contract Unit)**:
商品期货每手合约对应的数量乘数，用于将 Turnover/Volume 修正为正确价格口径。
_Avoid_: 乘数、合约乘数

**交易 Session (Trading Session)**:
商品期货的有效交易时段，用于判断 quote gap 是否属于跨 session 缺口。
_Avoid_: 交易时段、session

**下采样 (Downscale)**:
从秒级原始快照聚合到目标频率（如 5min）的过程，使用右闭右标窗口语义。
_Avoid_: 降采样、聚合

**右闭右标窗口 (Right-closed Right-labeled Window)**:
区间 `(t-k, t]` 内的快照聚合到标记为 `t` 的 bar，是 FineFT 下采样的标准窗口语义。
_Avoid_: 右开窗口、左闭窗口

**五档行情 (5-Level Orderbook)**:
商品期货仅使用第 1 到第 5 档买卖价格和数量，不合成第 6 到第 25 档。
_Avoid_: 浅档、5 档盘口

**参考价 (Reference Price)**:
商品期货环境使用的 mark_price 和 index_price，优先取 LastPrice，回退到 midprice。
_Avoid_: 标记价、参考价格

**Tick Rule**:
通过比较相邻秒均价判断成交方向（up/down/flat）的估计方法，输出 buy_estimated 和 sell_estimated。
_Avoid_: 成交方向规则、tick 方向

**Fail-fast**:
遇到数据异常（缺失列、非有限值、空结果）时立即报错停止，不静默跳过或填充。
_Avoid_: 快速失败、报错中断

**NaN 校验 (NaN Validation)**:
Scale Save 前后对 State Feature 进行的 NaN 检查，发现 NaN 时立即报错并输出含 NaN 的特征名和行号，防止下游训练静默失败。
_Avoid_: NaN 检查、空值校验

### Feature Engineering

**截面特征 (Cross-section Feature)**:
从单条快照或单 bar 数据直接计算的 KLINE、QUOTE 和 SNAPSHOT 特征。
_Avoid_: 横截面特征、快照特征

**Base_Time_feature**:
与 Base Feature 平级的商品期货时间编码特征产物，描述交易时间分钟、早盘/下午盘/夜间盘及开收盘半小时、合约所在月份和合约剩余生命周期；属于必须保留的 State Feature，不能被 Feature Selection 过滤掉，也不参与 Scale Save 缩放。
_Avoid_: 时间滚动窗口特征、绝对时间特征、日历特征

**Trading Session 内进度 (Trading Session Progress)**:
当前 timestamp 在所属 Trading Session 内按 session 持续分钟数归一化得到的进度，不跨午休或非交易空档连续计算。
_Avoid_: 全天分钟进度、绝对分钟数

**合约交割月份 (Contract Delivery Month)**:
合约代码中表示交割月份的月份字段，如 `fu2605` 中的 `05`。
_Avoid_: 当前交易日月份、自然月

**合约最后交易日 (Contract Last Trading Day)**:
在选择合约数据文件时，从原始下载中该合约全部 `TradingDay` 取最大值得到的真实最后交易日，用于计算合约剩余生命周期。
_Avoid_: 主力合约窗口结束日、样本结束日

**合约完整交易日数量 (Contract Total Trading Day Count)**:
在选择合约数据文件时，从原始下载中该合约全部不同 `TradingDay` 计数得到的交易日总数，用作合约剩余生命周期比例的分母。
_Avoid_: 主力窗口交易日数量、样本交易日数量

**合约剩余生命周期比例 (Contract Life Remaining Ratio)**:
当前 `TradingDay` 到合约最后交易日的剩余交易日数量除以合约完整交易日数量得到的非绝对生命周期特征。
_Avoid_: 剩余天数、自然日倒计时

**滚动窗口特征 (Rolling Window Feature)**:
基于历史窗口滚动计算的衍生特征，如移动平均、波动率等。
_Avoid_: Base_Time_feature、时序特征

**混频状态特征 (Mixed-frequency State Feature)**:
在目标频率 bar 上注入日级和周级市场状态的 State Feature；第一版日级仅使用上一 `TradingDay`，周级仅使用上一完整自然周。
_Avoid_: 周线因子、日周拼接特征、当前日完整特征、当前周完整特征

**混频基础数据 (Mixed-frequency Base Data)**:
用于生成混频状态特征的低频 OHLCV 基础产物；日基础数据每个 `TradingDay` 一行，周基础数据每个自然周一行。
_Avoid_: 按 bar 展开的日周基础特征、临时低频聚合

**自然周状态特征 (Calendar-week State Feature)**:
按 `TradingDay` 所属自然周聚合得到的周级混频状态特征，当前自然周内的日内 bar 只能使用上一完整自然周。
_Avoid_: 五交易日滚动周特征、当前周完整特征

**上一周期状态特征 (Previous-period State Feature)**:
仅使用上一完整交易日或上一完整交易周计算的日级或周级状态特征，用于给当前 5min bar 提供已完成周期背景。
_Avoid_: 昨日特征、上周特征（不明确时）

**混频可见性约束 (Mixed-frequency Visibility Rule)**:
混频状态特征在任一目标频率 bar 上只能使用上一完整日或上一完整周统计，不能使用当前未完成日、当前未完成周或未来 bar 的统计。
_Avoid_: forward-fill 周特征、未来可见特征

**风险状态特征 (Risk State Feature)**:
基于 OHLC 和收益率历史窗口计算的波动率类 State Feature，如 ATR%、Historical Volatility、Rolling Volatility、Parkinson Volatility、Garman-Klass Volatility 和 Realized Volatility。
_Avoid_: 交易过程特征、账户风险特征

**流动性状态特征 (Liquidity State Feature)**:
基于成交量、成交额和持仓量历史窗口计算的市场活跃度 State Feature，如 Relative Volume、Relative Amount、Relative Open Interest 和 Open Interest Change Ratio。
_Avoid_: 盘口深度特征、Microstructure 特征

**日内 Bar 数 (Bars Per Day)**:
根据商品期货品种 `Trading Session` 总交易时长和目标频率推导的每日 bar 数，用于 Historical Volatility 等日化计算。
_Avoid_: 24 小时固定 bar 数、自然日 bar 数

**Reward/Execution 列**:
环境执行所需的非训练列，包括 orderbook 深度列、涨跌停价、derivative reference 列等，不参与特征选择和缩放。
_Avoid_: 奖励列、执行列

**State Feature**:
经过特征选择后用于 RL agent 观测的训练特征，由 `state_features.npy` 记录。
_Avoid_: 状态特征、观测特征

**Feature Selection**:
通过 IC、RankIC、CatBoost Importance、Permutation Importance、Sharpe 等指标评估和筛选 State Feature 的流水线。
_Avoid_: 特征筛选、因子选择

**Feature Selection Manifest**:
Feature Selection 输出的 JSON 清单，记录候选特征来源、IC 结果路径、过滤配置、mandatory state feature 和最终特征列表。
_Avoid_: 特征选择清单、选择描述

**Feature Union**:
将多个合约的 candidate State Feature 去重合并为品种级统一特征列表的过程。
_Avoid_: 特征合集、特征合并

**Feature Union Manifest**:
Feature Union 输出的 JSON 清单，记录品种、合约列表、各合约特征数、State Feature 列表和输出路径。
_Avoid_: 特征联合清单、联合描述

**Scale Save**:
使用 train-only robust scaler 对 State Feature 进行标准化并裁剪到 `[-20, 20]`，输出下游可消费的 feather 和 csv 文件。
_Avoid_: 缩放保存、标准化输出

**Scale Manifest**:
Scale Save 输出的 JSON 清单，记录 scaler 版本、拟合范围（`fit_scope="train_all_contracts"`）、passthrough state feature、特征统计和裁剪配置。

**订单流不平衡 (Order Flow Imbalance, OFI)**:
基于 Level-1 至 Level-5 买卖双边挂单价格与挂单量变动计算的净订单流注入指标，通过盘口总深度或成交量进行归一化。
_Avoid_: 买卖挂单差、OFI 指标

**盘口耗竭与恢复 (Depth Depletion and Replenishment)**:
描述买卖盘口深度被大单冲击后的瞬间衰减比例 (Depletion) 及随时间恢复至历史均值的相对速率 (Replenishment)。
_Avoid_: 深度消耗、挂单恢复

**价差扩大状态 (Spread Widening Dynamics)**:
基于买卖一价差相对中间价的比率及其滚动历史 Z-Score 衡量的市场流动性冲击与做市退场特征。
_Avoid_: 点差扩大、Spread 异常

**成交方向持续性 (Trade Directional Persistence)**:
基于主动买卖成交量净额比率与方向连续性衰减平滑衡量的单边买卖盘力量推升连贯性。
_Avoid_: 主买主卖比、成交持续性

**趋势加速度 (Trend Acceleration)**:
价格一阶变化速度（如 EMA 斜率/差分）的二阶导数，除以历史波动率进行标准化，用于捕捉趋势见顶、见底或加速突破。
_Avoid_: 价格加速度、MACD 导数

**波动率 Regime (Volatility Regime Indicator)**:
描述无偏波动率（如 Garman-Klass、Parkinson）在滚动历史窗口内的连续分位数百分比分值。
_Avoid_: 波动率状态、波动率区间

**成交量持仓量 Regime (Volume Open Interest Regime)**:
结合价格变动、成交量与持仓量增减方向的连续三元交互特征，用于区分主力增仓建仓与平仓止损驱动。
_Avoid_: 量价持仓状态、量持仓 Regime

**跨月价差动态变化 (Cross-Month Spread Dynamics)**:
主力与次主力合约 Log 价差变化率及跨月持仓份额转移速率的动态特征。
_Avoid_: 跨月价差变动、套利价差加速度

_Avoid_: 缩放清单、scaler 描述

**OFI (Order Flow Imbalance)**:
从连续五档 quote 快照计算的订单流不平衡指标，按固定行数窗口聚合输出。
_Avoid_: 订单流、order flow

**Microstructure 特征**:
从一档 quote 快照派生的微观结构特征，包括 microprice pressure、relative spread、spread 变化计数和 queue pressure。
_Avoid_: 微观结构特征、微观指标

**Queue Pressure**:
基于一档价格不变时数量增减判断的队列补充/撤单压力指标。
_Avoid_: 队列压力、挂单压力

**Imbalance**:
盘口买卖量不平衡指标，支持 1 档、3 档和 5 档深度计算。
_Avoid_: 不平衡度、买卖失衡

**单边盘口 (Single-sided Orderbook)**:
买侧或卖侧一方数量为零但对方有效的合法盘口状态，需生成有限特征而非 NaN。
_Avoid_: 单侧盘口、一边空

**涨跌停价 (Limit Price)**:
LowerLimitPrice 和 UpperLimitPrice，属于 Reward/Execution 列，不进入 state candidate。
_Avoid_: 涨跌停、价格限制

**跨月合约结构特征 (Cross-Month Term Structure Feature)**:
基于多个合约（如主力合约与次主力合约，或到期月份序列 $M_1, M_2, M_3$）在同一时间 Bar 下的相对变动特征。它表达商品期货跨交割月期限结构，而不是单一合约自身的价格行为。
_Avoid_: 跨期特征、跨合约衍生指标

**主力/次主力动态配对 (Main-Sub Dynamic Pairing)**:
按当前交易日或交易月的流动性排名确定主力合约（`main`）与次主力合约（`sub`）身份的配对规则。
_Avoid_: 静态主力配对、固定合约配对

**到期月份序列配对 (Delivery Month Sequence Pairing)**:
按合约真实交割月份由近到远排序确定近月（$M_1$）、次近月（$M_2$）与远月（$M_3$）的配对规则。
_Avoid_: 挂牌顺序配对、自然月配对

**无绝对价格约束 (No Absolute Price Rule)**:
跨月合约结构特征不得表达绝对价格水平或原始价格差。包含价格的跨月表达必须是无量纲、相对化或平稳化的形式。
_Avoid_: 绝对价差、原始价格差

### Dataset And Splitting

**Dataset Split**:
按时间边界将数据分为 train/valid/test 集合的过程，输出 `dataset_split_manifest.json`。
_Avoid_: 数据切分、数据分割

**Dataset Manifest**:
描述 FineFT 数据集的合约级输入输出、行数、State Feature 路径和切片计划的 JSON 文件。
_Avoid_: 数据集清单、数据集描述

**Slice Manifest**:
描述 valid 阶段动态标签切片的合约视角和 label 视角聚合信息的 JSON 文件。
_Avoid_: 切片清单、切片描述

**Skipped Contract**:
在某个集合（如 valid）中没有命中交易日或数据不足的合约，manifest 中记录跳过原因。
_Avoid_: 跳过合约、缺失合约

**Train Slice**:
从 train 阶段合约数据按 chunk_length 切分的连续编号训练片段，不跨合约。
_Avoid_: 训练切片、训练分块

**Valid 动态切片 (Valid Dynamic Slice)**:
对 valid 阶段数据逐合约执行市场动态标签切片，输出 `valid/<contract>/label_*/df_*.feather`。
_Avoid_: 验证切片、验证分块

**合约级 Valid Feature (Contract-level Valid Feature)**:
valid 阶段按合约保存的完整特征文件，路径格式为 `valid/<contract>.feather`，用于合约级回测或路由评估。
_Avoid_: valid 特征、验证集 feature（不明确时）

### Reinforcement Learning Pipeline

**Stage I (低层训练)**:
训练价值基低层 agent 集成，使用选择性更新（ensemble TD error 驱动）和预训练 warmup。
_Avoid_: 第一阶段、低层阶段

**Stage II (Agent 筛选与 VAE)**:
在验证市场动态下回测/筛选 agent，训练 VAE 用于能力边界/OOD 检测。
_Avoid_: 第二阶段、VAE 阶段

**Stage III (风险感知路由)**:
使用 VAE 重构损失在滚动窗口上路由筛选后的 agent 和保守策略。
_Avoid_: 第三阶段、路由阶段

**Full-df Warmup**:
Stage I 训练前对每个训练分块使用空仓初始动作执行 DP 专家路径 warmup，直接更新网络参数。
_Avoid_: 全量 warmup、df warmup

**Qtable 预计算**:
训练循环前多进程预计算所有唯一 df_index 的最优 Q 表，缓存后供预训练使用。
_Avoid_: Q 表预计算、qtable 缓存

**DP Expert Action Path**:
从 Q 表和初始动作推导的动态规划最优动作序列，用于预训练 warmup 和盈利诊断。
_Avoid_: 专家路径、最优路径

**Diverse Training (多样化训练)**:
Stage I 中使用随机初始动作的探索训练阶段，与预训练 warmup 区分。
_Avoid_: 探索训练、随机训练

**Potential Model**:
低层 agent 选择后组装的模型文件，每个 label 对应一个选中的 qnet。
_Avoid_: 潜力模型、候选模型

**Selection Manifest**:
记录低层 agent 最终选择的 label、epoch、bin_index 和分数的 JSON 文件。
_Avoid_: 选择清单、选择描述

### VAE And Routing

**VAE 跨合约训练 (VAE Cross-contract Training)**:
从多合约训练数据合并生成统一 VAE 训练集的过程，通过物化 label 训练数据校验特征维度一致性。
_Avoid_: 跨合约 VAE、多合约 VAE

**VAE Training Manifest**:
描述跨合约 label 训练数据物化结果的 JSON 文件，包含 included/missing contracts。
_Avoid_: VAE 训练清单

**Label Summary**:
VAE 分析后每个 label 的测试合约 logpx 统计、分位数和 acceptance 指标。
_Avoid_: 标签摘要

**Routing Summary**:
跨 label 的路由胜出统计，记录每个测试合约和总体的 winner 分布和 margin 信息。
_Avoid_: 路由摘要

**OOD Detection**:
通过 VAE 重构损失识别超出训练分布的市场状态，触发保守策略。
_Avoid_: 异常检测、分布外检测

**Agent 策略原型档案库 (Agent Archetype Profile)**:
为每个 VAE Label 维护包含 12 大策略原型（动量、均值回归、盘口失衡、持仓量驱动等）离线择优选出的专属 Agent 智囊团档案。
_Avoid_: Agent 档案、策略池

**PnL 记忆追踪器 (PnL Memory Tracker)**:
动态记录各 Agent 近 50 步滚动收益与胜率的全局记忆池，输出综合 PnL 得分与近端回撤状态。
_Avoid_: 收益追踪器、Agent 记忆

**候选池生成器 (Candidate Generator)**:
接收门控 Label 智囊团 Agents 的拟执行动作，应用 Label 固有动作语义一致性校验 (Semantic Guard) 与 20% 近端 PnL 回撤硬隔离后生成安全候选集。
_Avoid_: 候选集筛选器、动作过滤器

**Label 方向语义 (Label Direction Semantics)**:
由 slope 市场动态切片规则赋予每个 Label 的方向与强度，权威记录在 `label_semantics.json`（`direction` / `direction_sign` / `strength`）。Label 索引按方向单调有序：`label_0` 跌停(strong_down) → `label_1/2` 下跌 → `label_3` 震荡 → `label_4/5` 上涨 → `label_6` 涨停(strong_up)；路由实际使用 7 个 Label（label_0~label_6），非 5 个。该表是 Semantic Guard 判定 Label 原生动作范围的唯一来源；表本身静态，推理期 VAE 仅观测当前 state，无未来泄漏。
_Avoid_: Label 含义、标签语义、label direction

**语义硬隔离 (Semantic Guard)**:
强制校验 Agent 动作是否符合所属 Label 的原生动作语义范围（如 Label 4 上涨方向禁止做空）；违者不再一票否决，改为按软惩罚计入 Meta Router 得分。原生动作范围由 Label 方向语义表决定。
_Avoid_: 动作语义防错、语义检查

**Meta Router**:
在安全候选集中计算 VAE 似然与 PnL 记忆多因子加权得分，并结合单合约累计回撤熔断门槛决定最终调度的 Agent 索引与动作。
_Avoid_: 元路由器、路由选择器

**单合约熔断保护 (Circuit Breaker)**:
当单合约累计最大回撤率超过 15% 时强行切断 Agent 路由，全量降级为规则平仓 (`macro_action = 5`)。
_Avoid_: 熔断器、强平保护

### Evaluation And Diagnostics

**Aggregate CSV**:
低层测试后按 label-action-bin 聚合的验证结果 CSV，包含跨合约的 reward/turnover 统计。
_Avoid_: 聚合结果、汇总 CSV

**Detail CSV**:
可选的逐时间步交易动作明细，将行情、已执行仓位和账户损益记录在同一行中，是 Agent 形态分析的逐步事实来源。
_Avoid_: 明细 CSV、交易明细

**行为轨迹 (Action Trajectory)**:
单个 valid segment 在给定初始动作下，agent 逐 step 产生的执行前后仓位、标记价格、动作及已实现/浮动盈亏序列；是策略形态分类器的原料，由 (label, epoch, bin_index, contract, df_seq, initial_action) 唯一确定。
_Avoid_: 轨迹、trajectory（不明确时）、episode、rollout

**市场动态片段 (Market Dynamic Segment)**:
slice_model 按 slope 标签切换点切出的连续同质行情区间文件（df_0, df_1, ..., df_n），是 label 切分过程中产生的时间切片；只表示一阶趋势同质性（方向 + 强度档位），不表示二阶 K 线形态。segment 是 K 线数据的来源文件，不构成分类维度。
_Avoid_: 片段、segment 文件（不明确时）、label 切片

**K 线形态 (Kline Pattern)**:
行情侧的二阶形态分类，用价格与成交量描述突破、回调、加速、反转、箱体、背离或涨跌停状态；单个形态识别窗口只属于一种 K 线形态。
_Avoid_: K 线形状、行情形态（不明确时）、价格形态

**策略二阶形态 (Strategy Second-order Pattern)**:
Agent 侧的二阶形态分类，描述已执行仓位变化与行情事件的关系，而不是纯动作形状；单个形态识别窗口可同时属于多种策略二阶形态。
_Avoid_: 动作形态、策略类型（不明确时）、agent 类型

**Agent 形态候选全集 (Agent Pattern Candidate Universe)**:
存在逐步 Detail CSV 的全部 `(label, epoch, bin_index)` Agent triple 集合；当前已选 Agent 是该全集的标记子集，不是分类输入边界。
_Avoid_: 已选 Agent 集合、selection manifest 内容

**Detail 覆盖率 (Detail Coverage)**:
可用行为轨迹相对应评估 checkpoint 与已选 Agent 集合的完整程度。
_Avoid_: 候选全集大小、训练 epoch 完整性

**Initial-action 情景 (Initial-action Scenario)**:
同一 Agent triple 在相同行情上以某个初始动作启动的反事实回测情景；情景是否存在取决于是否完成了该行为轨迹，与它是否命中某个形态无关。不同 Initial-action 情景不是可相加的独立账户。
_Avoid_: 独立账户、独立行情样本、可累加回测

**Agent 形态明细表 (Agent Pattern Detail Table)**:
以形态识别窗口为粒度的明细数据，每个 window_id 恰好一行，读法为"哪个 agent 在哪个 label 的哪个 K 线形态下用哪些策略盈利如何"。K 线形态保存为单元素数组，策略形态保存为多选数组，窗口 `gross_pnl` / `net_pnl` 各只保存一次。
_Avoid_: agent 分类表、形态对照表

**Agent 形态展开表 (Agent Pattern Expanded Table)**:
从 Agent 形态明细表展开两个形态数组得到的分析数据，每个 (window_id, K 线形态, 策略形态) 组合恰好一行，用于单形态与 7×6 组合分析。
_Avoid_: 明细表（不明确数据粒度时）、扁平表

**形态识别窗口 (Pattern Recognition Window)**:
K 线形态、策略二阶形态与窗口盈亏共享的最小归因单元。
_Avoid_: 滑窗、识别窗口（不明确时）、N 窗口

**涨跌停事件窗口 (Limit-state Event Window)**:
label_0/label_6 中由整条行为轨迹构成的形态识别窗口；其 K 线形态固定为 KX1。
_Avoid_: 短窗口、涨跌停轨迹（不明确时）

**Execution Metrics**:
环境每步暴露的真实手续费、已实现利润和滑点指标，供测试明细和诊断使用。
_Avoid_: 执行指标、交易指标

**窗口毛盈亏 / 净盈亏 (Window Gross / Net PnL)**:
窗口毛盈亏 `gross_pnl` 等于窗口内已实现 PnL 总和加浮动 PnL 的窗口边界变化；窗口净盈亏 `net_pnl` 再扣除窗口内手续费，是形态绩效聚合的默认口径。滑点已体现在实际成交价值与已实现 PnL 中，不得重复扣除。
_Avoid_: 盈亏值（未说明手续费口径时）、净盈亏再扣滑点

**Trading Process Feature (交易过程特征)**:
Agent 侧动作执行后的实时交易状态输入，由归一化 signed position exposure、当前持仓的收益率/最大回撤率和当前持仓时长组成。
_Avoid_: 交易特征、过程特征、3 个比率特征

**Previous Action**:
现有低层 Q 网络输入名，实际表示当前 position/leverage 映射到 action space 的编码。
_Avoid_: 上一条交易命令、完整交易状态

**当前持仓 (Current Holding)**:
从 position 由 0 变为非 0 或持仓方向改变时开始，并在平仓到 0 或持仓方向再次改变时结束的一段方向性风险暴露；同方向加仓或减仓不结束当前持仓。
_Avoid_: 未平仓交易、连续非零仓位

**当前持仓时长 (Current Holding Duration)**:
当前持仓已持续的 env step 数；空仓为 0，开仓后的第一个可观测状态为 1，同方向持仓、加仓或减仓每经过一个 env step 继续累加，平仓归 0，反手后新方向从 1 开始。
_Avoid_: 全局 episode 时间、自然时间持仓时长、订单批次年龄

**current_holding_duration_norm**:
`trading_info` 中表示当前持仓时长的归一化字段，取值为 `min(current_holding_duration / holding_duration_norm_steps, 1.0)`。
_Avoid_: current_holding_duration、holding_time、holding_length

**持仓时长归一化窗口 (Holding Duration Normalization Window)**:
将当前持仓时长传递给模型前使用的 env step 尺度参数；归一化值为 `min(current_holding_duration / window, 1.0)`，默认窗口为 180 个 env step。
_Avoid_: 固定 180 分钟、交易日内进度、未截断持仓时长

**Experiment Name**:
Stage I 串行训练的实验名参数，用于隔离模型输出和日志目录。
_Avoid_: 实验名、实验标识

**Valid Multi-Contract Trial Selection (验证集多合约 Trial 遴选)**:
在高层路由或启发式策略分析中，对验证集中多个活跃合约 (`dataset/{freq}/{symbol}/valid/*.feather`) 的测试结果按等权重均值 (Per-contract Mean Indicator) 与组合整体收益率 (Portfolio Return) 进行跨合约综合计算，遴选最优 Trial，画图与诊断统一使用验证集多合约数据并按合约单独出图。
_Avoid_: 单合约遴选、test 集合遴选

### Trading Actions

**仓位档位 (Position Level)**:
由 max_holding_number 和 position_choices 启动参数按交易环境公式生成的完整有序 signed position 集合，负值为空头、0 为空仓、正值为多头；形态分类器显式消费同一集合。
_Avoid_: 固定五档、观测到的仓位集合、仓位数量（不明确是档位还是持仓量时）

**同向加仓 (Same-direction Position Increase)**:
执行前后仓位同号且在仓位档位集合中向更大绝对风险暴露移动的调仓；多头和空头按绝对仓位对称判定。从 0 到非 0 是开仓，仓位变号是反手，均不属于同向加仓。
_Avoid_: 仓位数值增加、开仓（不明确时）、加多仓

**盈利后同向加仓 (Profitable Same-direction Position Increase)**:
执行同向加仓前当前持仓的浮动盈亏大于 0；是 ST3 金字塔递增型的判定事件，多头与空头使用相同的绝对仓位规则。
_Avoid_: 盈利加仓（不明确持仓方向时）、累计已实现盈亏加仓

**Reverse Position (反手)**:
在一步内先平掉当前仓位再反向开仓的动作；持多时先平多后开空，持空时先平空后开多；采用 best-effort 语义，平仓一定成功，反向开仓可能因保证金不足或深度不足而失败。
_Avoid_: 翻仓、仓位翻转、flip position
