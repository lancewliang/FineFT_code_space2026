# FineFT

FineFT 是面向期货交易的高效且具有风险感知能力的集成强化学习研究代码库，实现论文"FineFT: Efficient and Risk-Aware Ensemble Reinforcement Learning for Futures Trading"的三阶段管线。

## Language

### 数据与预处理

**主力合约 (Main Contract)**:
按自然月成交量最高的前 2 个合约加上高成交量天数入选合约的并集，用于拼接连续主力数据。
_Avoid_: 主力连续、连续合约

**主力合约日文件 (Main Contract Daily File)**:
按 `TradingDay` 拆分的主力合约 CSV 文件，路径格式为 `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv`，由 `stitch_main_contract.py` 生成。
_Avoid_: 连续主力文件、日度连续数据

**主力合约 Summary (Main Contract Summary)**:
描述日期范围内入选合约集合、交易日窗口、源文件路径、每个合约最后交易日和完整交易日数量的 JSON 文件，由 `stitch_main_contract.py` 生成。
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
Scale Save 前后对 state feature 进行的 NaN 检查，发现 NaN 时立即报错并输出含 NaN 的特征名和行号，防止下游训练静默失败。
_Avoid_: NaN 检查、空值校验

### 特征工程

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
通过 IC、RankIC、CatBoost Importance、Permutation Importance、Sharpe 等指标评估和筛选 state feature 的流水线。
_Avoid_: 特征筛选、因子选择

**Feature Selection Manifest**:
Feature Selection 输出的 JSON 清单，记录候选特征来源、IC 结果路径、过滤配置、mandatory state feature 和最终特征列表，使用 `FeatureSelectionManifest` dataclass 表达。
_Avoid_: 特征选择清单、选择描述

**Feature Union**:
将多个合约的 candidate state feature 去重合并为品种级统一特征列表的过程。
_Avoid_: 特征合集、特征合并

**Feature Union Manifest**:
Feature Union 输出的 JSON 清单，记录品种、合约列表、各合约特征数、state feature 列表和输出路径，使用 `FeatureUnionManifest` dataclass 表达。
_Avoid_: 特征联合清单、联合描述

**Scale Save**:
使用 train-only robust scaler 对 state feature 进行标准化并裁剪到 `[-20, 20]`，输出下游可消费的 feather 和 csv 文件。
_Avoid_: 缩放保存、标准化输出

**Scale Manifest**:
Scale Save 输出的 JSON 清单，记录 scaler 版本、拟合范围（`fit_scope="train_all_contracts"`）、passthrough state feature、特征统计和裁剪配置，使用 `ScaleManifest` dataclass 表达。
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
LowerLimitPrice 和 UpperLimitPrice，属于 reward/execution 列，不进入 state candidate。
_Avoid_: 涨跌停、价格限制

### 数据集与切分

**Dataset Split**:
按时间边界将数据分为 train/valid/test 集合的过程，输出 `dataset_split_manifest.json`。
_Avoid_: 数据切分、数据分割

**Dataset Manifest**:
描述 FineFT 数据集的合约级输入输出、行数、state feature 路径和切片计划的 JSON 文件。
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

### 强化学习管线

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

### VAE 与路由

**VAE 跨合约训练 (VAE Cross-contract Training)**:
从多合约训练数据合并生成统一 VAE 训练集的过程，通过 `merge_vae_train.py` 物化 label 训练数据，校验特征维度一致性，输出 `LabelTrainingManifest`。
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

### 评估与诊断

**Aggregate CSV**:
低层测试后按 label-action-bin 聚合的验证结果 CSV，包含跨合约的 reward/turnover 统计。
_Avoid_: 聚合结果、汇总 CSV

**Detail CSV**:
可选的逐时间步交易动作明细 CSV，记录每步的仓位变化、手续费、滑点和账户价值。
_Avoid_: 明细 CSV、交易明细

**Execution Metrics**:
环境每步暴露的真实手续费、已实现利润和滑点指标，供测试明细和诊断使用。
_Avoid_: 执行指标、交易指标

**Trading Process Feature (交易过程特征)**:
Agent 侧动作执行后的实时交易状态输入，由归一化 signed position exposure 和当前持仓的收益率/最大回撤率组成。
_Avoid_: 交易特征、过程特征、3 个比率特征

**Previous Action**:
现有低层 Q 网络输入名，实际表示当前 position/leverage 映射到 action space 的编码。
_Avoid_: 上一条交易命令、完整交易状态

**当前持仓 (Current Holding)**:
从 position 由 0 变为非 0 或持仓方向改变时开始，并在平仓到 0 或持仓方向再次改变时结束的一笔交易；同方向加仓或减仓不结束当前持仓。
_Avoid_: 未平仓交易、连续非零仓位

**Experiment Name**:
Stage I 串行训练的实验名参数，用于隔离模型输出和日志目录。
_Avoid_: 实验名、实验标识

### 交易动作

**Reverse Position (反手)**:
在一步内先平掉当前仓位再反向开仓的动作；持多时先平多后开空，持空时先平空后开多；采用 best-effort 语义，平仓一定成功，反向开仓可能因保证金不足或深度不足而失败（position 归零或截断到 position_list 中最大可行值）；通过 `allow_reverse_position` 开关控制，默认关闭。
_Avoid_: 翻仓、仓位翻转、flip position

## Relationships

- 一个 **主力合约 Summary** 包含多个 **主力合约**，每个合约有 **交易日** 窗口和 **事件时间戳**
- **主力合约日文件** 是 **主力合约** 按 **交易日** 拆分的输出，供 **下采样** 消费
- **下采样** 使用 **右闭右标窗口** 从秒级快照生成目标频率 bar
- **截面特征** + **时间特征** → **Feature Selection** → **Feature Selection Manifest** → **State Feature**
- **Feature Union** 合并多合约 **State Feature** 为品种级列表，输出 **Feature Union Manifest**
- **State Feature** → **Scale Save** → **Scale Manifest**，前后执行 **NaN 校验**
- **Dataset Split** → **Dataset Manifest** → **Train Slice** + **Valid 动态切片**
- **Stage I** 训练低层 agent → **Stage II** 筛选 agent 并训练 VAE → **Stage III** 路由
- **Full-df Warmup** 使用 **Qtable 预计算** 生成 **DP Expert Action Path**
- **VAE 跨合约训练** 合并多合约数据 → **VAE Training Manifest** → **Label Summary** → **Routing Summary**
- **Aggregate CSV** 和 **Detail CSV** 使用 **Execution Metrics**
- **单边盘口** 和 **涨跌停价** 属于 **Reward/Execution 列**，不进入 **State Feature**
- **Trading Process Feature** 与 **State Feature** 互补，前者描述 agent 当前持仓暴露和风险收益状态，后者描述市场状态。
- **Trading Process Feature** 中的 single_holding_return_rate 和 single_holding_max_drawdown 属于 **当前持仓**，平仓或持仓方向改变都会结束当前这一笔交易。
- **Trading Process Feature** 表示动作执行后的可观测状态，空仓、正常平仓和爆仓后的风险收益字段均为 0。
- **Previous Action** 与 **Trading Process Feature** 互补：前者是当前 position/leverage 的 action-space 编码，后者保留归一化 signed exposure 和当前持仓风险收益。

## Example dialogue

> **Dev:** "商品期货的 mark_price 用什么计算？"
> **Domain expert:** "商品期货用 **参考价**，优先取 LastPrice，回退到 midprice。不使用真实的 mark price 或 index price。"

> **Dev:** "为什么 valid 阶段有的合约没有 feather 文件？"
> **Domain expert:** "那是 **Skipped Contract**，在 valid 集合没有命中交易日。Dataset Manifest 会记录跳过原因，Scale Save 不要求为缺失的 valid 文件生成输出。"

> **Dev:** "OFI 和 microstructure 特征有什么区别？"
> **Domain expert:** "**OFI** 从五档快照计算订单流不平衡，按固定行数窗口聚合。**Microstructure 特征** 从一档快照派生 microprice pressure、spread 变化和 queue pressure，也是独立固定行窗口。两者互不包含，也不改变现有时间窗口 quote 下采样输出。"

> **Dev:** "VAE 训练为什么要跨合约合并数据？"
> **Domain expert:** "商品期货单个合约的训练数据量可能不足，**VAE 跨合约训练** 将同一品种多个合约的训练数据合并为统一训练集，校验特征维度一致性后物化。输出 **VAE Training Manifest** 记录每个合约的样本数和缺失情况，确保可追溯。"

> **Dev:** "position exposure 能和 return_rate 一起叫 3 个比率特征吗？"
> **Domain expert:** "不能。position exposure 是由原始 position 归一化得到的 signed exposure，只有 single_holding_return_rate 和 single_holding_max_drawdown 是当前持仓的比率型风险收益特征；三者合称 **Trading Process Feature**。"

> **Dev:** "加仓后 single_holding_return_rate 要重新开始算吗？"
> **Domain expert:** "不会。同方向加仓或减仓仍然延续同一笔 **当前持仓**；只有平仓或持仓方向改变才会结束当前持仓。"

## Flagged ambiguities

- "主力合约" 在旧代码中可能指"月度成交量最高的合约"或"连续主力拼接后的数据"——已统一为"按月度 top 2 + 高成交量天数入选的合约集合"。
- "IC" 在 feature selection 中可能指 Pearson IC 或 Rank IC——已明确 IC 为 Pearson correlation，RankIC 为 rank correlation。
- "pretrain" 可能指 sample-level pretrain 或 full-df warmup——已明确 full-df warmup 是训练前的独立阶段，pretrain_epoch 默认为 0。
- "3 个比率特征" 会误把 position exposure 当作收益/风险比率——已统一为 **Trading Process Feature**：1 个归一化 signed position exposure + 2 个当前持仓比率型风险收益特征。
- "这笔交易" 的边界不清——已统一为 **当前持仓**，同方向变仓不结束，平仓或持仓方向改变时结束。
- "`previous_action`" 容易被误读为上一条交易命令——已明确为当前 position/leverage 的 action-space 编码。
