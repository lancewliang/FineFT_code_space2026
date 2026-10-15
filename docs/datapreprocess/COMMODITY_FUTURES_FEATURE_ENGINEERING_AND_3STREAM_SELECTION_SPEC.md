# 商品期货特征工程与三流特征过滤门禁体系全景说明文档
# Comprehensive Specification of Commodity Futures Feature Engineering and Triple-Stream Selection Gating Architecture

- **文件定位**: 生产级核心技术规范文档（Primary Technical Specification）
- **适用标的**: 国内商品期货（SHFE/DCE/CZCE/INE，以燃料油 fu 为基准）
- **核心源码依据**:
  - 特征下采样与盘口构建: `data_preprocess/operator_futures/commodity/downscale.py`, `downscale_continuous_by_trading_day.py`
  - 截面形态与微观结构: `data_preprocess/operator_futures/cross_section/create_feature.py`, `base_feature_util.py`
  - 日历与时间上下文: `data_preprocess/operator_futures/commodity/base_time_feature.py`
  - 跨期期限结构与合约轮动: `data_preprocess/operator_futures/commodity/cross_month_feature.py`
  - 混频跨日/跨周宏观特征: `data_preprocess/operator_futures/commodity/daily_base_feature.py`, `weekly_base_feature.py`, `daily_mixed_frequency_feature.py`, `weekly_mixed_frequency_feature.py`, `mixed_frequency_feature.py`
  - 时间滚动算子与技术特征: `data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py`, `multi_processing_util.py`, `time_operator_util.py`
  - 三流特征选择与统计门禁: `data_preprocess/operator_futures/feature_selection/muti_contract/` (`pipeline.py`, `types.py`, `distribution_audit.py`, `predictive_audit.py`, `stationarity_audit.py`, `regime_audit.py`, `nonlinear_scoring.py`, `orthogonal_dedup.py`, `data_hygiene.py`, `blacklists.py`)
  - 架构设计规约: `docs/specs/2026-10-04-triple-stream-feature-decoupling-architecture.md`, `docs/specs/2026-09-30-modular-three-stage-funnel-feature-selection-and-statistical-gates.md`, `docs/specs/2026-09-30-front-loaded-filtering-distribution-drift-gate-and-decentralized-multi-contract-feature-selection.md`

---

## 一、系统架构与特征生成全流程全景

FineFT 商品期货数据处理流水线将底层高频 L2 Tick / 1 秒级快照转化为面向强化学习交易策略网络与双轴变分自编码器（Hierarchical Dual-Gating VAE）的高质量、高信噪比特征矩阵。全流程分为九个连续流水线阶段：

```
+---------------------------------------------------------------------------------------------------+
|                                 1. 下采样阶段 (Downscaling)                                        |
|  原始数据 (Raw L2/1s Tick) -> 5档盘口 (Depth=5) + Trades/Quotes 统计 (10min Bar) + 涨跌停板特征    |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                             2. 截面特征提取 (Cross-Section Features)                              |
|  K线形态特征 (klen, kup, klow等) + 盘口微观结构 (WAP, Spread, 深度不平衡, Top-k增量)               |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                         3. 宏观与跨期扩展 (Multi-Resolution Context)                              |
|  3.1 日历与交易时间进度 (Base Time)                                                               |
|  3.2 跨期期限结构与轮动速度 (Cross-Month Term Structure: M1/M2/M3 基差、蝶式价差、持仓迁移速度)   |
|  3.3 混频跨日/跨周宏观特征 (Daily/Weekly Macro: 振幅、实体、VWAP偏离、分位数排名变换)               |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                        4. 单日合并与历史拼接 (Merge & Shift & Concat)                              |
|  区分 Concurrent 特征 (执行环境列) 与 Future 特征 (状态特征，右移 shift(1) 防止未来信息泄露)      |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                             5. 时间序列滚动算子 (Time Rolling Operators)                           |
|  多周期价格/成交量收益率、趋势、波动率、高阶形态 (W in [1, 2, 6, 12, 16, 24, 48, 96, 192])        |
|  物理风险与流动性 (Parkinson, Garman-Klass, Realized Vol, ATR%) + 增强技术指标 (MACD, RSI, ADX)    |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                       6. 清洗与数据集切分 (Merge Clean & Dataset Split)                            |
|  按合约独立切分 Train (主力活跃期前段)、Valid (中段)、Test (尾段与交割前置期)                     |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                     7. 三流解耦特征选择体系 (Triple-Stream Feature Selection)                      |
|  输入: 566 状态候选特征 + 17 强制状态特征 (跨 14 个主力/次主力合约)                               |
|  - 分支 A (Slope VAE): 过滤有符号方向动量，剔除波动率噪声，单调性 Down/Flat/Up (保留 15 特征)     |
|  - 分支 B (Vol VAE): 过滤绝对收益离散度，剔除方向性偏置，单调性 Low/Mid/High Vol (保留 12 特征)    |
|  - 分支 C (RL Agent): 保留全维微观失衡、跨期结构与宏观动量，聚类去重 + 挽救条件锚点 (保留 148 特征)|
|  - 数学并集: S_union = S_slope U S_vol U S_rl (全集 165 特征)                                    |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                   8. 多合约自适应稳健缩放保存 (Scale Describe & Robust Scale)                     |
|  Robust Z-Score (Median / IQR 缩放) + Hard Clip [-20, 20]，强制特征直通，输出最终训练张量          |
+---------------------------------------------------------------------------------------------------+
```

---

## 二、特征语义、数学公式与时间窗口全景目录

本节系统罗列特征工程流水线中生成的全部特征类别、详细语义、精确数学公式与涉及的时间窗口。

当前预处理生成的目标数据集单合约数据（如 `PREPROCESS_DATASET/.../10min/fu/train/fu2311.feather`）总计包含 **1,324 列**：
- **元数据列 (2 列)**：`timestamp`（时间戳索引）、`contract`（合约代码标识符）。
- **合并伴生重复列 (245 列)**：以 `_right` 为后缀的特征（如 `ask1_price_right`、`prev_day_return_right`），系跨日与跨周期多表对齐合并时产生的伴生重复列。依据 `data_preprocess/operator_futures/feature_audit/eligibility.py` 的架构规范，该类列被前置硬性排除规则（`EXCLUDED_SUFFIXES = ("_right", "_y", "_duplicate")`）安全过滤，不参与特征门禁与模型训练。
- **有效特征候选集 (1,077 列)**：由以下 10 大核心数学与物理特征家族完整构成，本全景目录对其逐一核对、完备归纳并完成全量技术更新。

---

### “是否相对化与无量纲化”判定准则与“分布漂移问题”形式化定义

在商品期货多周期建模中，不同年份的价格绝对水平（例如燃料油在 2022~2023 年高位达 3500~4000 元/吨，而在下行周期跌破 2500 元/吨）与成交量绝对规模（活跃期单日成交上百万手 vs 沉寂期数万手）存在巨大断层。为了从源头上预防模型泛化失效，本规范为所有特征显式标注**“是否相对化与无量纲化”**并对其**“分布漂移风险”**进行形式化定义：

1. **物理量纲与绝对水平问题（Dimensionality & Level Variables）**：
   - 特征若以货币金额（元）、绝对报价（元/吨）、物理成交手数（手）或未除以价格基准的绝对价差点数（Ticks）为度量单位，则称为**有量纲变量**。
   - 这类变量在时间序列上天生是非平稳的（Non-Stationary），其期望 $\mathbb{E}[X_t]$ 和方差 $\text{Var}[X_t]$ 会随宏观周期和市场整体体量产生长期时变漂移。

2. **分布漂移的数学定义与危害（Distribution Shift / Covariate Shift）**：
   - 设历史训练期为 $D_{\text{train}}$，未来验证/测试或实盘期为 $D_{\text{test}}$。
   - **协变量漂移 (Covariate Shift)**：$P_{\text{train}}(X) \ne P_{\text{test}}(X)$。当测试期遇到训练集从未出现过的特征绝对数值区间时，神经网络激活函数进入未拟合或饱和区域，决策树分裂阈值越界失效。
   - **概念漂移与伪相关 (Concept Drift & Spurious Correlation)**：$P_{\text{train}}(Y \mid X) \ne P_{\text{test}}(Y \mid X)$。在全样本统计中，有量纲特征会因为偶然契合了样本区间的单边宏观走势而产生虚假的高预测相关性（例如绝对价格越低未来收益越高，本质只是样本期恰处于长期下跌周期），但在样本外完全崩溃。

3. **相对化与无量纲化评级标准**：
   - **`是 (无漂移风险)`**：特征通过对数收益率、价格相对百分比、截面份额比例、时间小节进度、或局部滚动 Z-Score/分位数排名等数学变换彻底消除了物理量纲。跨年份价格中枢与市场总成交量改变不影响其数值分布与经济学解释。
   - **`否 (高漂移风险)`**：特征保留原始绝对价格、绝对成交手数、绝对成交金额或未除以价格基准的绝对价差。此类特征禁止直接进入预测网络与 RL State 向量，必须在特征提取层被剔除或完成差分/无量纲化转换。
   - **`部分/条件 (中漂移风险)`**：公式进行了比率化或局部平滑，但分子分母中仍直接混杂绝对金额（如 tradeval）或绝对手数（如 volume），跨周期仍受全市场流动性总规模漂移的影响，必须依赖严格的多合约跨期分布门禁（PSI $\le 0.10$）与因果自适应缩放（Causal Rolling Scaler）。

---

### 统一数学记号

- 目标周期时间桶记为 $I_t = (T_{t-1}, T_t]$（默认 10 分钟，左开右闭）。
- 基础价格与挂单记号：
  - $P_i, V_i, S_i$：时间桶内的逐笔成交价格、成交量、买卖方向（$S_i \in \{+1, -1\}$）。
  - $AP_l, AS_l$：第 $l$ 档卖价与卖方挂单量（$l \in \{1, \dots, 5\}$）。
  - $BP_l, BS_l$：第 $l$ 档买价与买方挂单量（$l \in \{1, \dots, 5\}$）。
  - $P_{\text{limit\_up}}, P_{\text{limit\_down}}$：当日涨停板价与跌停板价。
- 防零与防数值溢出常数：
  - $\epsilon_1 = 10^{-15}$（微观截面防零）。
  - $\epsilon_2 = 10^{-12}$（时序滚动防零）。
  - $\epsilon_3 = 10^{-8}$（Z-Score 标准差防零）。
- 滚动时序算子（以当前 Bar 终点 $t$ 为基准，窗口长度 $w$）：
  - $\text{Mean}_w(X)_t = \frac{1}{w} \sum_{k=0}^{w-1} X_{t-k}$
  - $\text{Std}_w(X)_t = \sqrt{\frac{1}{w-1} \sum_{k=0}^{w-1} (X_{t-k} - \text{Mean}_w(X)_t)^2}$
  - $\Delta_k X_t = X_t - X_{t-k}$

---

### 1. 基础成交与盘口下采样特征 (Downscale Base Trades & Quotes Features)

源码：`data_preprocess/operator_futures/commodity/downscale.py`，数据集实际包含 **39 列**：

| # | 特征名称/模板 | 语义解释 | 精确数学公式 | 参数与时间窗口 | 是否相对化与无量纲化 |
|---|-------------|---------|-------------|--------------|---------------------|
| 1 | `open` | 桶内首笔成交价 | $P_{\text{first} \in I_t}$ | 单 Bar (10m) | **否 (高漂移风险)**：原始绝对价格，随年份价格中枢大幅漂移 |
| 2 | `high` | 桶内最高成交价 | $\max_{i \in I_t} P_i$ | 单 Bar (10m) | **否 (高漂移风险)**：原始绝对价格，随年份价格中枢大幅漂移 |
| 3 | `low` | 桶内最低成交价 | $\min_{i \in I_t} P_i$ | 单 Bar (10m) | **否 (高漂移风险)**：原始绝对价格，随年份价格中枢大幅漂移 |
| 4 | `close` | 桶内末笔成交价 | $P_{\text{last} \in I_t}$ | 单 Bar (10m) | **否 (高漂移风险)**：原始绝对价格，随年份价格中枢大幅漂移 |
| 5 | `volume` | 桶内累计成交量 | $\sum_{i \in I_t} V_i$ | 单 Bar (10m) | **否 (高漂移风险)**：绝对成交手数，受跨期市场活跃度与规模漂移影响 |
| 6 | `tradeval` | 桶内累计成交金额 | $\sum_{i \in I_t} P_i \cdot V_i \cdot \text{Unit}$ | 单 Bar (10m) | **否 (高漂移风险)**：绝对成交金额，受价格与成交量双重绝对漂移影响 |
| 7 | `open_interest` | 期末持仓总量 | $\text{Last}(OI_i \in I_t)$ | 单 Bar (10m) | **否 (高漂移风险)**：绝对持仓手数，随合约生命周期与年份规模剧烈漂移 |
| 8 | `index_price` | 指数标的基准价格 | $P_{\text{index}, t}$ | 单 Bar 快照 | **否 (高漂移风险)**：现货/综合指数绝对水平，随宏观周期大幅漂移 |
| 9 | `mark_price` | 衍生品标记价格 | $P_{\text{mark}, t}$ | 单 Bar 快照 | **否 (高漂移风险)**：绝对标记价格，具有货币量纲 |
| 10 | `funding_rate` | 综合资金费率/基差贴水率 | $F_t = \frac{P_{\text{futures}} - P_{\text{spot}}}{P_{\text{spot}}}$ | 单 Bar 快照 | **是 (无漂移风险)**：无量纲相对基差/费率比例，消除了绝对价格量纲 |
| 11 | `funding_timestamp` | 资金费率更新时间戳 | $T_{\text{funding}}$ | 毫秒时间戳 | **否 (高漂移风险)**：绝对单调递增时间戳，仅作为对齐索引，不可直接作为特征 |
| 12 | `ntrade_estimated` | 估计成交笔数 | 桶内有成交秒数或有效价格跳动次数 | 单 Bar (10m) | **否 (高漂移风险)**：绝对笔数计数，受跨年行情推送频率与活跃度影响 |
| 13 | `ntrade_up_estimated` | 主买驱动估计笔数 | $\sum_{i \in I_t} \mathbb{I}\{\Delta P_i > 0\}$ | 单 Bar (10m) | **否 (高漂移风险)**：绝对笔数计数，未按总成交笔数做无量纲归一化 |
| 14 | `ntrade_down_estimated` | 主卖驱动估计笔数 | $\sum_{i \in I_t} \mathbb{I}\{\Delta P_i < 0\}$ | 单 Bar (10m) | **否 (高漂移风险)**：绝对笔数计数，未按总成交笔数做无量纲归一化 |
| 15 | `ntrade_flat_estimated` | 平价变动估计笔数 | $\sum_{i \in I_t} \mathbb{I}\{\Delta P_i = 0\}$ | 单 Bar (10m) | **否 (高漂移风险)**：绝对笔数计数，未按总成交笔数做无量纲归一化 |
| 16 | `ntrade_estimated_up_udnorm` | 估计买方成交笔数归一化 | $\frac{\text{ntrade\_up\_estimated}}{\text{ntrade\_estimated} + \epsilon_1}$ | 单 Bar (10m) | **是 (无漂移风险)**：主买笔数占总笔数百分比 $[0.0, 1.0]$，消除绝对笔数漂移 |
| 17 | `ntrade_estimated_down_udnorm` | 估计卖方成交笔数归一化 | $\frac{\text{ntrade\_down\_estimated}}{\text{ntrade\_estimated} + \epsilon_1}$ | 单 Bar (10m) | **是 (无漂移风险)**：主卖笔数占总笔数百分比 $[0.0, 1.0]$，消除绝对笔数漂移 |
| 18 | `ntrade_estimated_flat_udnorm` | 估计平价成交笔数归一化 | $\frac{\text{ntrade\_flat\_estimated}}{\text{ntrade\_estimated} + \epsilon_1}$ | 单 Bar (10m) | **是 (无漂移风险)**：平价笔数占总笔数百分比 $[0.0, 1.0]$，消除绝对笔数漂移 |
| 19 | `ntrade_estimated_updownflat_vol_udnorm` | 每手成交笔数密度归一化 | $\frac{\text{ntrade\_estimated}}{\text{volume} + \epsilon_1}$ | 单 Bar (10m) | **是 (无漂移风险)**：单笔成交平均手数的倒数，纯强度比率，消除绝对规模漂移 |
| 20-24 | `bid1_price` ~ `bid5_price` | 1~5 档买价 | $BP_l, \quad l \in \{1, \dots, 5\}$ | 快照 (Bar 末) | **否 (高漂移风险)**：原始买方挂单绝对价格，随年份大幅漂移 |
| 25-29 | `bid1_size` ~ `bid5_size` | 1~5 档买量 | $BS_l, \quad l \in \{1, \dots, 5\}$ | 快照 (Bar 末) | **否 (高漂移风险)**：绝对挂单手数，受年份做市深度与流动性规模影响 |
| 30-34 | `ask1_price` ~ `ask5_price` | 1~5 档卖价 | $AP_l, \quad l \in \{1, \dots, 5\}$ | 快照 (Bar 末) | **否 (高漂移风险)**：原始卖方挂单绝对价格，随年份大幅漂移 |
| 35-39 | `ask1_size` ~ `ask5_size` | 1~5 档卖量 | $AS_l, \quad l \in \{1, \dots, 5\}$ | 快照 (Bar 末) | **否 (高漂移风险)**：绝对挂单手数，受年份做市深度与流动性规模影响 |

> **注**：基础均价 `vwap`、`twap`、`awap` 详见第 3 节与第 4 节，其在订单簿微观结构与 K 线几何形态中被全面相对化并派生高级特征。

---

### 2. 涨跌停板与非对称深度限制特征 (Price Limit & Asymmetric Liquidity)

源码：`data_preprocess/operator_futures/commodity/downscale.py`，数据集实际包含 **9 列**：

| # | 特征名称/模板 | 语义解释 | 精确数学公式 | 参数与时间窗口 | 是否相对化与无量纲化 |
|---|-------------|---------|-------------|--------------|---------------------|
| 1 | `UpperLimitPrice` | 交易所涨停板绝对价格 | $P_{\text{limit\_up}}$ | 交易日静态快照 | **否 (高漂移风险)**：交易所公布的每日静态价格绝对上限 |
| 2 | `LowerLimitPrice` | 交易所跌停板绝对价格 | $P_{\text{limit\_down}}$ | 交易日静态快照 | **否 (高漂移风险)**：交易所公布的每日静态价格绝对下限 |
| 3 | `ask_side_empty` | 卖一盘口无挂单标志 (涨停空盘) | $\mathbb{I}\{AS_1 == 0 \lor AP_1 \text{ 为 NaN}\}$ | 单 Bar 快照 | **是 (无漂移风险)**：布尔/离散 0/1 标志，表征涨停封死无卖单状态 |
| 4 | `bid_side_empty` | 买一盘口无挂单标志 (跌停空盘) | $\mathbb{I}\{BS_1 == 0 \lor BP_1 \text{ 为 NaN}\}$ | 单 Bar 快照 | **是 (无漂移风险)**：布尔/离散 0/1 标志，表征跌停封死无买单状态 |
| 5 | `limit_up_single_sided_ratio` | 涨停单边挂单比例 | $\frac{1}{T} \sum_{t=1}^T \mathbb{I}\{AP_1 \text{ 为空} \land P_t = P_{\text{limit\_up}}\}$ | 单 Bar (10m) 统计 | **是 (无漂移风险)**：时间占比 $[0.0, 1.0]$，无物理量纲 |
| 6 | `limit_down_single_sided_ratio` | 跌停单边挂单比例 | $\frac{1}{T} \sum_{t=1}^T \mathbb{I}\{BP_1 \text{ 为空} \land P_t = P_{\text{limit\_down}}\}$ | 单 Bar (10m) 统计 | **是 (无漂移风险)**：时间占比 $[0.0, 1.0]$，无物理量纲 |
| 7 | `limit_up_ask_depth_ratio_5` | 卖盘深度触及涨停比 | $\frac{1}{5} \sum_{l=1}^5 \mathbb{I}\{AP_l \ge P_{\text{limit\_up}} \lor AP_l \text{ 为空}\}$ | 5 档快照 (Bar 末) | **是 (无漂移风险)**：档位深度比例 $[0.0, 1.0]$，无物理量纲 |
| 8 | `limit_down_bid_depth_ratio_5` | 买盘深度触及跌停比 | $\frac{1}{5} \sum_{l=1}^5 \mathbb{I}\{BP_l \le P_{\text{limit\_down}} \lor BP_l \text{ 为空}\}$ | 5 档快照 (Bar 末) | **是 (无漂移风险)**：档位深度比例 $[0.0, 1.0]$，无物理量纲 |
| 9 | `limit_depth_imbalance_ratio_5` | 涨跌停深度不平衡比 | $\text{limit\_up\_ask\_depth\_ratio\_5} - \text{limit\_down\_bid\_depth\_ratio\_5}$ | 5 档快照 (Bar 末) | **是 (无漂移风险)**：深度差值 $[-1.0, 1.0]$，无物理量纲 |

---

### 3. 截面微观结构与订单簿不平衡特征 (Orderbook Microstructure & Imbalance)

源码：`data_preprocess/operator_futures/cross_section/base_feature_util.py`，数据集实际包含 **60 列**：

| # | 特征名称/模板 | 语义解释 | 精确数学公式 | 参数与时间窗口 | 是否相对化与无量纲化 |
|---|-------------|---------|-------------|--------------|---------------------|
| 1 | `midprice` | 盘口中间价 | $\frac{AP_1 + BP_1}{2}$ | 瞬时截面 | **否 (高漂移风险)**：绝对价格中枢，随年份宏观周期大幅漂移 |
| 2 | `vwap` | 桶内成交量加权平均价 | $\frac{\sum_{i} P_i V_i}{\sum_i V_i}$（若 volume=0 回退为 close） | 单 Bar (10m) | **否 (高漂移风险)**：绝对价格，具有货币物理量纲 |
| 3 | `twap` | 桶内时间算术加权平均价 | $\frac{1}{N} \sum_{i=1}^N P_i$ | 单 Bar (10m) | **否 (高漂移风险)**：绝对价格，具有货币物理量纲 |
| 4 | `awap` | 桶内算术平均价 (快照均值) | $\frac{\text{open} + \text{high} + \text{low} + \text{close}}{4}$ | 单 Bar (10m) | **否 (高漂移风险)**：绝对价格，具有货币物理量纲 |
| 5 | `wap_1` | 1 档订单簿加权均价 | $\frac{AS_1 \cdot BP_1 + BS_1 \cdot AP_1}{AS_1 + BS_1}$ | 瞬时截面 | **否 (高漂移风险)**：加权绝对价格，具有价格物理量纲 |
| 6 | `wap_2` | 2 档订单簿加权均价 | $\frac{AS_2 \cdot BP_2 + BS_2 \cdot AP_2}{AS_2 + BS_2}$ | 瞬时截面 | **否 (高漂移风险)**：加权绝对价格，具有价格物理量纲 |
| 7 | `wap_balance` | 1、2 档 WAP 价差 | $\text{wap\_1} - \text{wap\_2}$ | 瞬时截面 | **部分 (中漂移风险)**：绝对点差点数，高价格时期绝对价差点数自然放大 |
| 8 | `sell_wap` | 卖方全档位加权平均价 | $\sum_{l=1}^5 \frac{AS_l}{\sum AS} \cdot AP_l$ | 5 档截面 | **否 (高漂移风险)**：绝对报价，具有货币物理量纲 |
| 9 | `buy_wap` | 买方全档位加权平均价 | $\sum_{l=1}^5 \frac{BS_l}{\sum BS} \cdot BP_l$ | 5 档截面 | **否 (高漂移风险)**：绝对报价，具有货币物理量纲 |
| 10 | `buy_sell_wap_spread` | 深度加权买卖价差 | $\text{sell\_wap} - \text{buy\_wap}$ | 5 档截面 | **部分 (中漂移风险)**：全档深度绝对价差，高油价时期绝对价差增大 |
| 11 | `buy_spread_oe_max` | 买一相对最优买盘价差 | $\max(BP_1 - BP_2, \dots)$ | 瞬时截面 | **部分 (中漂移风险)**：微观买盘价差阶跃点数，受价格尺度影响 |
| 12 | `sell_spread_oe_max` | 卖一相对最优卖盘价差 | $\max(AP_2 - AP_1, \dots)$ | 瞬时截面 | **部分 (中漂移风险)**：微观卖盘价差阶跃点数，受价格尺度影响 |
| 13 | `buy_volume_oe` | 买方驱动订单执行量 | $\sum_{i \in I_t} V_i \mathbb{I}\{\Delta P_i > 0\}$ | 单 Bar (10m) | **否 (高漂移风险)**：绝对手数，受年份做市活跃度漂移影响 |
| 14 | `sell_volume_oe` | 卖方驱动订单执行量 | $\sum_{i \in I_t} V_i \mathbb{I}\{\Delta P_i < 0\}$ | 单 Bar (10m) | **否 (高漂移风险)**：绝对手数，受年份做市活跃度漂移影响 |
| 15 | `imblance_volume_oe` | 买卖驱动执行量失衡 | $\text{buy\_volume\_oe} - \text{sell\_volume\_oe}$ | 单 Bar (10m) | **否 (高漂移风险)**：绝对买卖量净差额（在时序算子中被转化为 trend 与 log_return） |
| 16-20 | `ask1_size_n` ~ `ask5_size_n` | 卖方各档挂单归一化占比 | $\frac{AS_l}{\sum_{j=1}^5 AS_j + \epsilon_1}, \quad l \in \{1, \dots, 5\}$ | 5 档截面 | **是 (无漂移风险)**：各档量占前5档挂单总量比例 $[0.0, 1.0]$，纯无量纲比例 |
| 21-25 | `bid1_size_n` ~ `bid5_size_n` | 买方各档挂单归一化占比 | $\frac{BS_l}{\sum_{j=1}^5 BS_j + \epsilon_1}, \quad l \in \{1, \dots, 5\}$ | 5 档截面 | **是 (无漂移风险)**：各档量占前5档挂单总量比例 $[0.0, 1.0]$，纯无量纲比例 |
| 26-30 | `ask_price_topk_size_{k}_increments` | 卖盘价格累计递增跨度 | $AP_k - AP_1, \quad k \in \{1, \dots, 5\}$ | 5 档截面 | **部分 (中漂移风险)**：绝对价差阶跃点数，随基础报价大小轻微漂移 |
| 31-35 | `bid_price_topk_size_{k}_increments` | 买盘价格累计递减跨度 | $BP_1 - BP_k, \quad k \in \{1, \dots, 5\}$ | 5 档截面 | **部分 (中漂移风险)**：绝对价差阶跃点数，随基础报价大小轻微漂移 |
| 36-40 | `ask_size_topk_size_{k}_increments` | 卖盘前 $k$ 档累计挂单量增量 | $\sum_{l=1}^k AS_l - AS_1, \quad k \in \{1, \dots, 5\}$ | 5 档截面 | **部分 (中漂移风险)**：绝对手数增量，受市场总挂单量级影响 |
| 41-45 | `bid_size_topk_size_{k}_increments` | 买盘前 $k$ 档累计挂单量增量 | $\sum_{l=1}^k BS_l - BS_1, \quad k \in \{1, \dots, 5\}$ | 5 档截面 | **部分 (中漂移风险)**：绝对手数增量，受市场总挂单量级影响 |
| 46-50 | `ask_size_topk_size_{k}_share` | 卖盘前 $k$ 档挂单深度占比 | $\frac{\sum_{l=1}^k AS_l}{\sum_{l=1}^5 AS_l + \epsilon_1}, \quad k \in \{1, \dots, 5\}$ | 5 档截面 | **是 (无漂移风险)**：前 $k$ 档挂单量相对前 5 档总量的占比 $[0.0, 1.0]$ |
| 51-55 | `bid_size_topk_size_{k}_share` | 买盘前 $k$ 档挂单深度占比 | $\frac{\sum_{l=1}^k BS_l}{\sum_{l=1}^5 BS_l + \epsilon_1}, \quad k \in \{1, \dots, 5\}$ | 5 档截面 | **是 (无漂移风险)**：前 $k$ 档挂单量相对前 5 档总量的占比 $[0.0, 1.0]$ |
| 56 | `ntrade_estimated_updown_imbalance_udnorm` | 估计买卖笔数归一化净失衡 | $\frac{N_{\text{up}} - N_{\text{down}}}{N_{\text{up}} + N_{\text{down}} + \epsilon_1}$ | 单 Bar (10m) | **是 (无漂移风险)**：笔数归一化净失衡率 $[-1.0, 1.0]$，彻底消除绝对笔数 |
| 57-58 | `price_to_vwap_zscore_{w}` | 现价相对 VWAP 偏离 Z-Score | $\frac{\text{Dev}_t - \text{Mean}_w(\text{Dev})}{\text{Std}_w(\text{Dev}) + \epsilon_3}, \quad \text{Dev} = \frac{C - \text{VWAP}}{C}$ | $w \in \{48, 96\}$ | **是 (无漂移风险)**：相对价格偏离百分比经局部滚动 Z-Score 平稳化 |
| 59-60 | `macro_trade_imbalance_continuous_{w}` | 宏观平滑主动买卖失衡率 | $\frac{\sum_w N_{\text{up}} - \sum_w N_{\text{down}}}{\sum_w N_{\text{up}} + \sum_w N_{\text{down}} + 10w}$ | $w \in \{48, 240\}$ | **是 (无漂移风险)**：宏观主动买卖笔数归一化失衡率 $[-1.0, 1.0]$ |

---

### 4. 截面 K 线形态特征 (Cross-Section K-Line Morphology)

源码：`data_preprocess/operator_futures/cross_section/base_feature_util.py`，数据集实际包含 **21 列**：

| # | 特征名称/模板 | 语义解释 | 精确数学公式 | 参数与时间窗口 | 是否相对化与无量纲化 |
|---|-------------|---------|-------------|--------------|---------------------|
| 1 | `klen` | K 线全振幅比率 | $\frac{H - L}{O + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：振幅除以开盘价，纯相对百分比收益率 |
| 2 | `kmid` | 实体相对开盘价比例 | $\frac{\vert C - O \vert}{O + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：K 线绝对实体高度百分比 |
| 3 | `kmid2` | 实体相对振幅归一化 | $\frac{C - O}{H - L + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：有符号实体占比 $[-1.0, 1.0]$，无物理量纲 |
| 4 | `kup` | 上影线相对开盘价比例 | $\frac{H - \max(O, C)}{O + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：上影线高度百分比 |
| 5 | `kup2` | 上影线相对振幅归一化 | $\frac{H - \max(O, C)}{H - L + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：上影线占总振幅比率 $[0.0, 1.0]$ |
| 6 | `klow` | 下影线相对开盘价比例 | $\frac{\min(O, C) - L}{O + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：下影线高度百分比 |
| 7 | `klow2` | 下影线相对振幅归一化 | $\frac{\min(O, C) - L}{H - L + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：下影线占总振幅比率 $[0.0, 1.0]$ |
| 8 | `ksft` | 收盘价偏移中心 (开盘归一化) | $\frac{2C - H - L}{O + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：收盘价在最高最低区间内的偏置度，除以开盘价 |
| 9 | `ksft2` | 收盘价偏移中心 (振幅归一化) | $\frac{2C - H - L}{H - L + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：收盘价在最高最低区间内的相对位置 $[-1.0, 1.0]$ |
| 10 | `kotwap` | 开盘价偏离 TWAP (开盘归一化) | $\frac{O - \text{TWAP}}{O + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：开盘价相对桶内时间均价偏离率 |
| 11 | `kotwap2` | 开盘价偏离 TWAP (振幅归一化) | $\frac{O - \text{TWAP}}{H - L + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：开盘偏离占 Bar 振幅的比率 |
| 12 | `kctwap` | 收盘价偏离 TWAP (开盘归一化) | $\frac{C - \text{TWAP}}{O + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：收盘价相对桶内时间均价偏离率 |
| 13 | `kctwap2` | 收盘价偏离 TWAP (振幅归一化) | $\frac{C - \text{TWAP}}{H - L + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：收盘偏离占 Bar 振幅的比率 |
| 14 | `koawap` | 开盘价偏离 AWAP (开盘归一化) | $\frac{O - \text{AWAP}}{O + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：开盘价相对算术均价偏离率 |
| 15 | `koawap2` | 开盘价偏离 AWAP (振幅归一化) | $\frac{O - \text{AWAP}}{H - L + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：开盘偏离占 Bar 振幅的比率 |
| 16 | `kcawap` | 收盘价偏离 AWAP (开盘归一化) | $\frac{C - \text{AWAP}}{O + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：收盘价相对算术均价偏离率 |
| 17 | `kcawap2` | 收盘价偏离 AWAP (振幅归一化) | $\frac{C - \text{AWAP}}{H - L + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：收盘偏离占 Bar 振幅的比率 |
| 18 | `kovwap` | 开盘价偏离 VWAP (开盘归一化) | $\frac{O - \text{VWAP}}{O + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：开盘价相对量加权均价偏离率 |
| 19 | `kovwap2` | 开盘价偏离 VWAP (振幅归一化) | $\frac{O - \text{VWAP}}{H - L + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：开盘偏离占 Bar 振幅的比率 |
| 20 | `kcvwap` | 收盘价偏离 VWAP (开盘归一化) | $\frac{C - \text{VWAP}}{O + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：收盘价相对量加权均价偏离率 |
| 21 | `kcvwap2` | 收盘价偏离 VWAP (振幅归一化) | $\frac{C - \text{VWAP}}{H - L + \epsilon_1}$ | 单 Bar 截面 | **是 (无漂移风险)**：收盘偏离占 Bar 振幅的比率 |

> **说明**：初期技术预研中规划的 `kskew`、`kamt`、`kvld` 等有量纲形态特征，已在工程实现中被上述通过开盘价与极值振幅严格归一化的 21 个纯无量纲微观几何特征全面替代，彻底消除了绝对价格中枢对 K 线形态的干扰。

---

### 5. 日历与交易时间上下文特征 (Base Time Features)

源码：`data_preprocess/operator_futures/calendar/base_time.py`，数据集实际包含 **11 列**（另有 1 列 `prev_day_contract_role_tier` 归入混频特征）：

| # | 特征名称/模板 | 语义解释 | 精确数学公式 | 参数与时间窗口 | 是否相对化与无量纲化 |
|---|-------------|---------|-------------|--------------|---------------------|
| 1 | `trading_minute_progress` | 交易小节内分钟进度 | $\frac{t - T_{\text{session\_start}}}{T_{\text{session\_end}} - T_{\text{session\_start}}}$ | 当前交易小节 | **是 (无漂移风险)**：日内时间流进度 $[0.0, 1.0]$，消除日历绝对时间漂移 |
| 2 | `morning_session` | 早盘交易时段虚拟变量 | $\mathbb{I}\{\text{Session} == \text{Morning}\}$ | 09:00~11:30 | **是 (无漂移风险)**：二值 One-Hot 编码 $\{0, 1\}$ |
| 3 | `afternoon_session` | 午盘交易时段虚拟变量 | $\mathbb{I}\{\text{Session} == \text{Afternoon}\}$ | 13:30~15:00 | **是 (无漂移风险)**：二值 One-Hot 编码 $\{0, 1\}$ |
| 4 | `night_session` | 夜盘交易时段虚拟变量 | $\mathbb{I}\{\text{Session} == \text{Night}\}$ | 21:00~23:00/02:30 | **是 (无漂移风险)**：二值 One-Hot 编码 $\{0, 1\}$ |
| 5 | `is_opening_30m` | 开盘前 30 分钟强冲击标志 | $\mathbb{I}\{t - T_{\text{open}} \le 30\text{min}\}$ | 开盘前 3 个 Bar | **是 (无漂移风险)**：二值状态标志 $\{0, 1\}$ |
| 6 | `is_closing_30m` | 收盘前 30 分钟平仓潮标志 | $\mathbb{I}\{T_{\text{close}} - t \le 30\text{min}\}$ | 尾盘后 3 个 Bar | **是 (无漂移风险)**：二值状态标志 $\{0, 1\}$ |
| 7 | `is_session_first_bar` | 交易小节首个 Bar 标志 | $\mathbb{I}\{t == T_{\text{session\_first}}\}$ | 单 Bar 瞬时 | **是 (无漂移风险)**：二值脉冲标志 $\{0, 1\}$，识别集合竞价后开盘效应 |
| 8 | `is_session_last_bar` | 交易小节末尾 Bar 标志 | $\mathbb{I}\{t == T_{\text{session\_last}}\}$ | 单 Bar 瞬时 | **是 (无漂移风险)**：二值脉冲标志 $\{0, 1\}$，识别小节收盘效应 |
| 9 | `contract_month_sin` | 交割合约月份周期正弦编码 | $\sin\left(\frac{2\pi \cdot M}{12}\right), \quad M \in \{1, \dots, 12\}$ | 静态属性 | **是 (无漂移风险)**：连续周期编码 $[-1.0, 1.0]$，消除季节月份阶跃非连续性 |
| 10 | `contract_month_cos` | 交割合约月份周期余弦编码 | $\cos\left(\frac{2\pi \cdot M}{12}\right), \quad M \in \{1, \dots, 12\}$ | 静态属性 | **是 (无漂移风险)**：连续周期编码 $[-1.0, 1.0]$，消除季节月份阶跃非连续性 |
| 11 | `contract_life_remaining_ratio` | 合约剩余生命周期比例 | $\frac{T_{\text{expire}} - t}{T_{\text{expire}} - T_{\text{listed}}}$ | 动态计算 | **是 (无漂移风险)**：生命周期进度 $[0.0, 1.0]$，跨所有年份合约具备相同物理尺度 |
| 12 | `prev_day_contract_role_tier` | 前一日合约角色分层 (交叉列) | 1 (主力), 2 (次主力), 3 (非主力/远月) | 跨日更新 | **是 (无漂移风险)**：离散序数编码 $\{1, 2, 3\}$，在第 7 节混频特征中统一对齐 |

---

### 6. 跨期期限结构与合约轮动特征 (Cross-Month Term Structure Features)

源码：`data_preprocess/operator_futures/calendar/cross_month.py`，数据集实际包含 **32 列**：

| # | 特征名称/模板 | 语义解释 | 精确数学公式 | 参数与时间窗口 | 是否相对化与无量纲化 |
|---|-------------|---------|-------------|--------------|---------------------|
| 1-3 | `cm_contract_role_{main/sub/other}` | 合约角色 One-Hot 编码 | $\mathbb{I}\{\text{Role} == r\}, \quad r \in \{\text{main}, \text{sub}, \text{other}\}$ | 动态截面判断 | **是 (无漂移风险)**：离散 $\{0, 1\}$ 分类标志 |
| 4 | `cm_current_main_log_price_ratio` | 当前合约与主力对数比率 | $\ln(P_{\text{current}} / P_{\text{main}})$ | 截面对齐 | **是 (无漂移风险)**：对数价格差，代表相对主力溢折价率，无物理量纲 |
| 5 | `cm_current_main_relative_price_spread`| 当前合约相对主力相对价差 | $\frac{P_{\text{current}} - P_{\text{main}}}{P_{\text{main}}}$ | 截面对齐 | **是 (无漂移风险)**：相对价差百分比，无物理量纲 |
| 6 | `cm_current_main_volume_share_current` | 当前合约相对主力成交量占比 | $\frac{V_{\text{current}}}{V_{\text{current}} + V_{\text{main}} + \epsilon_1}$ | 单 Bar (10m) | **是 (无漂移风险)**：成交活跃度相对份额 $[0.0, 1.0]$ |
| 7 | `cm_current_main_open_interest_share_current`| 当前合约相对主力持仓占比 | $\frac{OI_{\text{current}}}{OI_{\text{current}} + OI_{\text{main}} + \epsilon_1}$ | 单 Bar (10m) | **是 (无漂移风险)**：持仓沉淀相对份额 $[0.0, 1.0]$ |
| 8 | `cm_current_main_log_price_spread_velocity_10m` | 当前与主力价差变动速度 | $\Delta_1(\ln(P_{\text{current}} / P_{\text{main}}))$ | 跨 1 Bar (10m) | **是 (无漂移风险)**：基差斜率变动速度，无物理量纲 |
| 9-10 | `cm_current_main_spread_rolling_zscore_{w}` | 当前与主力价差滚动 Z-Score | $\frac{\text{Spread}_t - \text{Mean}_w(\text{Spread})}{\text{Std}_w(\text{Spread}) + \epsilon_3}$ | $w \in \{48, 192\}$ | **是 (无漂移风险)**：局部滚动 Z-Score 标准化，零均值单位方差 |
| 11-15 | `cm_current_sub_*` | 当前合约与次主力跨期特征 | 对应当前合约与次主力合约的对数比率、相对价差、成交量占比、持仓占比、价差速度 | 截面对齐 | **是 (无漂移风险)**：完全相对化无量纲特征 (5 列) |
| 16-17 | `cm_current_sub_spread_rolling_zscore_{w}` | 当前与次主力价差滚动 Z-Score | $\frac{\text{Spread}_t - \text{Mean}_w(\text{Spread})}{\text{Std}_w(\text{Spread}) + \epsilon_3}$ | $w \in \{48, 192\}$ | **是 (无漂移风险)**：局部滚动 Z-Score 标准化 (2 列) |
| 18-22 | `cm_main_sub_*` | 主力合约与次主力跨期基差组 | 对应主次合约对数比率、相对价差、次主力成交占比、次主力持仓占比、主次价差速度 | 截面对齐 | **是 (无漂移风险)**：核心期限结构指标 (5 列) |
| 23-24 | `cm_main_sub_spread_rolling_zscore_{w}` | 主力与次主力价差滚动 Z-Score | $\frac{\text{Spread}_t - \text{Mean}_w(\text{Spread})}{\text{Std}_w(\text{Spread}) + \epsilon_3}$ | $w \in \{48, 192\}$ | **是 (无漂移风险)**：局部滚动 Z-Score 标准化 (2 列) |
| 25 | `cm_open_interest_shift_speed_10m` | 换月主力资金移仓速度 | $\Delta_1\left(\frac{OI_{\text{sub}}}{OI_{\text{main}} + OI_{\text{sub}} + \epsilon_1}\right)$ | 跨 1 Bar (10m) | **是 (无漂移风险)**：持仓迁移速率，衡量资金换月速度，纯比例增量 |
| 26-27 | `cm_m1_m2_log_price_spread_velocity_10m`, `cm_m1_m2_open_interest_share_m2` | 近月与次近月跨期动力学 | 近月 M1 与次近月 M2 的对数价差速度及持仓份额占比 | 截面对齐 | **是 (无漂移风险)**：无量纲跨期流动性与期限曲线倾斜度 (2 列) |
| 28-29 | `cm_m2_m3_log_price_spread_velocity_10m`, `cm_m2_m3_open_interest_share_m3` | 次近月与远月跨期动力学 | 次近月 M2 与远月 M3 的对数价差速度及持仓份额占比 | 截面对齐 | **是 (无漂移风险)**：无量纲远端期限曲线倾斜度 (2 列) |
| 30 | `cm_m1_m2_m3_butterfly_spread_velocity_10m` | 跨期蝶式价差变动速度 | $\Delta_1(\ln(P_{M1}) - 2\ln(P_{M2}) + \ln(P_{M3}))$ | 跨 1 Bar (10m) | **是 (无漂移风险)**：二阶曲率变化速度，纯无量纲曲率变动 |
| 31-32 | `cm_spread_rolling_zscore_{w}` | 综合跨期价差滚动 Z-Score | $\frac{S_t - \text{Mean}_w(S)}{\text{Std}_w(S) + \epsilon_3}, \quad S = \ln(P_t / P_{\text{main}})$ | $w \in \{48, 192\}$ | **是 (无漂移风险)**：通用基差局部 Z-Score 标准化 (2 列) |

---

### 7. 混频跨日与跨周宏观状态特征 (Mixed-Frequency Daily & Weekly Features)

源码：`data_preprocess/operator_futures/commodity/macro_indicator.py`，数据集实际包含 **165 列**（含 1 列 `prev_day_contract_role_tier` 与 164 列日/周频及对应分位数特征）：

| # | 特征名称/模板 | 语义解释 | 精确数学公式 | 参数与时间窗口 | 是否相对化与无量纲化 |
|---|-------------|---------|-------------|--------------|---------------------|
| 1 | `prev_day_contract_role_tier` | 前一日合约角色分层 | 1 (主力), 2 (次主力), 3 (其他) | 跨日更新 | **是 (无漂移风险)**：离散序数编码 $\{1, 2, 3\}$ |
| 2 | `prev_day_return` | 前 1 日收益率 | $\frac{C_{d-1} - C_{d-2}}{C_{d-2}}$ | $d=1$ 天 | **是 (无漂移风险)**：百分比收益率，无物理量纲 |
| 3 | `prev_day_range_pct` | 前 1 日相对振幅 | $\frac{H_{d-1} - L_{d-1}}{C_{d-2}}$ | $d=1$ 天 | **是 (无漂移风险)**：归一化日波动幅度 |
| 4 | `prev_day_body_pct` | 前 1 日实体相对幅度 | $\frac{\vert C_{d-1} - O_{d-1} \vert}{C_{d-2}}$ | $d=1$ 天 | **是 (无漂移风险)**：日 K 线实体相对比例 |
| 5 | `prev_day_close_position` | 前 1 日收盘在日内极值相对位置 | $\frac{C_{d-1} - L_{d-1}}{H_{d-1} - L_{d-1} + \epsilon_1}$ | $d=1$ 天 | **是 (无漂移风险)**：日内分位数 $[0.0, 1.0]$ |
| 6 | `prev_day_vwap_deviation_pct` | 前 1 日收盘相对 VWAP 偏离度 | $\frac{C_{d-1} - \text{VWAP}_{d-1}}{\text{VWAP}_{d-1}}$ | $d=1$ 天 | **是 (无漂移风险)**：相对偏离百分比 |
| 7 | `prev_day_twap_deviation_pct` | 前 1 日收盘相对 TWAP 偏离度 | $\frac{C_{d-1} - \text{TWAP}_{d-1}}{\text{TWAP}_{d-1}}$ | $d=1$ 天 | **是 (无漂移风险)**：相对偏离百分比 |
| 8-9 | `prev_day_upper/lower_shadow_pct` | 前 1 日上下影线相对幅度 | $\frac{H - \max(O, C)}{C_{\text{prev}}}, \frac{\min(O, C) - L}{C_{\text{prev}}}$ | $d=1$ 天 | **是 (无漂移风险)**：影线高度百分比 (2 列) |
| 10-12 | `prev_day_upper/lower_shadow_to_range`, `prev_day_body_to_range` | 前 1 日影线/实体与振幅比 | $\frac{\text{Shadow}}{H - L + \epsilon_1}, \frac{\vert C - O \vert}{H - L + \epsilon_1}$ | $d=1$ 天 | **是 (无漂移风险)**：K 线形态无量纲比例 $[0.0, 1.0]$ (3 列) |
| 13 | `prev_day_trade_imbalance` | 前 1 日主动买卖笔数失衡率 | $\frac{N_{\text{up}} - N_{\text{down}}}{N_{\text{up}} + N_{\text{down}} + \epsilon_1}$ | $d=1$ 天 | **是 (无漂移风险)**：主动买卖笔数相对失衡比率 $[-1.0, 1.0]$ |
| 14 | `prev_day_open_interest_change` | 前 1 日持仓变动率 | $\frac{OI_{d-1} - OI_{d-2}}{OI_{d-2} + \epsilon_1}$ | $d=1$ 天 | **是 (无漂移风险)**：持仓相对增长率 |
| 15 | `prev_day_turnover_rate` | 前 1 日持仓换手率 | $\frac{V_{d-1}}{OI_{d-1} + \epsilon_1}$ | $d=1$ 天 | **是 (无漂移风险)**：成交量与持仓量之比，纯无量纲周转倍数 |
| 16-17 | `prev_day_trade_up/down_ratio` | 前 1 日主动买/卖成交笔数占比 | $\frac{N_{\text{up}}}{N_{\text{total}}}, \frac{N_{\text{down}}}{N_{\text{total}}}$ | $d=1$ 天 | **是 (无漂移风险)**：成交方向占比 $[0.0, 1.0]$ (2 列) |
| 18-19 | `prev_day_limit_up/down_single_sided_ratio` | 前 1 日单边涨/跌停封板时间占比 | 交易所逐笔挂单涨跌停封死时长比率 | $d=1$ 天 | **是 (无漂移风险)**：单边封停时间占比 $[0.0, 1.0]$ (2 列) |
| 20-37 | `prev_day_*_quantile_rank` | 前 1 日指标全市场分位数秩 | $\text{Rank}_{\text{cross-section}}(X) / N_{\text{contracts}}$ | $d=1$ 天截面 | **是 (无漂移风险)**：截面经验分位数 $[0.0, 1.0]$，对应上述 18 个指标 (18 列) |
| 38-72 | `prev_{d}_day_{metric}` | 多日复合宏观状态特征 | 针对 7 大核心指标计算 $d$ 日复合状态 | $d \in \{2, 5, 10, 15, 30\}$ | **是 (无漂移风险)**：涵盖 `trade_imbalance`、`open_interest_change`、`turnover_rate`、`trade_up/down_ratio`、`limit_up/down_ratio` (5 窗口 $\times$ 7 指标 = 35 列) |
| 73-107 | `prev_{d}_day_{metric}_quantile_rank` | 多日指标全市场分位数秩 | 对应上述 35 个多日复合指标的截面分位数 | $d \in \{2, 5, 10, 15, 30\}$ | **是 (无漂移风险)**：截面经验分位数 $[0.0, 1.0]$ (35 列) |
| 108-121 | `prev_week_{metric}` | 前 1 周宏观状态特征 | 涵盖周频收益率、振幅比、实体比、极值位置、VWAP/TWAP 偏离、影线振幅比、买卖失衡、持仓变动、换手率、方向占比 | $w=1$ 周 (5 交易日) | **是 (无漂移风险)**：完全相对化周频状态 (14 列) |
| 122-135 | `prev_week_{metric}_quantile_rank` | 前 1 周指标全市场分位数秩 | 对应上述 14 个周频特征的截面分位数排名 | $w=1$ 周截面 | **是 (无漂移风险)**：截面经验分位数 $[0.0, 1.0]$ (14 列) |
| 136-150 | `prev_{w}_week_{metric}` | 多周复合宏观状态特征 | 针对 5 大核心指标计算 $w$ 周复合状态 | $w \in \{2, 4, 6\}$ 周 | **是 (无漂移风险)**：涵盖 `trade_imbalance`、`open_interest_change`、`turnover_rate`、`trade_up/down_ratio` (3 窗口 $\times$ 5 指标 = 15 列) |
| 151-165 | `prev_{w}_week_{metric}_quantile_rank` | 多周指标全市场分位数秩 | 对应上述 15 个多周复合指标的截面分位数排名 | $w \in \{2, 4, 6\}$ 周截面 | **是 (无漂移风险)**：截面经验分位数 $[0.0, 1.0]$ (15 列) |

---

### 8. 单列价格与成交量滚动时序特征 (Time Rolling Price & Volume Features)

源码：`data_preprocess/operator_futures/time_operator/time_operator_util.py`，数据集实际包含 **623 列**：

滚动算子的基础参数窗口集合为 $W = \{2, 6, 12, 16, 24, 48, 96, 192\}$（8 个尺度，覆盖 20 分钟至 32 小时交易跨度）。

| # | 特征名称/模板 | 语义解释 | 精确数学公式 | 参数与时间窗口 | 是否相对化与无量纲化 |
|---|-------------|---------|-------------|--------------|---------------------|
| 1-30 | `{base}_log_return_2` | 30 个基础序列的 2 步对数差分 | $\ln(X_t / X_{t-2})$ | 窗口 2 Bar (20m) | **是 (无漂移风险)**：对数收益率/变动率，覆盖 10 档挂单价、10 档挂单量占比、4 个微观 WAP、2 个买卖价差、3 个执行成交量、1 个标记价格 (30 列) |
| 31-270 | `{base}_trend_{w}` | 30 个基础序列的归一化线性斜率 | $\frac{X_t - X_{t-w}}{w \cdot (X_{t-w} + \epsilon_2)}$ | $w \in W, 30$ 个基础序列 | **是 (无漂移风险)**：消除基数影响的无量纲相对线性趋势速度 ($30 \times 8 = 240$ 列) |
| 271-342 | `{op}_{w}_std_norm_origin` | 9 组核心算子的标准差归一化变体 | $\frac{X_t - \text{Mean}_w(X)}{\text{Std}_w(X) + \epsilon_3}$ | 9 算子 $\times 8$ 窗口 | **是 (无漂移风险)**：局部滚动 Z-Score 平稳化，涵盖 `beta`, `ma`, `max`, `min`, `qtld`, `qtlu`, `roc`, `rsv`, `vma` ($9 \times 8 = 72$ 列) |
| 343-582 | `{price_op}_{w}_origin` | 30 组价格滚动统计算子 | 详见下方价格算子公式定义 | 30 算子 $\times 8$ 窗口 | **是 (无漂移风险)**：均除以现价 $C_t$ 或窗口极值极差，实现百分比相对化 ($30 \times 8 = 240$ 列) |
| 583-622 | `{volume_op}_{w}_origin` | 5 组成交量滚动统计算子 | 详见下方成交量算子公式定义 | 5 算子 $\times 8$ 窗口 | **是 (无漂移风险)**：均除以当前成交量 $V_t$ 或总成交量，实现倍数相对化 ($5 \times 8 = 40$ 列) |
| 623 | `log_volume_origin` | 单 Bar 对数成交量 | $\ln(V_t + 1.0)$ | 单 Bar 瞬时 | **否 (高漂移风险)**：单 Bar 绝对成交量对数变换，未做局部滚动均值或跨期总量归一化，受不同年份市场活跃度规模漂移影响 |

#### 价格与成交量滚动时序算子精确数学公式清单 ($w \in W$)

- **变动与动量**：
  - `roc_{w}`: $\frac{C_{t-w}}{C_t}$ (无量纲比值)
  - `beta_{w}`: $\frac{C_{t-w} - C_t}{w \cdot C_t}$ (相对变化速率)
- **极值与分位数**：
  - `ma_{w}`: $\frac{\text{Mean}_w(C)_t}{C_t}$ (均价现价比)
  - `std_{w}`: $\frac{\text{Std}_w(C)_t}{C_t}$ (相对波动率)
  - `max_{w}`, `min_{w}`: $\frac{\max_w(C)}{C_t}, \frac{\min_w(C)}{C_t}$ (极值现价比)
  - `qtlu_{w}`, `qtld_{w}`: $\frac{\text{Quantile}_{0.8}(C)}{C_t}, \frac{\text{Quantile}_{0.2}(C)}{C_t}$ (分位数现价比)
  - `rank_{w}`: $\frac{\text{Rank}_w(C_t)}{w}$ (窗口内时间分位数 $[0, 1]$)
  - `imax_{w}`, `imin_{w}`, `imxd_{w}`: $\frac{\text{argmax}_w(H)}{w}, \frac{\text{argmin}_w(L)}{w}, \text{imax} - \text{imin}$ (极值时间距离比)
  - `rsv_{w}`: $\frac{C_t - \min_w(L)}{\max_w(H) - \min_w(L) + \epsilon_2}$ (威廉/KD RSV指标 $[0, 1]$)
- **通道与布林带**：
  - `bollinger_bandwidth_{w}`: $\frac{2 \cdot 2\text{Std}_w(C)}{\text{Mean}_w(C)}$ (布林带相对带宽)
  - `bollinger_upper_{w}`, `bollinger_lower_{w}`: $\frac{\text{Mean}_w(C) \pm 2\text{Std}_w(C)}{C_t}$ (上下轨相对现价比)
- **枢轴点 (Pivot Points)**：
  - `pivot_pp_{w}`: $\frac{\max(H) + \min(L) + C}{3 \cdot C_t}$
  - `pivot_r1_{w}`, `pivot_s1_{w}`: $\frac{2 \cdot PP - \min(L)}{C_t}, \frac{2 \cdot PP - \max(H)}{C_t}$
  - `pivot_r2_{w}`, `pivot_s2_{w}`: $\frac{PP + (\max(H) - \min(L))}{C_t}, \frac{PP - (\max(H) - \min(L))}{C_t}$
- **方向计数与正负积累**：
  - `cntp_{w}`, `cntn_{w}`: $\frac{1}{w} \sum_{k=0}^{w-1} \mathbb{I}\{r_{t-k} > 0\}, \frac{1}{w} \sum_{k=0}^{w-1} \mathbb{I}\{r_{t-k} < 0\}$ (涨跌 Bar 比例)
  - `cntd_{w}`: $\frac{\text{cntp} - \text{cntn}}{\text{cntp} + \text{cntn} + 10^{-6}}$ (涨跌 Bar 净差占比)
  - `sump_{w}`, `sumn_{w}`, `sumd_{w}`: 过去 $w$ 期内收益率正向增量占绝对增量之和的比率 ($[0, 1]$ 与 $[-1, 1]$)
- **价量协同**：
  - `corr_{w}`: 价格与成交量滚动对数相关系数 $\text{Corr}_w(C, \ln(V+1)) \in [-1, 1]$ (无量纲)
  - `cord_{w}`: 收益率变动与成交量变动滚动相关系数 $\text{Corr}_w(\Delta C / C, \Delta V / V) \in [-1, 1]$ (无量纲)
  - `wvma_{w}`: 加权成交量均值比率 $\frac{\text{Std}_w(\vert \Delta C / C \vert \cdot V)}{\text{Mean}_w(\vert \Delta C / C \vert \cdot V) + \epsilon_2}$ (无量纲离散系数)
- **成交量动力学算子**：
  - `vma_{w}`: $\frac{\text{Mean}_w(V)_t}{V_t + \epsilon_2}$ (滚动均量与现量比)
  - `vstd_{w}`: $\text{clip}\left(\frac{\text{Std}_w(V)_t}{\max(V_t, 1.0)}, 0.0, 10.0\right)$ (成交量相对离散度)
  - `vsump_{w}`, `vsumn_{w}`, `vsumd_{w}`: 成交量正向增量占总绝对增量的相对累计比例 ($[0, 1]$ 与 $[-1, 1]$)

---

### 9. 物理风险与流动性状态特征 (Physical Risk & Liquidity State Features)

源码：`data_preprocess/operator_futures/time_operator/time_operator_util.py` (`process_volatility_features` 等)，数据集实际包含 **80 列**（10 个物理指标 $\times$ 8 个时间窗口 $W = \{2, 6, 12, 16, 24, 48, 96, 192\}$）：

| # | 特征名称/模板 | 语义解释 | 精确数学公式 | 参数与时间窗口 | 是否相对化与无量纲化 |
|---|-------------|---------|-------------|--------------|---------------------|
| 1-8 | `atr_pct_{w}` | 真实波幅相对百分比 | $\frac{\text{Mean}_w(\text{TR})_t}{C_t} \cdot 100\%, \quad \text{TR} = \max(H - L, \vert H - C_{t-1} \vert, \vert L - C_{t-1} \vert)$ | $w \in W$ (8 列) | **是 (无漂移风险)**：真实波幅除以现价，纯无量纲百分比 |
| 9-16 | `historical_volatility_{w}` | 历史年化波动率 | $\text{Std}_w(r)_t \cdot \sqrt{\text{BarsPerDay}}$ | $w \in W$ (8 列) | **是 (无漂移风险)**：基于对数收益率的滚动标准差，消除绝对价格 |
| 17-24 | `rolling_volatility_{w}` | 指数加权移动波动率 | $\text{EWM\_Std}_w(r)_t$ | $w \in W$ (8 列) | **是 (无漂移风险)**：基于对数收益率的指数加权波动率 |
| 25-32 | `parkinson_volatility_{w}` | 帕金森极值方差波动率 | $\sqrt{\frac{1}{4 \ln 2 \cdot w} \sum_{k=0}^{w-1} \ln\left(\frac{H_{t-k}}{L_{t-k}}\right)^2}$ | $w \in W$ (8 列) | **是 (无漂移风险)**：基于对数最高最低价差，纯相对波动率 |
| 33-40 | `garman_klass_volatility_{w}` | 加曼-克拉斯开高低收波动率 | $\sqrt{\frac{1}{w} \sum_{k=0}^{w-1} \left[0.5 \ln\left(\frac{H}{L}\right)^2 - (2\ln 2 - 1)\ln\left(\frac{C}{O}\right)^2\right]}$ | $w \in W$ (8 列) | **是 (无漂移风险)**：全 K 线几何开高低收连续波动率估计 |
| 41-48 | `realized_volatility_{w}` | 滚动已实现波动率 | $\sqrt{\sum_{k=0}^{w-1} r_{t-k}^2}$ | $w \in W$ (8 列) | **是 (无漂移风险)**：收益率平方和平方根，纯无量纲波动率 |
| 49-56 | `relative_volume_{w}` | 相对成交量倍数 | $\frac{V_t}{\text{Mean}_w(V)_t + \epsilon_2}$ | $w \in W$ (8 列) | **是 (无漂移风险)**：现量相对自身局部均量比值，消除绝对量级 |
| 57-64 | `relative_amount_{w}` | 相对成交金额倍数 | $\frac{\text{tradeval}_t}{\text{Mean}_w(\text{tradeval})_t + \epsilon_2}$ | $w \in W$ (8 列) | **是 (无漂移风险)**：成交额相对自身局部均额比值，消除绝对货币量级 |
| 65-72 | `relative_open_interest_{w}` | 相对持仓量倍数 | $\frac{OI_t}{\text{Mean}_w(OI)_t + \epsilon_2}$ | $w \in W$ (8 列) | **是 (无漂移风险)**：持仓量相对自身局部均持仓比值，消除绝对量级 |
| 73-80 | `open_interest_change_ratio_{w}` | 跨窗口持仓量变动率 | $\frac{OI_t - OI_{t-w}}{OI_{t-w} + \epsilon_2}$ | $w \in W$ (8 列) | **是 (无漂移风险)**：持仓相对增长率，消除绝对手数 |

---

### 10. 动量、振荡器与高阶增强特征 (Momentum, Oscillators & Enhanced Features)

源码：`data_preprocess/operator_futures/time_operator/time_operator_util.py` (`process_enhanced_state_features` 与 `_rolling_log_price_anchors`)，数据集实际包含 **37 列**：

| # | 特征名称/模板 | 语义解释 | 精确数学公式 | 参数与时间窗口 | 是否相对化与无量纲化 |
|---|-------------|---------|-------------|--------------|---------------------|
| 1 | `price_velocity_10m` | 价格短期均线速度 | $\frac{\text{EMA}_{10}(C) - \text{EMA}_{20}(C)}{C}$ | span=(10, 20) | **是 (无漂移风险)**：双均线差除以现价归一化，无物理量纲 |
| 2 | `price_acceleration_10m_norm` | 标准化价格加速度 | $\frac{\Delta_1(\text{price\_velocity\_10m})}{\text{RollingStd}_{10}(r) + \epsilon_3}$ | span=10, std=10 | **是 (无漂移风险)**：除以收益率局部滚动标准差，标准化无量纲 |
| 3-5 | `ema_divergence_{s1}_{s2}` | 多尺度 EMA 发散度 | $\frac{\text{EMA}_{s1}(C) - \text{EMA}_{s2}(C)}{\text{EMA}_{s2}(C)}$ | (20, 60), (60, 120), (20, 120) | **是 (无漂移风险)**：双均线相对百分比发散度 (3 列) |
| 6 | `macd_histogram_norm` | 归一化 MACD 柱状图 | $\text{DIF} - \text{DEA}, \quad \text{DIF} = \frac{\text{EMA}_{12}(C) - \text{EMA}_{26}(C)}{C}, \text{DEA} = \text{EMA}_9(\text{DIF})$ | (12, 26, 9) | **是 (无漂移风险)**：DIF 除以现价归一化，消除绝对价格基数 |
| 7 | `rsi_14_norm` | 归一化相对强弱指标 | $\frac{\text{RSI}_{14} - 50}{50}, \quad \text{RSI}_{14} = 100 \cdot \frac{\text{EWM}(\text{Gain})}{\text{EWM}(\text{Gain}) + \text{EWM}(\text{Loss})}$ | Wilder 14 周期 | **是 (无漂移风险)**：线性映射至 $[-1.0, 1.0]$，无物理量纲 |
| 8-10 | `plus_di_14`, `minus_di_14`, `adx_14` | 经典 DMI/ADX 趋势强度 | $+DI_{14} = 100 \frac{\text{EWM}(+DM)}{\text{ATR}}, -DI_{14} = 100 \frac{\text{EWM}(-DM)}{\text{ATR}}, ADX = \text{EWM}\left(100 \frac{\vert +DI - -DI \vert}{+DI + -DI}\right)$ | Wilder 14 周期 | **是 (无漂移风险)**：除以 ATR 归一化，无物理量纲 (3 列) |
| 11-12 | `stoch_k_14_norm`, `stoch_d_14_norm` | 归一化随机振荡器 KD | $\frac{K - 50}{50}, \frac{D - 50}{50}, \quad K = \text{EWM}_{3}(\text{RSV}_{14}), D = \text{EWM}_{3}(K)$ | 周期 (14, 3) | **是 (无漂移风险)**：线性映射至 $[-1.0, 1.0]$，无物理量纲 (2 列) |
| 13 | `cci_14_norm` | 归一化顺势指标 | $\text{clip}\left(\frac{TP - \text{Mean}_{14}(TP)}{0.015 \cdot \text{MeanDev}_{14}(TP)} \cdot \frac{1}{100}, -3.0, 3.0\right), TP = \frac{H+L+C}{3}$ | 周期 14 | **是 (无漂移风险)**：平均绝对偏差归一化并硬截断至 $[-3.0, 3.0]$ |
| 14-15 | `ema_slope_{w}` | EMA 宏观长期斜率 | $\frac{\text{EMA}_{20}(C)_t - \text{EMA}_{20}(C)_{t-w}}{\text{EMA}_{20}(C)_{t-w} \cdot w}$ | $w \in \{96, 192\}$ | **是 (无漂移风险)**：相对百分比变动斜率 (2 列) |
| 16-17 | `vwap_slope_{w}` | VWAP 宏观长期斜率 | $\frac{\text{VWAP}_t - \text{VWAP}_{t-w}}{\text{VWAP}_{t-w} \cdot w}$ | $w \in \{96, 192\}$ | **是 (无漂移风险)**：相对百分比变动斜率 (2 列) |
| 18-19 | `cvd_slope_{w}` | 累积成交量增量 (CVD) 斜率 | $\frac{\sum_{k=0}^{w-1} (V_k \cdot \text{Direction}_k)}{\sum_{k=0}^{w-1} V_k}$ | $w \in \{96, 192\}$ | **是 (无漂移风险)**：有向成交量除以总成交量 $[-1.0, 1.0]$，纯无量纲比例 (2 列) |
| 20-21 | `log_price_slope_{w}` | 对数价格最小二乘趋势斜率 | $\text{OLS\_Slope}_w(\ln(C))$ | $w \in \{48, 96\}$ | **是 (无漂移风险)**：对数价格局部无偏线性回归斜率 (2 列) |
| 22 | `log_price_slope_quantile_192` | 对数斜率滚动分位数秩 | $\text{QuantileRank}_{192}(\text{log\_price\_slope\_48})$ | 滚动 192 Bar | **是 (无漂移风险)**：局部滚动分位数 $[0.0, 1.0]$ |
| 23 | `signed_efficiency_48` | 有符号价格运行效率比 | $\text{clip}\left(\frac{\ln(C_t) - \ln(C_{t-48})}{\sum_{k=0}^{47} \vert \ln(C_{t-k}) - \ln(C_{t-k-1}) \vert + \epsilon_3}, -1.0, 1.0\right)$ | $w=48$ | **是 (无漂移风险)**：净位移与累积路径之比 $[-1.0, 1.0]$ |
| 24 | `trend_r2_48` | 对数价格回归拟合优度 $R^2$ | $\frac{(\text{Slope}_{48})^2 \sum (t - \bar{t})^2}{\sum (\ln(C) - \overline{\ln(C)})^2}$ | $w=48$ | **是 (无漂移风险)**：判定系数 $[0.0, 1.0]$，纯无量纲趋势纯度 |
| 25-26 | `trend_to_noise_{w}` | 趋势与噪声信噪比 | $\frac{\text{log\_price\_slope}_w}{\max(\text{RollingStd}_w(r), \epsilon_3)}$ | $w \in \{48, 96\}$ | **是 (无漂移风险)**：趋势斜率除以收益率局部波动率 (2 列) |
| 27 | `trade_direction_net_ratio_5m` | 主动买卖方向净比例 | $\text{clip}\left(\frac{N_{\text{up}} - N_{\text{down}}}{N_{\text{up}} + N_{\text{down}} + \epsilon_3}, -1.0, 1.0\right)$ | 单 Bar 瞬时 | **是 (无漂移风险)**：瞬时主动买卖笔数相对净差 $[-1.0, 1.0]$ |
| 28 | `trade_direction_persistence_20m` | 买卖方向指数平滑持续性 | $\text{EMA}_{20}(\text{trade\_direction\_net\_ratio\_5m})$ | span = 20 | **是 (无漂移风险)**：平滑后的买卖方向持续性 $[-1.0, 1.0]$ |
| 29 | `price_oi_vol_interaction_10m` | 价格-持仓-量价非线性交互 | $\text{sign}(\Delta_2 C) \cdot \text{sign}(\Delta_2 OI) \cdot \frac{V}{\text{Mean}_{10}(V) + \epsilon_3}$ | 跨 2 步差分 | **是 (无漂移风险)**：方向二值符号与相对成交量乘积，无绝对量纲 |
| 30 | `oi_change_rate_norm_10m` | 10 分钟持仓变动率归一化 | $\frac{OI_t - OI_{t-10}}{OI_{t-10} + \epsilon_3}$ | 跨 10 Bar (100m) | **是 (无漂移风险)**：相对持仓变动百分比 |
| 31 | `realized_vol_zscore_192` | 波动率区制偏离 Z-Score | $\frac{\text{RV}_{12} - \text{Mean}_{192}(\text{RV}_{12})}{\text{Std}_{192}(\text{RV}_{12}) + \epsilon_3}$ | 滚动 192 Bar (约 4 天) | **是 (无漂移风险)**：滚动 Z-Score 零均值单位方差平稳化 |
| 32 | `realized_vol_term_ratio_12_48` | 波动率期限结构期限比 | $\frac{\text{Realized\_Vol}_{12}}{\text{Realized\_Vol}_{48} + \epsilon_3}$ | 短期 12 对中期 48 比率 | **是 (无漂移风险)**：短期与长期波动率之比，纯无量纲比值 |
| 33 | `log_return_vol_quantile_192` | 收益率波动率滚动分位数秩 | $\text{QuantileRank}_{192}(\text{Volatility}_{48})$ | 滚动 192 Bar | **是 (无漂移风险)**：局部滚动经验分位数 $[0.0, 1.0]$ |
| 34-35 | `macro_oi_change_rate_{w}` | 宏观持仓量迁移变动率 | $\text{clip}\left(\frac{OI_t - OI_{t-w}}{\text{Mean}_w(OI) + 1.0}, -0.5, 0.5\right)$ | $w \in \{48, 240\}$ | **是 (无漂移风险)**：持仓变动除以滚动均持仓，相对变动率截断 (2 列) |
| 36-37 | `macro_turnover_rate_log_{w}` | 宏观对数累计换手率 | $\ln\left(1.0 + \frac{\sum_w V_t}{OI_t + 1.0}\right)$ | $w \in \{48, 240\}$ | **是 (无漂移风险)**：对数累计换手率，无物理量纲 (2 列) |

---

### 11. 附录：数据合并伴生列说明与去重规范 (Join Artifact Handling)

数据集中的 245 个以 `_right` 为后缀的特征列（例如 `UpperLimitPrice_right`、`ask1_price_right`、`prev_day_return_right` 等）：
1. **生成成因**：在 Stage 4/Stage 6 混频数据多表拼接（如按交易日连接日频宏观表、按合约月连接跨期表）时，左右表中存在同名基础字段，系统在连接时未做重合字段前置投影裁剪，从而生成了后缀为 `_right` 的副本列。
2. **审查与剔除规则**：在 Stage 6.5 特征审查与 Stage 7 三流特征选择中，`eligibility.py` 定义了严格的黑名单过滤规则：
   ```python
   EXCLUDED_SUFFIXES = ("_right", "_y", "_duplicate")
   ```
   所有带有 `_right` 后缀的列被 100% 自动拦截，确保特征选择空间严格限定在 1,077 个无重复的候选特征全集中。

## 三、三流特征过滤体系：门禁、阈值与物理/数学原理

### 1. 架构动机：三流解耦（Triple-Stream Decoupling）的物理与数学必然性

在 FineFT 统一强化学习交易系统中，高层宏观路由模型（High-Level Routing）与底层执行策略网络（Low-Level Agents）面临截然不同的任务目标与数学损失约束：

1. **趋势区制轴（Slope Regimes: Down / Flat / Up）**:
   - **目标**: 识别具有有符号方向性的市场动量与价格漂移，驱动高层路由在多空头寸策略之间切换。
   - **物理需求**: 特征必须具备**奇对称性（Odd Symmetry）**和跨 Down/Flat/Up 区制的**严格单调性（Monotonicity）**。
   - **解耦约束**: **严禁纯波动率/离散度指标（如 ATR、Bollinger 宽度、Parkinson 波动率）混入**。若混入无方向性的离散度指标，将导致 VAE 潜空间 $z_{\text{slope}}$ 的方向信噪比被稀释超过 50%，使高层路由产生高频无序翻转。

2. **波动率区制轴（Volatility Regimes: Low / Mid / High Vol）**:
   - **目标**: 识别市场的绝对波动剧烈程度，驱动高层路由在保守套利与激进趋势/做市之间切换。
   - **物理需求**: 衡量标量分散度（Dispersion），特征必须针对未来**绝对收益 $|R_{t, w}|$** 具备显著的秩相关性（$\text{VolRankIC} \ge 0.030$），且跨 Low $\to$ Mid $\to$ High Vol 严格单调递增。
   - **解耦约束**: **严禁有符号方向性指标（如收益率、动量斜率、多空差比）混入**。零均值对称的双峰分布会迫使 VAE 高斯重构解码器 $\|x - \hat{x}\|^2$ 坍塌，在单边暴涨暴跌行情下引发重构误差伪峰，触发虚假的 OOD 熔断防线。

3. **RL 决策流（RL Decision Stream）**:
   - **目标**: 直接作为底层 Agent 的状态观测输入（Actor/Critic 策略网络），决定当期具体的挂单、撤单、持仓与执行行为。
   - **物理需求**: 兼具微观 5 档订单簿即时失衡（OFI、微观价差）、跨期期限结构与宏观动量，保留高维表征能力（$K \in [100, 155]$），同时通过条件锚点机制保留在极端状态下具有极高预测力的稀缺特征。

---

### 2. 共享前置门禁与数据净化 (Data Hygiene & Shared Pre-Filters)

在分流之前，所有候选特征首先在多合约数据集上执行共享数据净化：

1. **零/低方差门禁 (`min_variance = 1e-6`)**:
   - 过滤公式: $\text{Var}_m(f) \ge 10^{-6}, \quad \forall m$。剔除几乎恒定的静止特征。
2. **离散众数频率门禁 (`max_mode_frequency = 0.98`)**:
   - 过滤公式: $\max_{v} \frac{\sum \mathbb{I}\{f_t = v\}}{N} \le 0.98$。剔除 98% 以上取值完全相同的伪常数特征。
3. **稳健异常值双侧截断 (`IQR Winsorization`)**:
   - 采用分位数间距进行软饱和截断：$[Q_1 - 5 \cdot \text{IQR}, Q_3 + 5 \cdot \text{IQR}]$，消除极端 Tick 异常值对后续统计协方差的污染。
4. **全局卫生黑名单 (`global blacklist`)**:
   - 剔除历史已证实存在前瞻泄露或非平稳伪相关的绝对价格列及未对齐列。

---

### 3. 九大核心过滤门禁与数学定义

在多合约三流评估中，每个特征流独立经历以下统计检验门禁的逐级筛选：

#### 门禁 1：流特定黑名单门禁 (Stream Blacklist Gate)
- **Slope VAE 黑名单**: 屏蔽所有包含 `_std_`, `_volatility_`, `atr_`, `bollinger_bandwidth_`, `vstd_`, `realized_vol_` 等纯分散度指标。
- **Volatility VAE 黑名单**: 屏蔽所有包含 `_log_return_`, `_trend_`, `roc_`, `macd_`, `kmid`, `cntd_`, `sumd_`, `vsumd_` 等带符号方向性指标。
- **RL 黑名单**: 屏蔽特定采样周期下的高频不稳定微观伪指标。

#### 门禁 2：多合约跨期分布漂移门禁 (Distribution Drift Gate, PSI)
计算各合约间分布的总体稳定性指标（Population Stability Index, PSI）。将样本划分为 $B=10$ 个等频分箱，对任意合约对 $(A, B)$：
$$\text{PSI}(A, B) = \sum_{b=1}^{10} (P_{A, b} - P_{B, b}) \cdot \ln\left(\frac{P_{A, b} + \epsilon}{P_{B, b} + \epsilon}\right)$$
- **Mean PSI**: 所有合约对的平均 PSI，$\overline{\text{PSI}} = \frac{2}{M(M-1)} \sum_{i < j} \text{PSI}(m_i, m_j)$。
- **Max Pair PSI**: 任意两合约间的最大 PSI，$\text{PSI}_{\max} = \max_{i, j} \text{PSI}(m_i, m_j)$。
- **Forward Outpost PSI**: 针对未来前哨测试集（Test Outpost）的分布偏离度检验。

#### 门禁 3：持续性噪声门禁 (Persistence Noise Gate)
针对高频时序收益特征（`_log_return_1`, `_log_return_2`），拟合 Ornstein-Uhlenbeck 连续均值回归过程：
$$\Delta X_t = \lambda X_{t-1} + \epsilon_t, \quad t_{1/2} = -\frac{\ln 2}{\lambda}$$
若中位数半衰期 $t_{1/2} < 2.0$ 根 Bar，表明该特征属于白噪声级别的高频微观抖动，在 VAE 宏观区制流中予以剔除（`filter_micro_persistence = True`）。

#### 门禁 4：硬性预测相关性门禁 (Predictive Hard Filter)
在目标决策窗口 $w=6$（即 60 分钟未来视界）上评估特征与未来目标的 Spearman 秩相关系数（RankIC）：
- **方向性评估 (Slope / RL)**:
  $$R_{t, w} = \frac{P_{t+w} - P_t}{P_t}, \quad \text{RankIC}(f) = \text{Corr}_{\text{Spearman}}(f_t, R_{t, w})$$
- **离散度评估 (Volatility VAE)**:
  $$V_{t, w} = |R_{t, w}|, \quad \text{VolRankIC}(f) = \text{Corr}_{\text{Spearman}}(f_t, V_{t, w})$$
- **门禁要求**: 跨合约均值 $|\overline{\text{RankIC}}| \ge \text{Threshold}$。

#### 门禁 5：跨合约符号一致性门禁 (Sign Consistency Gate)
检验特征在不同到期月份合约上预测方向的一致性，防范单一合约主导的过拟合：
$$\text{SignConsistency}(f) = \frac{1}{M} \sum_{m=1}^M \mathbb{I}\{\text{sign}(\text{IC}_m) = \text{sign}(\overline{\text{IC}})\}$$
要求符号一致比例达到预设阈值（例如 75% 的合约必须同向）。

#### 门禁 6：预测稳定性信息比率门禁 (Stability IR Gate)
评估 RankIC 在不同合约间的离散度，衡量特征跨周期泛化稳定性：
$$\text{RankIC\_IR}(f) = \frac{|\overline{\text{RankIC}}|}{\text{Std}_m(\text{RankIC}) + 10^{-6}}$$

#### 门禁 7：区制单调性门禁 (Regime Monotonicity Gate)
根据 Butterworth 滤波提取的斜率与波动率，将历史样本严格切分为 3 个宏观区制：
- **Slope VAE**: 分为 Down (0), Flat (1), Up (2)。要求各区制内特征均值严格单调：
  $$(m_0 < m_1 < m_2) \quad \lor \quad (m_0 > m_1 > m_2)$$
- **Volatility VAE**: 分为 Low Vol (0), Mid Vol (1), High Vol (2)。要求特征均值随市场活跃度单调递增：
  $$m_0 < m_1 < m_2$$

#### 门禁 8：区制显著性单因素方差分析门禁 (ANOVA F-Test Gate)
检验特征在不同区制组间的均值差异是否在统计学上显著超越组内方差：
$$F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} = \frac{\sum_{g=1}^3 N_g (\bar{x}_g - \bar{x})^2 / (3 - 1)}{\sum_{g=1}^3 \sum_{i=1}^{N_g} (x_{gi} - \bar{x}_g)^2 / (N - 3)}$$
要求 $F \ge \text{min\_anova\_f}$，确保入选特征能够对宏观区制提供坚实的概率分割。

#### 门禁 9：非线性综合评分与正交聚类去重门禁 (Composite Priority & Orthogonal Dedup)
1. **多目标加权综合评分 (Composite Score)**:
   $$\text{Score} = w_{\text{psi}} \cdot \text{Rank}(\text{PSI}^{-1}) + w_{\text{ic}} \cdot \text{Rank}(|\text{IC}|) + w_{\text{cb}} \cdot \text{Rank}(\text{CatBoost})$$
2. **多合约加权 Spearman 距离与 Ward 层次聚类**:
   $$\rho(f_i, f_j) = \frac{1}{M} \sum_{m=1}^M \text{SpearmanCorr}_m(f_i, f_j), \quad d(f_i, f_j) = \sqrt{\frac{1 - \rho(f_i, f_j)}{2}}$$
   基于 Ward 最小方差法（Ward's Linkage）对特征树进行剪枝，限制各流聚类簇数在 $[K_{\min}, K_{\max}]$，优先选取簇内综合评分最高的代表性特征。
3. **方差膨胀因子 (VIF) 多重共线性剪枝**:
   $$VIF_j = \frac{1}{1 - R_j^2} \le \text{max\_vif}$$

---

### 4. 三流门禁参数与阈值全景对比表

| 门禁与参数维度 | 符号与配置项 | Slope VAE (趋势区制流) | Volatility VAE (波动区制流) | RL Decision (交易决策流) |
|---|---|---|---|---|
| **流名称** | `profile.name` | `vae_slope` | `vae_volatility` | `rl_decision` |
| **优化目标** | `target_metric` | `RankIC` (有符号未来收益) | `VolRankIC` (绝对收益离散度) | `RankIC` (有符号未来收益) |
| **平均分布漂移上限** | `max_mean_psi` | $\le \mathbf{0.10}$ (极严格) | $\le \mathbf{0.12}$ (极严格) | $\le \mathbf{0.45}$ (允许微观自适应) |
| **最大合约对漂移上限** | `max_pair_psi` | $\le \mathbf{0.20}$ | $\le \mathbf{0.25}$ | $\le \mathbf{1.30}$ |
| **硬相关性门槛** | `min_abs_ic` | $\ge \mathbf{0.020}$ | $\ge \mathbf{0.030}$ | $\ge \mathbf{0.010}$ |
| **符号一致性下限** | `min_sign_consistency` | $\ge \mathbf{0.75}$ (75% 合约同向) | $\ge \mathbf{0.75}$ (75% 合约同向) | $\ge \mathbf{0.55}$ (兼顾短期反转) |
| **预测稳定性下限** | `min_rank_ic_ir` | $\ge \mathbf{0.35}$ | $\ge \mathbf{0.40}$ | $\ge \mathbf{0.18}$ |
| **高频噪声半衰期过滤** | `filter_micro_persistence` | **开启 (True)** ($t_{1/2} \ge 2.0$) | **开启 (True)** ($t_{1/2} \ge 2.0$) | **关闭 (False)** (保留即时高频冲击) |
| **区制单调性检验** | `require_monotonic` | **开启 (Down/Flat/Up 单调)** | **开启 (Low/Mid/High 严格递增)** | **关闭 (False)** |
| **区制方差检验** | `min_anova_f` | $\ge \mathbf{4.0}$ | $\ge \mathbf{6.0}$ | $\mathbf{0.0}$ (不设限) |
| **综合评分权重: PSI** | `psi_weight` | **0.45** (稳定性优先) | **0.45** (稳定性优先) | **0.15** |
| **综合评分权重: IC** | `rank_ic_weight` | **0.35** | **0.35** | **0.70** (预测力优先) |
| **综合评分权重: CatBoost** | `catboost_weight`| **0.20** | **0.20** | **0.15** |
| **正交聚类最大相关性** | `max_correlation` | $\le \mathbf{0.65}$ | $\le \mathbf{0.60}$ | $\le \mathbf{0.80}$ |
| **聚类簇数范围** | `[min_clusters, max_clusters]` | **[12, 16]** 维 | **[10, 14]** 维 | **[100, 155]** 维 |
| **多重共线性上限** | `max_vif` | $\le \mathbf{10.0}$ | $\le \mathbf{8.0}$ (严禁高度共线性) | $\le \mathbf{10.0}$ |
| **强制时间特征模式** | `mandatory_feature_pattern` | `^(base_time_&#124;time_&#124;trading_minute_)` | `^(trading_minute_progress)` | `None` (接收全部 17 个强制特征) |

---

### 5. 条件锚点挽救机制与数学并集规则

1. **条件锚点机制 (`Conditional Anchors`)**:
   - 在多合约极端行情区制（如极端空头高波、极端多头低波）下，某些特征可能在全样本平均下的 RankIC 未达全域门禁，但在特定区制分箱（Target Regime Bin）内展现出显著预测力（LCB 90% 置信下限 $> 0.035$）且跨合约方差比 $\le 3.0$。
   - 管道支持通过条件锚点机制将这类关键稀缺特征（如 `trend_to_noise_96`, `trend_r2_48`, `log_return_vol_quantile_192`）挽救入 RL 决策流。

2. **数学并集 (Mathematical Union)**:
   最终生成供底层环境和模型加载的全集特征张量满足严格的数学并集定理：
   $$S_{\text{union}} = S_{\text{vae\_slope}} \cup S_{\text{vae\_vol}} \cup S_{\text{rl}}$$
   - 保证下游单一数据加载器只需单次加载、单次 Robust Scaling，即可通过特征切片索引快速分发给 Slope VAE、Volatility VAE 以及 RL 策略网络，彻底杜绝数据冗余与内存碎片。
