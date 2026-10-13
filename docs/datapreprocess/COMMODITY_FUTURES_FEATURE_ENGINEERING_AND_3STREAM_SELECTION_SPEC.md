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

源码：`data_preprocess/operator_futures/commodity/downscale.py`

| # | 特征名称 | 语义解释 | 精确数学公式 | 时间窗口 |
|---|---------|---------|-------------|---------|
| 1 | `open` | 桶内首笔成交价 | $P_{\text{first} \in I_t}$ | 单 Bar (10m) |
| 2 | `high` | 桶内最高成交价 | $\max_{i \in I_t} P_i$ | 单 Bar (10m) |
| 3 | `low` | 桶内最低成交价 | $\min_{i \in I_t} P_i$ | 单 Bar (10m) |
| 4 | `close` | 桶内末笔成交价 | $P_{\text{last} \in I_t}$ | 单 Bar (10m) |
| 5 | `volume` | 桶内累计成交量 | $\sum_{i \in I_t} V_i$ | 单 Bar (10m) |
| 6 | `tradeval` | 桶内累计成交金额 | $\sum_{i \in I_t} P_i \cdot V_i \cdot \text{Unit}$ | 单 Bar (10m) |
| 7 | `open_interest` | 期末持仓总量 | $\text{Last}(OI_i \in I_t)$ | 单 Bar (10m) |
| 8 | `vwap` | 成交量加权平均价 | $\frac{\text{tradeval}}{\text{volume} \cdot \text{Unit}}$（若 volume=0 回退为 close） | 单 Bar (10m) |
| 9 | `awap` / `twap` | 时间算术平均价 | $\frac{1}{N} \sum_{i \in I_t} P_i$ | 单 Bar (10m) |
| 10 | `ntrade_estimated` | 估计成交笔数 | 桶内有成交秒数或有效价格跳动次数 | 单 Bar (10m) |
| 11 | `ntrade_up_estimated` | 主买驱动估计笔数 | $\sum_{i \in I_t} \mathbb{I}\{\Delta P_i > 0\}$ | 单 Bar (10m) |
| 12 | `ntrade_down_estimated` | 主卖驱动估计笔数 | $\sum_{i \in I_t} \mathbb{I}\{\Delta P_i < 0\}$ | 单 Bar (10m) |
| 13 | `ntrade_flat_estimated` | 平价变动估计笔数 | $\sum_{i \in I_t} \mathbb{I}\{\Delta P_i = 0\}$ | 单 Bar (10m) |
| 14-23 | `bid1_price` ~ `bid5_price` | 1~5 档买价 | $BP_l, \quad l \in \{1, \dots, 5\}$ | 快照 (Bar 末) |
| 24-33 | `bid1_size` ~ `bid5_size` | 1~5 档买量 | $BS_l, \quad l \in \{1, \dots, 5\}$ | 快照 (Bar 末) |
| 34-43 | `ask1_price` ~ `ask5_price` | 1~5 档卖价 | $AP_l, \quad l \in \{1, \dots, 5\}$ | 快照 (Bar 末) |
| 44-53 | `ask1_size` ~ `ask5_size` | 1~5 档卖量 | $AS_l, \quad l \in \{1, \dots, 5\}$ | 快照 (Bar 末) |

---

### 2. 涨跌停板与非对称深度限制特征 (Price Limit & Asymmetric Liquidity)

源码：`data_preprocess/operator_futures/commodity/downscale.py`

| # | 特征名称 | 语义解释 | 精确数学公式 | 时间窗口 |
|---|---------|---------|-------------|---------|
| 1 | `limit_up_single_sided_ratio` | 涨停单边挂单比例 | $\frac{1}{T} \sum_{t=1}^T \mathbb{I}\{AP_1 \text{ 为空} \land P_t = P_{\text{limit\_up}} \land BP_1 > 0 \land BS_1 > 0\}$ | 单 Bar 内部采样 |
| 2 | `limit_down_single_sided_ratio`| 跌停单边挂单比例 | $\frac{1}{T} \sum_{t=1}^T \mathbb{I}\{BP_1 \text{ 为空} \land P_t = P_{\text{limit\_down}} \land AP_1 > 0 \land AS_1 > 0\}$ | 单 Bar 内部采样 |
| 3 | `limit_up_ask_depth_ratio_5` | 卖盘深度触及涨停比 | $\frac{1}{5} \sum_{l=1}^5 \mathbb{I}\{AP_l \ge P_{\text{limit\_up}} \lor AP_l \le 0 \lor AS_l = 0\}$ | 5 档截面平均 |
| 4 | `limit_down_bid_depth_ratio_5` | 买盘深度触及跌停比 | $\frac{1}{5} \sum_{l=1}^5 \mathbb{I}\{BP_l \le P_{\text{limit\_down}} \lor BP_l \le 0 \lor BS_l = 0\}$ | 5 档截面平均 |
| 5 | `limit_depth_imbalance_ratio_5`| 涨跌停深度不平衡比 | $\text{limit\_up\_ask\_depth\_ratio\_5} - \text{limit\_down\_bid\_depth\_ratio\_5}$ | 5 档截面差值 |

---

### 3. 截面微观结构与订单簿不平衡特征 (Orderbook Microstructure & Imbalance)

源码：`data_preprocess/operator_futures/cross_section/base_feature_util.py`, `downscale.py`

| # | 特征名称 | 语义解释 | 精确数学公式 | 时间窗口 |
|---|---------|---------|-------------|---------|
| 1 | `midprice` | 盘口中间价 | $\frac{AP_1 + BP_1}{2}$ | 瞬时截面 |
| 2 | `wap_1` | 1 档订单簿加权均价 | $\frac{AS_1 \cdot BP_1 + BS_1 \cdot AP_1}{AS_1 + BS_1}$ | 瞬时截面 |
| 3 | `wap_2` | 2 档订单簿加权均价 | $\frac{AS_2 \cdot BP_2 + BS_2 \cdot AP_2}{AS_2 + BS_2}$ | 瞬时截面 |
| 4 | `wap_balance` | 1、2 档 WAP 价差 | $\text{wap\_1} - \text{wap\_2}$ | 瞬时截面 |
| 5 | `sell_wap` | 卖方全档位加权平均价 | $\sum_{l=1}^5 \frac{AS_l}{\sum AS} \cdot AP_l$ | 5 档截面 |
| 6 | `buy_wap` | 买方全档位加权平均价 | $\sum_{l=1}^5 \frac{BS_l}{\sum BS} \cdot BP_l$ | 5 档截面 |
| 7 | `buy_sell_wap_spread` | 全档位买卖 WAP 价差 | $\text{buy\_wap} - \text{sell\_wap}$ | 5 档截面 |
| 8 | `buy_spread_oe_max` | 买盘最深最浅价差 | $|BP_1 - BP_5|$ | 5 档截面 |
| 9 | `sell_spread_oe_max` | 卖盘最深最浅价差 | $|AP_1 - AP_5|$ | 5 档截面 |
| 10 | `imbalance_volume` / `imbalance_1` | 1 档挂单量失衡比率 | $\frac{BS_1 - AS_1}{BS_1 + AS_1 + \epsilon_1}$ | 瞬时截面 |
| 11 | `imbalance_3` | 1~3 档累计深度失衡比 | $\frac{\sum_{l=1}^3 BS_l - \sum_{l=1}^3 AS_l}{\sum_{l=1}^3 BS_l + \sum_{l=1}^3 AS_l + \epsilon_1}$ | 3 档截面 |
| 12 | `imbalance_5` | 1~5 档全深度失衡比 | $\frac{\sum_{l=1}^5 BS_l - \sum_{l=1}^5 AS_l}{\sum_{l=1}^5 BS_l + \sum_{l=1}^5 AS_l + \epsilon_1}$ | 5 档截面 |
| 13-17 | `ask_price_topk_size_{k}_increments` | Top-k 大挂单价差增量 | $|AP_{\text{top}_k} - AP_1|$ | 截面排序 (k=1..5) |
| 18-22 | `bid_price_topk_size_{k}_increments` | Top-k 大挂单价差增量 | $|BP_{\text{top}_k} - BP_1|$ | 截面排序 (k=1..5) |
| 23-27 | `ask_size_topk_size_{k}_share` | Top-k 档位挂单量占比 | $\frac{AS_{\text{top}_k}}{\sum_{l=1}^5 AS_l + \epsilon_1}$ | 截面排序 (k=1..5) |
| 28-32 | `bid_size_topk_size_{k}_share` | Top-k 档位挂单量占比 | $\frac{BS_{\text{top}_k}}{\sum_{l=1}^5 BS_l + \epsilon_1}$ | 截面排序 (k=1..5) |

---

### 4. 截面 K 线形态特征 (Cross-Section K-Line Morphology)

源码：`data_preprocess/operator_futures/cross_section/base_feature_util.py` (`process_k_line_feature`)

针对包含 $O, H, L, C$ 的列组（包括 trades 全量、buy 子集、sell 子集以及盘口 spread 等），生成标准化形态特征：

| # | 特征后缀 | 语义解释 | 精确数学公式 | 物理内涵 |
|---|---------|---------|-------------|---------|
| 1 | `klen` | K 线全振幅率 | $\frac{H - L}{O + \epsilon_1}$ | 衡量单 Bar 价格波动极差大小 |
| 2 | `kmid` | 实体相对收益率 | $\frac{C - O}{O + \epsilon_1}$ | 衡量单 Bar 实体方向与相对涨跌幅度 |
| 3 | `kmid2` | 实体占振幅比例 | $\frac{C - O}{H - L + \epsilon_1}$ | 衡量多空博弈净胜程度（范围 [-1, 1]） |
| 4 | `kup` | 上影线相对长度 | $\frac{H - \max(O, C)}{O + \epsilon_1}$ | 衡量上方空头反压抛压强度 |
| 5 | `kup2` | 上影线占振幅比 | $\frac{H - \max(O, C)}{H - L + \epsilon_1}$ | 衡量冲高回落的相对受阻程度 |
| 6 | `klow` | 下影线相对长度 | $\frac{\min(O, C) - L}{O + \epsilon_1}$ | 衡量下方多头下买托盘强度 |
| 7 | `klow2` | 下影线占振幅比 | $\frac{\min(O, C) - L}{H - L + \epsilon_1}$ | 衡量探底回升的下影支撑力度 |
| 8 | `kskew` | 上下影线不对称度 | $\frac{\text{kup} - \text{klow}}{H - L + \epsilon_1}$ | 刻画 K 线重心与上下影线的不对称偏度 |
| 9 | `kamt` | 实体金额密度 | $\frac{\text{tradeval} \cdot \text{kmid2}}{\text{klen} + \epsilon_1}$ | 结合成交金额的实体推升动能密度 |
| 10 | `kvld` | 波动换手密度 | $\frac{\text{volume}}{\text{klen} + \epsilon_1}$ | 衡量每单位价格振幅所消耗的成交量 |

---

### 5. 日历与交易时间上下文特征 (Base Time Features)

源码：`data_preprocess/operator_futures/commodity/base_time_feature.py`

| # | 特征名称 | 语义解释 | 精确数学公式 | 取值范围 |
|---|---------|---------|-------------|---------|
| 1 | `trading_minute_progress` | 交易小节时间进度 | $\frac{t_{\text{elapsed\_in\_session}}}{t_{\text{session\_duration}}}$ | $[0.0, 1.0]$ 严格线性递增 |
| 2 | `morning_session` | 早盘交易小节标识 | $\mathbb{I}\{\text{Session Start Hour} \in [6, 12)\}$ | $\{0.0, 1.0\}$ One-Hot |
| 3 | `afternoon_session` | 午盘交易小节标识 | $\mathbb{I}\{\text{Session Start Hour} \in [12, 18)\}$ | $\{0.0, 1.0\}$ One-Hot |
| 4 | `night_session` | 夜盘交易小节标识 | $\mathbb{I}\{\text{Session Start Hour} \ge 18 \lor < 6\}$ | $\{0.0, 1.0\}$ One-Hot |
| 5 | `is_opening_30m` | 开盘前 30 分钟窗口 | $\mathbb{I}\{t_{\text{elapsed\_in\_session}} \le 1800\,\text{s}\}$ | $\{0.0, 1.0\}$ |
| 6 | `is_closing_30m` | 收盘前 30 分钟窗口 | $\mathbb{I}\{(t_{\text{duration}} - t_{\text{elapsed}}) \le 1800\,\text{s}\}$ | $\{0.0, 1.0\}$ |
| 7 | `is_session_first_bar` | 小节前 2 根 Bar 脉冲 | $\mathbb{I}\{\text{Bar Index in Session} \in \{0, 1\}\}$ | $\{0.0, 1.0\}$ |
| 8 | `is_session_last_bar` | 小节最后 2 根 Bar 脉冲 | $\mathbb{I}\{\text{Bar Index from Session End} \in \{0, 1\}\}$ | $\{0.0, 1.0\}$ |
| 9 | `contract_month_sin` | 交割月份正弦编码 | $\sin\left(\frac{2\pi \cdot \text{DeliveryMonth}}{12}\right)$ | $[-1.0, 1.0]$ 周期编码 |
| 10 | `contract_month_cos` | 交割月份余弦编码 | $\cos\left(\frac{2\pi \cdot \text{DeliveryMonth}}{12}\right)$ | $[-1.0, 1.0]$ 周期编码 |
| 11 | `contract_life_remaining_ratio` | 合约生命剩余天数比 | $\frac{\max(\text{BusDays}(t, T_{\text{last}}), 1)}{\text{TotalTradingDays}}$ | $(0.0, 1.0]$ |
| 12 | `prev_day_contract_role_tier` | 前一交易日角色等级 | 1.0 (主力), 2.0 (次主力), 3.0 (其他合约) | 阶梯离散值 |

---

### 6. 跨期期限结构与合约轮动特征 (Cross-Month Term Structure Features)

源码：`data_preprocess/operator_futures/commodity/cross_month_feature.py`

针对主力（Main）、次主力（Sub）、当前合约（Current）以及近月交割序列（M1, M2, M3），提取无绝对价格违规的纯比率、价差和动态转移速度：

| # | 特征名称 | 语义解释 | 精确数学公式 | 时间窗口 |
|---|---------|---------|-------------|---------|
| 1-3 | `cm_contract_role_{main/sub/other}` | 合约角色 One-Hot | $\mathbb{I}\{\text{Role} == \text{target}\}$ | 瞬时 (每日更新) |
| 4 | `cm_current_main_log_price_ratio` | 当前合约相对主力对数价比 | $\ln\left(\frac{P_{\text{current}}}{P_{\text{main}}}\right)$ | 瞬时截面 |
| 5 | `cm_current_main_relative_price_spread` | 当前合约相对主力相对价差 | $\frac{2(P_{\text{current}} - P_{\text{main}})}{P_{\text{current}} + P_{\text{main}}}$ | 瞬时截面 |
| 6 | `cm_current_main_volume_share_current` | 当前合约占成交量份额 | $\frac{V_{\text{current}}}{V_{\text{current}} + V_{\text{main}} + \epsilon_1}$ | 瞬时截面 |
| 7 | `cm_current_main_open_interest_share_current` | 当前合约占持仓量份额 | $\frac{OI_{\text{current}}}{OI_{\text{current}} + OI_{\text{main}} + \epsilon_1}$ | 瞬时截面 |
| 8-11 | `cm_current_sub_*` | 当前合约与次主力配对特征 | 对应计算公式同上 | 瞬时截面 |
| 12-15 | `cm_main_sub_*` | 主力与次主力基差配对特征 | 对应计算公式同上 | 瞬时截面 |
| 16 | `cm_main_sub_log_price_spread_velocity_10m` | 主次主力价差变动速度 | $0.05 \cdot \tanh\left(\frac{\Delta_{10} \ln(P_{\text{main}} / P_{\text{sub}})}{0.02}\right)$ | 10 Bar (100m) 差分 |
| 17 | `cm_open_interest_shift_speed_10m` | 换月持仓量迁移速度 | $\Delta_{10}(\text{share}_{\text{sub}}) = \text{share}_{\text{sub}, t} - \text{share}_{\text{sub}, t-10}$ | 10 Bar (100m) 差分 |
| 18 | `cm_m1_m2_log_price_spread_velocity_10m` | M1 与 M2 价差变动速度 | $0.05 \cdot \tanh\left(\frac{\Delta_{10} \ln(P_{M1} / P_{M2})}{0.02}\right)$ | 10 Bar (100m) 差分 |
| 19 | `cm_m2_m3_log_price_spread_velocity_10m` | M2 与 M3 价差变动速度 | $0.05 \cdot \tanh\left(\frac{\Delta_{10} \ln(P_{M2} / P_{M3})}{0.02}\right)$ | 10 Bar (100m) 差分 |
| 20 | `cm_m1_m2_m3_butterfly_spread_velocity_10m`| 期限结构蝶式价差速度 | $0.05 \cdot \tanh\left(\frac{\Delta_{10} (\ln P_{M1} - 2\ln P_{M2} + \ln P_{M3})}{0.02}\right)$ | 10 Bar (100m) 差分 |
| 21-23 | `cm_{pair}_spread_rolling_zscore_{w}` | 相对价差滚动平稳 Z-Score | $\frac{\text{Spread}_t - \text{Mean}_w(\text{Spread})_t}{\text{Std}_w(\text{Spread})_t + \epsilon_3}$ | $w \in \{48, 192\}$ |

---

### 7. 混频跨日与跨周宏观状态特征 (Mixed-Frequency Daily & Weekly Features)

源码：`data_preprocess/operator_futures/commodity/daily_mixed_frequency_feature.py`, `weekly_mixed_frequency_feature.py`

混频特征捕捉上一交易日（Prev Day）与上一自然周（Prev Week）以及滚动多日/多周的高级别宏观锚点：

| # | 特征名称模板 | 语义解释 | 精确数学公式 | 滚动窗口集 |
|---|-------------|---------|-------------|-----------|
| 1 | `prev_{d}_day_return` | 历史前 d 日累计收益率 | $\frac{C_{\text{prev}} - C_{\text{prev}-d}}{C_{\text{prev}-d}}$ | $d \in \{1, 2, 5, 10, 15, 30\}$ 日 |
| 2 | `prev_{d}_day_range_pct` | 历史前 d 日振幅比例 | $\frac{\max(H) - \min(L)}{C_{\text{prev}-d}}$ | $d \in \{1, 2, 5, 10, 15, 30\}$ 日 |
| 3 | `prev_{d}_day_body_pct` | 历史前 d 日实体比例 | $\frac{|C - O|}{C_{\text{prev}-d}}$ | $d \in \{1, 2, 5, 10, 15, 30\}$ 日 |
| 4 | `prev_day_close_position` | 前日收盘在振幅中相对位置 | $\frac{C - L}{H - L + \epsilon_1}$ | 前 1 交易日 |
| 5 | `prev_day_vwap_deviation_pct`| 前日 VWAP 与收盘偏离度 | $\frac{\text{VWAP}_{\text{day}} - C}{C}$ | 前 1 交易日 |
| 6 | `prev_{d}_day_trade_imbalance` | 历史 d 日主动买卖失衡率 | $\frac{V_{\text{up}} - V_{\text{down}}}{V_{\text{up}} + V_{\text{down}} + \epsilon_1}$ | $d \in \{1, 2, 5, 10, 15, 30\}$ 日 |
| 7 | `prev_{d}_day_open_interest_change` | 历史 d 日持仓变动率 | $\frac{OI_t - OI_{t-d}}{OI_{t-d} + \epsilon_1}$ | $d \in \{1, 2, 5, 10, 15, 30\}$ 日 |
| 8 | `prev_{d}_day_turnover_rate` | 历史 d 日年化换手率 | $\frac{\sum V}{OI + \epsilon_1}$ | $d \in \{1, 2, 5, 10, 15, 30\}$ 日 |
| 9-16 | `prev_{w}_week_*` | 历史前 w 周对应指标 | 包含 return, range, body, vwap/twap deviation, imbalance 等 | $w \in \{1, 2, 4, 6\}$ 自然周 |
| 17 | `prev_*_quantile_rank` | 宏观状态滚动历史分位数排名 | $\text{RankPct}_{192}(\text{Feature}) = \frac{\text{Rank}(\text{Feature}_t \text{ in } W=192)}{192}$ | 滚动 192 Bar (约 4 天) |

---

### 8. 单列价格与成交量滚动时序特征 (Time Rolling Price & Volume Features)

源码：`data_preprocess/operator_futures/time_operator/multi_processing_util.py`

对输入的价格序列（`bid1~5_price`, `ask1~5_price`, `wap_1`, `wap_2`, `buy_wap`, `sell_wap`, `midprice`, `mark_price` 等）和挂单量序列（`bid1~5_size`, `ask1~5_size`, `buy_volume_oe`, `sell_volume_oe` 等），在时间窗口集合 $W = \{1, 2, 6, 12, 16, 24, 48, 96, 192\}$ 上生成标准时序特征：

| # | 特征名称后缀 | 语义解释 | 精确数学公式 | 时间窗口 $w$ |
|---|-------------|---------|-------------|-------------|
| 1 | `_log_return_{w}` | 对数收益率 (放大千倍) | $1000 \cdot \ln\left(\frac{X_t}{X_{t-w} + \epsilon_2}\right)$ | $w \in \{1, 2, 6, 12, 16, 24, 48\}$ |
| 2 | `_trend_{w}` | 均值偏离标准化趋势 (Z-Trend) | $\frac{X_t - \text{Mean}_w(X)_t}{\text{Std}_w(X)_t + \epsilon_2}$ | $w \in \{2, 6, 12, 16, 24, 48, 96, 192\}$ |
| 3 | `roc_{w}` | 变动速率 (Price Ratio) | $\frac{C_{t-w}}{C_t + \epsilon_2}$ | $w \in W$ |
| 4 | `ma_{w}` | 滚动均值与现价比率 | $\frac{\text{Mean}_w(C)_t}{C_t + \epsilon_2}$ | $w \in W$ |
| 5 | `std_{w}` | 滚动无偏标准差与现价比 | $\frac{\text{Std}_w(C)_t}{C_t + \epsilon_2}$ | $w \in W$ |
| 6 | `bollinger_bandwidth_{w}` | 布林带相对带宽对数 | $\ln\left(\frac{4 \cdot \text{Std}_w(C)_t}{\text{Mean}_w(C)_t + \epsilon_2} + 10^{-6}\right)$ | $w \in W$ |
| 7 | `bollinger_upper_{w}` | 布林带上轨与现价比 | $\frac{\text{Mean}_w(C)_t + 2 \cdot \text{Std}_w(C)_t}{C_t + \epsilon_2}$ | $w \in W$ |
| 8 | `bollinger_lower_{w}` | 布林带下轨与现价比 | $\frac{\text{Mean}_w(C)_t - 2 \cdot \text{Std}_w(C)_t}{C_t + \epsilon_2}$ | $w \in W$ |
| 9-13 | `pivot_{pp/r1/s1/r2/s2}_{w}` | 经典枢轴点支撑阻力系统与现价比 | $PP = \frac{\max_w(H) + \min_w(L) + C}{3}, R_1 = 2PP - L, S_1 = 2PP - H$ 等除以 $C$ | $w \in W$ |
| 14 | `beta_{w}` | 线性动量斜率比率 | $\frac{C_t - C_{t-w}}{w \cdot (C_t + \epsilon_2)}$ | $w \in W$ |
| 15-16 | `max_{w}`, `min_{w}` | 窗口内极值与现价比率 | $\frac{\max_w(C)}{C_t + \epsilon_2}, \quad \frac{\min_w(C)}{C_t + \epsilon_2}$ | $w \in W$ |
| 17-18 | `qtlu_{w}`, `qtld_{w}` | 80% / 20% 分位数与现价比率 | $\frac{\text{Quantile}_{0.8, w}(C)}{C_t + \epsilon_2}, \quad \frac{\text{Quantile}_{0.2, w}(C)}{C_t + \epsilon_2}$ | $w \in W$ |
| 19 | `rank_{w}` | 现价在窗口内的百分比排名 | $\frac{\text{Rank}(C_t \text{ in } W_t)}{w}$ | $w \in W$ |
| 20-22 | `imax_{w}`, `imin_{w}`, `imxd_{w}` | 极值发生位置与位置差 | $\frac{\text{argmax}(H)}{w}, \frac{\text{argmin}(L)}{w}, \text{imax} - \text{imin}$ | $w \in W$ |
| 23 | `rsv_{w}` | 未成熟随机值 (Stochastic RSV) | $\frac{C_t - \min_w(L)}{\max_w(H) - \min_w(L) + \epsilon_2}$ | $w \in W$ |
| 24-26 | `cntp_{w}`, `cntn_{w}`, `cntd_{w}` | 上涨/下跌 Bar 计数占比与差比 | $\frac{N_{\text{up}}}{w}, \frac{N_{\text{down}}}{w}, \frac{N_{\text{up}} - N_{\text{down}}}{N_{\text{up}} + N_{\text{down}} + 10^{-6}}$ | $w \in W$ |
| 27-29 | `sump_{w}`, `sumn_{w}`, `sumd_{w}` | 正负对数收益率累计绝对值占比 | $\frac{\sum |r^+|}{\sum |r|}, \frac{\sum |r^-|}{\sum |r|}, \frac{\sum |r^+| - \sum |r^-|}{\sum |r|}$ | $w \in W$ |
| 30 | `vma_{w}` | 成交量滚动均值与现量比 | $\frac{\text{Mean}_w(V)_t}{V_t + \epsilon_2}$ | $w \in W$ |
| 31 | `vstd_{w}` | 成交量相对波动率 | $\text{clip}\left(\frac{\text{Std}_w(V)_t}{\max(V_t, 1.0)}, 0.0, 10.0\right)$ | $w \in W$ |
| 32-34 | `vsump_{w}`, `vsumn_{w}`, `vsumd_{w}` | 成交量变动增减累计占比 | 量差分正增量、负增量的相对累计比例 | $w \in W$ |
| 35-42 | `*_std_norm` | 标准差归一化变体 | $\frac{C_t - C_{t-w}}{\text{Std}_w(C)}, \frac{C_t - \text{Mean}_w(C)}{\text{Std}_w(C)}, \frac{C_t - \text{Min}_w(C)}{\text{Std}_w(C)}$ 等 | $w \in W$ |

---

### 9. 物理风险与流动性状态特征 (Physical Risk & Liquidity State Features)

源码：`data_preprocess/operator_futures/time_operator/multi_processing_util.py` (`get_risk_and_liquidity_state_features`)

| # | 特征名称模板 | 语义解释 | 精确数学公式 | 时间窗口 $w$ |
|---|-------------|---------|-------------|-------------|
| 1 | `atr_pct_{w}` | 真实波幅百分比 | $\frac{\text{Mean}_w(\text{TR})_t}{C_t} \cdot 100\%, \quad \text{TR} = \max(H - L, |H - C_{-1}|, |L - C_{-1}|)$ | $w \in \{1, 2, 6, 12, 16, 24, 48, 96\}$ |
| 2 | `historical_volatility_{w}`| 历史年化波动率 | $\text{Std}_w(r)_t \cdot \sqrt{\text{BarsPerDay}}$ | $w \in \{1, 2, 6, 12, 16, 24, 48, 96\}$ |
| 3 | `rolling_volatility_{w}` | 指数加权移动波动率 | $\text{EWM\_Std}_w(r)_t$ | $w \in \{1, 2, 6, 12, 16, 24, 48, 96\}$ |
| 4 | `parkinson_volatility_{w}` | 帕金森极值方差波动率 | $\sqrt{\frac{1}{4 \ln 2 \cdot w} \sum_{k=0}^{w-1} \ln\left(\frac{H_{t-k}}{L_{t-k}}\right)^2}$ | $w \in \{1, 2, 6, 12, 16, 24, 48, 96\}$ |
| 5 | `garman_klass_volatility_{w}`| 加曼-克拉斯开高低收波动率 | $\sqrt{\frac{1}{w} \sum_{k=0}^{w-1} \left[0.5 \ln\left(\frac{H}{L}\right)^2 - (2\ln 2 - 1) \ln\left(\frac{C}{O}\right)^2\right]}$ | $w \in \{1, 2, 6, 12, 16, 24, 48, 96\}$ |
| 6 | `realized_volatility_{w}` | 滚动已实现波动率 | $\sqrt{\sum_{k=0}^{w-1} r_{t-k}^2}$ | $w \in \{1, 2, 6, 12, 16, 24, 48, 96\}$ |
| 7 | `relative_volume_{w}` | 相对成交量倍数 | $\frac{V_t}{\text{Mean}_w(V)_t + \epsilon_2}$ | $w \in \{1, 2, 6, 12, 16, 24, 48, 96\}$ |
| 8 | `relative_amount_{w}` | 相对成交金额倍数 | $\frac{\text{tradeval}_t}{\text{Mean}_w(\text{tradeval})_t + \epsilon_2}$ | $w \in \{1, 2, 6, 12, 16, 24, 48, 96\}$ |
| 9 | `relative_open_interest_{w}`| 相对持仓量倍数 | $\frac{OI_t}{\text{Mean}_w(OI)_t + \epsilon_2}$ | $w \in \{1, 2, 6, 12, 16, 24, 48, 96\}$ |
| 10 | `open_interest_change_ratio_{w}`| 跨窗口持仓量变动率 | $\frac{OI_t - OI_{t-w}}{OI_{t-w} + \epsilon_2}$ | $w \in \{1, 2, 6, 12, 16, 24, 48, 96\}$ |

---

### 10. 动量、振荡器与高阶增强特征 (Momentum, Oscillators & Enhanced Features)

源码：`data_preprocess/operator_futures/time_operator/time_operator_util.py` (`process_enhanced_state_features`)

| # | 特征名称 | 语义解释 | 精确数学公式 | 参数与窗口 |
|---|---------|---------|-------------|-----------|
| 1 | `price_velocity_10m` | 价格短期均线速度 | $\frac{\text{EMA}_{10}(C) - \text{EMA}_{20}(C)}{C}$ | span=(10, 20) |
| 2 | `price_acceleration_10m_norm` | 标准化价格加速度 | $\frac{\Delta_1(\text{price\_velocity\_10m})}{\text{RollingStd}_{10}(r) + \epsilon_3}$ | span=10, std=10 |
| 3-5 | `ema_divergence_{s1}_{s2}` | 多尺度 EMA 发散度 | $\frac{\text{EMA}_{s1}(C) - \text{EMA}_{s2}(C)}{\text{EMA}_{s2}(C)}$ | (20, 60), (60, 120), (20, 120) |
| 6 | `macd_histogram_norm` | 归一化 MACD 柱状图 | $\text{DIF} - \text{DEA}, \quad \text{DIF} = \frac{\text{EMA}_{12}(C) - \text{EMA}_{26}(C)}{C}, \text{DEA} = \text{EMA}_9(\text{DIF})$ | (12, 26, 9) |
| 7 | `rsi_14_norm` | 归一化相对强弱指标 | $\frac{\text{RSI}_{14} - 50}{50}, \quad \text{RSI}_{14} = 100 \cdot \frac{\text{EWM}(\text{Gain})}{\text{EWM}(\text{Gain}) + \text{EWM}(\text{Loss})}$ | Wilder 14 周期 |
| 8-10 | `plus_di_14`, `minus_di_14`, `adx_14`| 经典 DMI/ADX 趋势强度 | $+DI_{14} = 100 \frac{\text{EWM}(+DM)}{\text{ATR}}, -DI_{14} = 100 \frac{\text{EWM}(-DM)}{\text{ATR}}, ADX = \text{EWM}\left(100 \frac{|+DI - -DI|}{+DI + -DI}\right)$ | Wilder 14 周期 |
| 11-12 | `stoch_k_14_norm`, `stoch_d_14_norm` | 归一化随机振荡器 KD | $\frac{K - 50}{50}, \frac{D - 50}{50}, \quad K = \text{EWM}_{3}(\text{RSV}_{14}), D = \text{EWM}_{3}(K)$ | 周期 (14, 3) |
| 13 | `cci_14_norm` | 归一化顺势指标 | $\text{clip}\left(\frac{TP - \text{Mean}_{14}(TP)}{0.015 \cdot \text{MeanDev}_{14}(TP)} \cdot \frac{1}{100}, -3.0, 3.0\right), TP = \frac{H+L+C}{3}$ | 周期 14 |
| 14-15 | `price_to_vwap_zscore_{w}` | 现价相对 VWAP 偏离 Z-Score | $\frac{\text{Dev}_t - \text{Mean}_w(\text{Dev})}{\text{Std}_w(\text{Dev}) + \epsilon_3}, \quad \text{Dev} = \frac{C - \text{VWAP}}{C}$ | $w \in \{48, 96\}$ |
| 16-17 | `vwap_slope_{w}` | VWAP 宏观斜率 | $\frac{\text{VWAP}_t - \text{VWAP}_{t-w}}{\text{VWAP}_{t-w} \cdot w}$ | $w \in \{96, 192\}$ |
| 18 | `ofi_norm_zscore_48` | 订单流不平衡 (OFI) Z-Score | $\frac{\text{OFI}_t - \text{Mean}_{48}(\text{OFI})}{\text{Std}_{48}(\text{OFI}) + \epsilon_3}$ | $w = 48$ |
| 19 | `ofi_persistence_20m` | OFI 指数持续性平滑 | $\text{EWM}_{20}(\text{OFI})$ | span = 20 |
| 20-21 | `cvd_slope_{w}` | 累积成交量增量 (CVD) 斜率 | $\frac{\sum_{k=0}^{w-1} (V \cdot \text{Direction})}{\sum_{k=0}^{w-1} V}$ | $w \in \{96, 192\}$ |
| 22 | `realized_vol_zscore_192` | 波动率区制偏离 Z-Score | $\frac{\text{RV}_{12} - \text{Mean}_{192}(\text{RV}_{12})}{\text{Std}_{192}(\text{RV}_{12}) + \epsilon_3}$ | 滚动 192 Bar (约 4 天) |
| 23 | `realized_vol_term_ratio_12_48` | 波动率期限结构期限比 | $\frac{\text{Realized\_Vol}_{12}}{\text{Realized\_Vol}_{48} + \epsilon_3}$ | 短期 12 对中期 48 比率 |
| 24-26 | `macro_trade_imbalance_continuous_{w}` | 宏观平滑主动买卖失衡率 | $\frac{\sum_w N_{\text{up}} - \sum_w N_{\text{down}}}{\sum_w N_{\text{up}} + \sum_w N_{\text{down}} + 10w}$ | $w \in \{48, 240\}$ |
| 27-29 | `macro_oi_change_rate_{w}` | 宏观持仓量迁移变动率 | $\text{clip}\left(\frac{OI_t - OI_{t-w}}{\text{Mean}_w(OI) + 1.0}, -0.5, 0.5\right)$ | $w \in \{48, 240\}$ |
| 30-32 | `macro_turnover_rate_log_{w}` | 宏观对数累计换手率 | $\ln\left(1.0 + \frac{\sum_w V_t}{OI_t + 1.0}\right)$ | $w \in \{48, 240\}$ |

---

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
| **强制时间特征模式** | `mandatory_feature_pattern` | `^(base_time_\|time_\|trading_minute_)` | `^(trading_minute_progress)` | `None` (接收全部 17 个强制特征) |

---

### 5. 条件锚点挽救机制与数学并集规则

1. **条件锚点机制 (`Conditional Anchors`)**:
   - 在多合约极端行情区制（如极端空头高波、极端多头低波）下，某些特征可能在全样本平均下的 RankIC 未达全域门禁，但在特定区制分箱（Target Regime Bin）内展现出显著预测力（LCB 90% 置信下限 $> 0.035$）且跨合约方差比 $\le 3.0$。
   - 管道支持通过条件锚点机制将这类关键稀缺特征（如 `trend_to_noise_96`, `trend_r2_48`, `log_return_vol_quantile_192`）挽救入 RL 决策流。

2. **数学并集 (Mathematical Union)**:
   最终生成供底层环境和模型加载的全集特征张量满足严格的数学并集定理：
   $$S_{\text{union}} = S_{\text{vae\_slope}} \cup S_{\text{vae\_vol}} \cup S_{\text{rl}}$$
   - 保证下游单一数据加载器只需单次加载、单次 Robust Scaling，即可通过特征切片索引快速分发给 Slope VAE、Volatility VAE 以及 RL 策略网络，彻底杜绝数据冗余与内存碎片。
