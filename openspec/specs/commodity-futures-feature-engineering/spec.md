# Commodity Futures Feature Engineering Specification

## Purpose
本文档反向提炼并定义当前商品期货（如燃料油 `fu` 等）特征工程模块的完整技术规范与特征字典。涵盖数据预处理、五档行情下采样、截面特征派生、Order Flow Imbalance (OFI)、微观结构压力、时间维度滚动统计、风险与流动性状态特征以及强化学习交易过程特征的计算逻辑、工程作用和精确数学公式。

---

## 1. Problem Statement & Architecture

高频商品期货交易与强化学习模型（如 FineFT）依赖丰富且高质量的微观结构与统计特征。商品期货数据具备五档盘口快照（Depth-5）、高频秒级成交/挂单、非连续交易 Session（含夜盘）等特点。

为了支撑特征工程的全流程计算与模型训练，系统划分了以下子模块 pipeline：
1. **`commodity/downscale.py` & `downscale_single_day.py`**: 负责五档快照及秒级 Trade/Quote 数据降采样，派生基础 OHLCV、VWAP/TWAP、多档盘口压力、五档 OFI 行窗口特征及微观结构队列压力特征。
2. **`cross_section/base_feature_util.py` & `create_feature.py`**: 负责截面 K 线几何特征（`klen`, `kmid`, `kup`, `ksft` 等）、归一化 Quote 比率及 Orderbook 深度特征。
3. **`commodity/base_time_feature.py`**: 派生 session 内时间进度、Session One-hot 编码、交割月份三角函数编码及合约剩余生命周期比例。
4. **`time_operator/multi_processing_util.py` & `time_operator_util.py`**: 派生动量、波动率、分位数、相关性等滚动时间序列特征，以及滚动风险特征 (ATR, HV, PV, GKV, RV) 和流动性特征 (RelVol, RelAmt, RelOI, OI_Change)。
5. **FineFT RL Trading Process**: 在强化学习环境中生成实时交易状态特征（`position_exposure`, `single_holding_return_rate`, `single_holding_max_drawdown`, `current_holding_duration_norm`）。

---

## 2. User Stories

1. **As a Quantitative Researcher**, I want a formal specification of all commodity futures features, so that I can understand feature semantics, verify statistical properties, and maintain exact math formulas.
2. **As an ML Engineer**, I want clear contracts on candidate vs protected state features (such as `BASE_TIME_FEATURE`), so that scaling, filtering, and model inputs remain consistent and deterministic.
3. **As a System Developer**, I want strict data validation and fail-fast assertions on missing, non-finite, or bad quote/trade inputs, so that downstream models are protected from corrupt features.

---

## 3. Comprehensive Feature Dictionary & Mathematical Formulas

### 3.1 基础行情与参考价下采样特征 (Base & Reference Price Features)

#### 特征列表与含义
- `mark_price` / `index_price`: 环境参考标记价与指数价，优先采用秒级有效 `LastPrice`；若缺失或异常则退回最优买卖中间价 `mid`。
- `vwap`: 目标窗口内成交量加权平均价，作用在于提供准确的真实成交均价 benchmark。
- `twap` / `awap`: 时间加权与算术平均价。
- `open_interest`: 窗口末尾的真实持仓量。
- `buy_estimated` / `sell_estimated`: 结合 Tick Rule 的估计成交方向。

#### 数学公式
- **合约价格修正 (Contract Price Scaling)**:
  $$\text{second\_avg\_price}_t = \frac{\Delta \text{Turnover}_t}{\Delta \text{Volume}_t \times \text{contract\_unit}}$$
- **成交量加权均价 (VWAP)**:
  $$\text{VWAP} = \frac{\sum \Delta \text{Turnover}_t}{\left(\sum \Delta \text{Volume}_t\right) \times \text{contract\_unit}}$$
- **参考价 (Mark/Index Price)**:
  $$\text{MarkPrice} = \begin{cases} \text{LastPrice}, & \text{if } \text{LastPrice} \in (\text{LowerLimit}, \text{UpperLimit}) \text{ and } \text{LastPrice} > 0 \\ \frac{\text{BidPrice1} + \text{AskPrice1}}{2}, & \text{otherwise} \end{cases}$$
- **Tick Rule 估计成交方向**:
  $$\Delta p_{\text{avg}, t} = \text{second\_avg\_price}_t - \text{second\_avg\_price}_{t-1}$$
  $$\text{Direction} = \begin{cases} \text{buy\_estimated}, & \text{if } \Delta p_{\text{avg}, t} > 0 \\ \text{sell\_estimated}, & \text{if } \Delta p_{\text{avg}, t} < 0 \\ \text{flat}, & \text{if } \Delta p_{\text{avg}, t} = 0 \end{cases}$$

---

### 3.2 盘口 Quote 与多档盘口压力特征 (Quote & Multi-Depth Imbalance Features)

#### 特征列表与含义
- `spread`: 一档买卖价差 ($\text{AskPrice1} - \text{BidPrice1}$)，衡量盘口流动性成本。
- `mid`: 盘口中间价 ($\frac{\text{AskPrice1} + \text{BidPrice1}}{2}$)。
- `imbalance_1`, `imbalance_3`, `imbalance_5`: 1档、3档及5档盘口挂单量不平衡度，作用在于度量买卖双向深度的即时相对压力。
- 窗口聚合算子: 对 `spread`, `mid`, `imbalance_*`, `bid`, `ask`, `bidsize`, `asksize` 生成 `open`, `high`, `low`, `close`, `twap`, `awap`, `std` 统计列。

#### 数学公式
- **多档盘口挂单压力不平衡度 ($k \in \{1, 3, 5\}$)**:
  $$\text{imbalance}_k = \frac{\sum_{i=1}^k \text{BidVolume}_i - \sum_{i=1}^k \text{AskVolume}_i}{\sum_{i=1}^k \text{BidVolume}_i + \sum_{i=1}^k \text{AskVolume}_i}$$
  *若分母 $\sum (\text{BidVolume}_i + \text{AskVolume}_i) = 0$，则定义 $\text{imbalance}_k = 0.0$。*

---

### 3.3 五档 Order Flow Imbalance (OFI) 行窗口特征

#### 特征列表与含义
- `ofi_bid1` ~ `ofi_bid5`: 1到5档买盘订单流不平衡量。
- `ofi_ask1` ~ `ofi_ask5`: 1到5档卖盘订单流不平衡量。
- `ofi_bid`, `ofi_ask`, `ofi`: 买侧、卖侧及全盘口 OFI 汇总。
- `ofi_bid_norm`, `ofi_ask_norm`, `ofi_norm`: 深度挂单量归一化后的 OFI，消除绝对挂单规模影响，反映订单流净驱动力。

#### 数学公式
- **单步买盘 $i$ 档 OFI ($i \in \{1 \dots 5\}$)**:
  $$\text{OFI}_{bid, i, t} = \begin{cases} +\text{BidVolume}_{i, t}, & \text{if } \text{BidPrice}_{i, t} > \text{BidPrice}_{i, t-1} \\ \text{BidVolume}_{i, t} - \text{BidVolume}_{i, t-1}, & \text{if } \text{BidPrice}_{i, t} = \text{BidPrice}_{i, t-1} \\ -\text{BidVolume}_{i, t-1}, & \text{if } \text{BidPrice}_{i, t} < \text{BidPrice}_{i, t-1} \end{cases}$$

- **单步卖盘 $i$ 档 OFI ($i \in \{1 \dots 5\}$)**:
  $$\text{OFI}_{ask, i, t} = \begin{cases} -\text{AskVolume}_{i, t}, & \text{if } \text{AskPrice}_{i, t} < \text{AskPrice}_{i, t-1} \\ -(\text{AskVolume}_{i, t} - \text{AskVolume}_{i, t-1}), & \text{if } \text{AskPrice}_{i, t} = \text{AskPrice}_{i, t-1} \\ +\text{AskVolume}_{i, t-1}, & \text{if } \text{AskPrice}_{i, t} > \text{AskPrice}_{i, t-1} \end{cases}$$

- **行窗口 ($W$ 行快照) 归一化 OFI**:
  $$\text{ofi\_norm} = \frac{\sum_{t \in W} \left( \sum_{i=1}^5 \text{OFI}_{bid, i, t} + \sum_{i=1}^5 \text{OFI}_{ask, i, t} \right)}{\sum_{t \in W} \sum_{i=1}^5 \left( \text{BidVolume}_{i, t} + \text{AskVolume}_{i, t} \right)}$$

---

### 3.4 盘口微观结构与队列压力特征 (Quote Microstructure & Queue Pressure)

#### 特征列表与含义
- `microprice`: 盘口挂单量加权微观价格。
- `mean_microprice_pressure`: 微观价格偏离中间价相对价差的比率均值，度量挂单倾斜诱导的价格短期偏向。
- `mean_relative_spread`: 相对价差 ($\text{spread} / \text{mid}$) 的均值。
- `spread_widen_count` / `spread_narrow_count` / `spread_flat_count` / `spread_widen_ratio`: 价差扩大/缩小/持平计数与扩大比例。
- `bid_refill_count` / `bid_deplete_count` / `ask_refill_count` / `ask_deplete_count`: 一档挂单补充（撤单/成交消耗）事件计数。
- `queue_refill_imbalance`: 队列补充/消耗的不平衡度。
- `bid_side_empty_ratio` / `ask_side_empty_ratio`: 单侧盘口为空的比率。
- `limit_up_single_sided_ratio` / `limit_down_single_sided_ratio`: 涨跌停单边盘口比例。

#### 数学公式
- **微观价格 (Microprice)**:
  $$\text{Microprice} = \frac{\text{AskPrice1} \cdot \text{BidVolume1} + \text{BidPrice1} \cdot \text{AskVolume1}}{\text{BidVolume1} + \text{AskVolume1}}$$
- **微观价格压力 (Microprice Pressure)**:
  $$\text{Microprice\_Pressure} = \frac{\text{Microprice} - \text{Mid}}{\text{Spread}}$$
- **队列补充不平衡度 (Queue Refill Imbalance)**:
  $$\text{Queue\_Refill\_Imbalance} = \frac{(\text{bid\_refill\_count} + \text{ask\_deplete\_count}) - (\text{bid\_deplete\_count} + \text{ask\_refill\_count})}{\text{total\_queue\_events}}$$

---

### 3.5 截面 K 线几何特征与盘口深度特征 (Cross-Section KLine Geometry & Snapshot)

#### 特征列表与含义
- `klen`: K 线实体与影线总长度相对 Open 的比例 ($\frac{\text{High} - \text{Low}}{\text{Open}}$)。
- `kmid` / `kmid2`: K 线实体长度相对 Open 及相对 High-Low 总波幅的比例。
- `kup` / `kup2`: 上影线长度相对 Open 及总波幅的比例。
- `klow` / `klow2`: 下影线长度相对 Open 及总波幅的比例。
- `ksft` / `ksft2`: 收盘价偏置指标，表达收盘价在 High-Low 区间内的相对位置。
- `kotwap`, `kctwap`, `koawap`, `kcawap`, `kovwap`, `kcvwap`: Open/Close 偏离 TWAP/AWAP/VWAP 的相对比例。
- `wap_1`, `wap_2`, `wap_balance`: 1档及2档加权价格与平衡度。
- `buy_wap`, `sell_wap`, `buy_sell_wap_spread`: 整体买侧/卖侧加权价格及买卖 WAP 价差。

#### 数学公式
- **实体相对波幅 (KMID2)**:
  $$\text{KMID2} = \frac{\text{Close} - \text{Open}}{(\text{High} - \text{Low}) + \epsilon}$$
- **上影线相对波幅 (KUP2)**:
  $$\text{KUP2} = \frac{\text{High} - \max(\text{Open}, \text{Close})}{(\text{High} - \text{Low}) + \epsilon}$$
- **收盘偏置 (KSFT2)**:
  $$\text{KSFT2} = \frac{2 \cdot \text{Close} - \text{High} - \text{Low}}{(\text{High} - \text{Low}) + \epsilon}$$
- **第一档加权价格 (WAP_1)**:
  $$\text{WAP}_1 = \frac{\text{AskVolume1} \cdot \text{BidPrice1} + \text{BidVolume1} \cdot \text{AskPrice1}}{\text{BidVolume1} + \text{AskVolume1}}$$

---

### 3.6 基础时间编码状态特征 (BASE_TIME_FEATURE)

强制保留作为状态特征，跳过 Feature Selection 筛选和 Robust Scaler 缩放。

#### 特征列表与含义
1. `trading_minute_progress`: 当前时间戳在所属 Trading Session 中的归一化时间进度 $[0, 1]$。
2. `morning_session`: 早盘 Session One-hot 标记。
3. `afternoon_session`: 午盘 Session One-hot 标记。
4. `night_session`: 夜盘 Session One-hot 标记。
5. `is_opening_30m`: 当前是否处于 Session 开盘前 30 分钟。
6. `is_closing_30m`: 当前是否处于 Session 收盘前 30 分钟。
7. `contract_month_sin`: 合约交割月份的正弦周期编码。
8. `contract_month_cos`: 合约交割月份的余弦周期编码。
9. `contract_life_remaining_ratio`: 合约剩余交易日生命周期比例 $[0, 1]$。

#### 数学公式
- **月份周期编码**:
  $$\theta = \frac{2\pi \cdot \text{Month}}{12}, \quad \text{sin} = \sin(\theta), \quad \text{cos} = \cos(\theta)$$
- **Session 内分钟进度**:
  $$\text{progress} = \min\left(1.0, \max\left(0.0, \frac{\text{Timestamp} - \text{SessionStart}}{\text{SessionEnd} - \text{SessionStart}}\right)\right)$$
- **剩余生命周期比例**:
  $$\text{contract\_life\_remaining\_ratio} = \frac{\max(\text{busday\_count}(\text{CurrentDate}, \text{LastTradingDay}) + 1, 1)}{\text{TotalTradingDayCount}}$$

---

### 3.7 滚动风险与流动性状态特征 (Rolling Risk & Liquidity Features)

基于 5min 行情数据和窗口大小 $w \in \{12, 24, 48, 96, 288\}$ 滚动计算：

#### 风险特征 (Risk State Features)
1. `atr_pct_{w}`: 真实波幅 (ATR) 相对 Close 的百分比。
   $$\text{TR}_t = \max(\text{High}_t - \text{Low}_t, |\text{High}_t - \text{Close}_{t-1}|, |\text{Low}_t - \text{Close}_{t-1}|)$$
   $$\text{atr\_pct}_w = \frac{\frac{1}{w}\sum_{i=0}^{w-1} \text{TR}_{t-i}}{\text{Close}_t} \times 100$$
2. `historical_volatility_{w}`: 日化历史收益率波动率。
   $$r_t = \ln\left(\frac{\text{Close}_t}{\text{Close}_{t-1}}\right), \quad \text{HV}_w = \text{Std}(r, w) \times \sqrt{\text{bars\_per\_day}}$$
3. `rolling_volatility_{w}`: 指数加权收益率波动率 ($\text{ewm\_std}(r, w)$)。
4. `parkinson_volatility_{w}`: Parkinson 极差波动率。
   $$\text{PV}_w = \sqrt{\frac{1}{4 \ln 2 \cdot w} \sum_{i=0}^{w-1} \left(\ln \frac{\text{High}_{t-i}}{\text{Low}_{t-i}}\right)^2}$$
5. `garman_klass_volatility_{w}`: Garman-Klass 开高低收波动率。
   $$\text{GKV}_w = \sqrt{\frac{1}{w} \sum_{i=0}^{w-1} \left[ 0.5 \left(\ln \frac{\text{High}_{t-i}}{\text{Low}_{t-i}}\right)^2 - (2\ln 2 - 1) \left(\ln \frac{\text{Close}_{t-i}}{\text{Open}_{t-i}}\right)^2 \right]}$$
6. `realized_volatility_{w}`: 已实现波动率。
   $$\text{RV}_w = \sqrt{\sum_{i=0}^{w-1} r_{t-i}^2}$$

#### 流动性特征 (Liquidity State Features)
1. `relative_volume_{w}`: 当前成交量相对窗口均量的倍数 ($\frac{\text{Volume}_t}{\text{Mean}(\text{Volume}, w)}$)。
2. `relative_amount_{w}`: 当前成交额相对窗口均额的倍数 ($\frac{\text{Tradeval}_t}{\text{Mean}(\text{Tradeval}, w)}$)。
3. `relative_open_interest_{w}`: 当前持仓量相对窗口均持仓量的倍数 ($\frac{\text{OpenInterest}_t}{\text{Mean}(\text{OpenInterest}, w)}$)。
4. `open_interest_change_ratio_{w}`: 持仓量相对 $w$ 根 bar 前的变化率。
   $$\text{OI\_Change}_w = \frac{\text{OpenInterest}_t - \text{OpenInterest}_{t-w}}{\text{OpenInterest}_{t-w}}$$

---

### 3.8 滚动时间算子特征 (Time-Operator Rolling OHLCV Features)

窗口 $w \in \{5, 10, 20, 60\}$ 算子：
- `roc_w`: 变动率指标 ($\frac{\text{Close}_{t-w}}{\text{Close}_t}$)。
- `ma_w` / `std_w`: 均值与标准差相对 Close 比例。
- `beta_w`: 线性回归斜率估算。
- `max_w` / `min_w` / `qtlu_w` / `qtld_w`: 80%/20% 滚动分位数比率。
- `rank_w`: 当前 Close 在窗口内的百分位秩。
- `imax_w` / `imin_w` / `imxd_w`: 最高价/最低价在窗口内的出现位置比率及位置距离。
- `rsv_w`: 未成熟随机指标 ($RSV = \frac{\text{Close} - \text{Low}_w}{\text{High}_w - \text{Low}_w + \epsilon}$)。
- `cntp_w` / `cntn_w` / `cntd_w`: 过去收益率为正/负的 bar 占比及差值。
- `corr_w`: Close 收益率与 Volume 变化的滚动相关系数。
- `sump_w` / `sumn_w` / `sumd_w`: 上涨收益率和占总收益率绝对值和的比率。

---

### 3.9 FineFT RL 交易过程状态特征 (Trading Process Features)

作为强化学习环境 `reset()` 与 `step()` 中 `info["trading_info"]` 返回给 Q 网络的 4 维连续输入：

1. `position_exposure`: 归一化持仓暴露。
   $$\text{position\_exposure} = \frac{\text{position}}{\text{max\_abs\_position}} \in [-1, 1]$$
2. `single_holding_return_rate`: 单次持仓收益率（平仓或反向变仓时重置）。
3. `single_holding_max_drawdown`: 单次持仓最大回撤率。
4. `current_holding_duration_norm`: 归一化当前持仓持续步数。
   $$\text{current\_holding\_duration\_norm} = \min\left(\frac{\text{holding\_steps}}{\text{holding\_duration\_norm\_steps}}, 1.0\right)$$

---

## 4. Data Quality & Fail-Fast Verification

为保证高频特征数据质量，系统执行以下 fail-fast 校验机制：
1. **Quote 校验**: 检查最优报价 `BidPrice1 > 0`, `AskPrice1 > 0` 且 `BidPrice1 < AskPrice1`（涨跌停单边盘口除外）。
2. **Turnover 校验**: 若 $\Delta \text{Volume} > 0$ 但 $\Delta \text{Turnover} \le 0$ 或缺失，立即报错。
3. **OpenInterest 校验**: 若 `OpenInterest` 包含 Null、NaN 或 Inf，下采样过程立即终止。
4. **有限值校验**: 所有输出特征在写入 Feather/NPY 前必须清理 Null/NaN/Inf，保证模型输入 100% 有限。

