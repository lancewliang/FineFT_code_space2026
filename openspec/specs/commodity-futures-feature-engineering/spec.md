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
4. **`commodity/daily_base_feature.py` & `weekly_base_feature.py`**: 将日线与周线 OHLCV、成交额、持仓量等基础周期数据整理为混频特征输入。
5. **`commodity/daily_mixed_frequency_feature.py`, `weekly_mixed_frequency_feature.py` & `mixed_frequency_feature.py`**: 派生日/周级历史状态特征，并按 `trading_day` 与 `calendar_week` 合并回分钟级样本。
6. **`commodity/cross_month_feature.py`**: 派生主力/次主力跨月价差、相对价差、成交量占比、持仓占比、跨月曲线形态及主力切换状态特征。
7. **`time_operator/multi_processing_util.py` & `time_operator_util.py`**: 派生动量、波动率、分位数、相关性等滚动时间序列特征，以及滚动风险特征 (ATR, HV, PV, GKV, RV) 和流动性特征 (RelVol, RelAmt, RelOI, OI_Change)。
8. **FineFT RL Trading Process**: 在强化学习环境中生成实时交易状态特征（`position_exposure`, `single_holding_return_rate`, `single_holding_max_drawdown`, `current_holding_duration_norm`）。

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
7. `is_session_first_bar`: 当前是否为所属 Session 实际存在的前两根 Bar 之一。
8. `is_session_last_bar`: 当前是否为所属 Session 实际存在的最后两根 Bar 之一。
9. `contract_month_sin`: 合约交割月份的正弦周期编码。
10. `contract_month_cos`: 合约交割月份的余弦周期编码。
11. `contract_life_remaining_ratio`: 合约剩余交易日生命周期比例 $[0, 1]$。

#### 数学公式
- **月份周期编码**:
  $$\theta = \frac{2\pi \cdot \text{Month}}{12}, \quad \text{sin} = \sin(\theta), \quad \text{cos} = \cos(\theta)$$
- **Session 内分钟进度**:
  $$\text{progress} = \min\left(1.0, \max\left(0.0, \frac{\text{Timestamp} - \text{SessionStart}}{\text{SessionEnd} - \text{SessionStart}}\right)\right)$$
- **剩余生命周期比例**:
  $$\text{contract\_life\_remaining\_ratio} = \frac{\max(\text{busday\_count}(\text{CurrentDate}, \text{LastTradingDay}) + 1, 1)}{\text{TotalTradingDayCount}}$$

---

### 3.7 日/周混频历史状态特征 (Mixed-Frequency Daily & Weekly Features)

当前实现将日线与周线历史状态特征拼接到分钟级商品期货样本中。输出列以 `timestamp` 加 `MIXED_FREQUENCY_FEATURE_COLUMNS` 为准，日线特征按 `trading_day` 对齐，周线特征按 `calendar_week` 对齐。

#### 日线特征窗口
- `DAY_ROLLING_WINDOWS = (1, 2, 5, 10, 15, 30)`。
- `prev_day_*` 输出完整的上一交易日 K 线形态、成交结构与持仓变化特征。
- `prev_{N}_day_*`（`N != 1`）仅输出多日窗口成交结构与持仓变化特征，避免输出多日绝对价格/成交量水平。

#### 日线输出列
- `prev_day_return`: 上一交易日收益率。
- `prev_day_range_pct`: 上一交易日振幅比例。
- `prev_day_body_pct`: 上一交易日实体比例。
- `prev_day_upper_shadow_pct` / `prev_day_lower_shadow_pct`: 上/下影线比例。
- `prev_day_close_position`: 收盘价在 High-Low 区间内的位置。
- `prev_day_body_to_range`: 实体占总振幅比例。
- `prev_day_upper_shadow_to_range` / `prev_day_lower_shadow_to_range`: 上/下影线占总振幅比例。
- `prev_day_vwap_deviation_pct` / `prev_day_twap_deviation_pct`: 收盘价相对 VWAP/TWAP 的偏离比例。
- `prev_day_trade_up_ratio` / `prev_day_trade_down_ratio` / `prev_day_trade_imbalance`: 上涨/下跌成交占比及不平衡度。
- `prev_day_open_interest_change`: 持仓量变化率。
- `prev_day_turnover_rate`: 成交量相对持仓量的换手率。
- `prev_{N}_day_trade_up_ratio`, `prev_{N}_day_trade_down_ratio`, `prev_{N}_day_trade_imbalance`, `prev_{N}_day_open_interest_change`, `prev_{N}_day_turnover_rate`: 多日窗口聚合成交结构与持仓状态。

#### 周线特征窗口
- `WEEK_ROLLING_WINDOWS = (1, 2, 4, 6)`。
- `prev_week_*` 输出完整的上一自然交易周 K 线形态、成交结构与持仓变化特征。
- `prev_{N}_week_*`（`N != 1`）仅输出多周窗口成交结构与持仓变化特征，避免输出多周绝对价格/成交量水平。

#### 周线输出列
- `prev_week_return`: 上一周收益率。
- `prev_week_range_pct`: 上一周振幅比例。
- `prev_week_body_pct`: 上一周实体比例。
- `prev_week_close_position`: 周收盘价在 High-Low 区间内的位置。
- `prev_week_body_to_range`: 周实体占总振幅比例。
- `prev_week_upper_shadow_to_range` / `prev_week_lower_shadow_to_range`: 周上/下影线占总振幅比例。
- `prev_week_vwap_deviation_pct` / `prev_week_twap_deviation_pct`: 周收盘价相对 VWAP/TWAP 的偏离比例。
- `prev_week_trade_up_ratio` / `prev_week_trade_down_ratio` / `prev_week_trade_imbalance`: 周上涨/下跌成交占比及不平衡度。
- `prev_week_open_interest_change`: 周持仓量变化率。
- `prev_week_turnover_rate`: 周成交量相对持仓量的换手率。
- `prev_{N}_week_trade_up_ratio`, `prev_{N}_week_trade_down_ratio`, `prev_{N}_week_trade_imbalance`, `prev_{N}_week_open_interest_change`, `prev_{N}_week_turnover_rate`: 多周窗口聚合成交结构与持仓状态。

#### 数学公式
- **周期收益率**:
  $$\text{return} = \frac{\text{Close} - \text{Open}}{\text{Open}}$$
- **周期振幅比例**:
  $$\text{range\_pct} = \frac{\text{High} - \text{Low}}{\text{Close}}$$
- **收盘位置**:
  $$\text{close\_position} = \frac{\text{Close} - \text{Low}}{\text{High} - \text{Low} + \epsilon}$$
- **成交不平衡度**:
  $$\text{trade\_imbalance} = \frac{\text{trade\_up} - \text{trade\_down}}{\text{trade\_up} + \text{trade\_down} + \epsilon}$$
- **换手率**:
  $$\text{turnover\_rate} = \frac{\text{Volume}}{\text{OpenInterest} + \epsilon}$$
- **上/下影线比例**:
  $$\text{upper\_shadow\_pct} = \frac{\text{High} - \max(\text{Open}, \text{Close})}{\text{Open}}$$
  $$\text{lower\_shadow\_pct} = \frac{\min(\text{Open}, \text{Close}) - \text{Low}}{\text{Open}}$$

---

### 3.8 跨月合约状态特征 (Cross-Month Contract State Features)

当前实现输出 `CROSS_MONTH_FEATURE_COLUMNS` 中的 22 个 `cm_*` 特征，分为角色标记、当前合约对主力/次主力的关系特征，以及主力/次主力和三合约序列的跨月结构特征。

#### 特征列表与含义
- `cm_contract_role_main` / `cm_contract_role_sub` / `cm_contract_role_other`: 当前合约在主力/次主力/其他中的角色标记。
- `cm_current_main_log_price_ratio` / `cm_current_sub_log_price_ratio`: 当前合约与主力/次主力的对数价比。
- `cm_current_main_relative_price_spread` / `cm_current_sub_relative_price_spread`: 当前合约与主力/次主力的相对价差。
- `cm_current_main_volume_share_current` / `cm_current_sub_volume_share_current`: 当前合约相对主力/次主力的成交量份额。
- `cm_current_main_open_interest_share_current` / `cm_current_sub_open_interest_share_current`: 当前合约相对主力/次主力的持仓份额。
- `cm_main_sub_log_price_ratio`: 主力与次主力对数价比。
- `cm_main_sub_relative_price_spread`: 主力与次主力相对价差。
- `cm_main_sub_volume_share_sub`: 主力与次主力成交量份额中的次主力占比。
- `cm_main_sub_open_interest_share_sub`: 主力与次主力持仓份额中的次主力占比。
- `cm_m1_m2_log_price_ratio` / `cm_m2_m3_log_price_ratio`: 相邻月度合约对数价比。
- `cm_m1_m2_relative_price_spread` / `cm_m2_m3_relative_price_spread`: 相邻月度合约相对价差。
- `cm_m1_m2_m3_butterfly_ratio`: 三腿蝶式结构比例。
- `cm_m1_m2_open_interest_share_m2` / `cm_m2_m3_open_interest_share_m3`: 相邻远月合约的持仓份额。

#### 数学公式
- **相对跨月价差**:
  $$\text{spread\_pct\_main\_next} = \frac{\text{LastPrice}_{main} - \text{LastPrice}_{next}}{\text{LastPrice}_{main} + \epsilon}$$
- **成交量份额**:
  $$\text{main\_volume\_share} = \frac{\text{Volume}_{main}}{\text{Volume}_{main} + \text{Volume}_{next} + \epsilon}$$
- **换月压力**:
  $$\text{main\_contract\_switch\_pressure} = \frac{\text{Volume}_{next} + \text{OpenInterest}_{next}}{\text{Volume}_{main} + \text{OpenInterest}_{main} + \epsilon}$$

---

### 3.9 滚动风险与流动性状态特征 (Rolling Risk & Liquidity Features)

基于 5min 行情数据和配置窗口 `windows` 滚动计算。当前实现由调用方传入窗口集合，测试覆盖 `w = 12`；历史配置中常用窗口为 $w \in \{12, 24, 48, 96, 288\}$。

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

### 3.10 滚动时间算子特征 (Time-Operator Rolling OHLCV Features)

当前 `expected_columns.py` 注册的商品期货滚动时间特征窗口为 $w \in \{2, 6, 12, 16, 24, 48\}$。实现按列族模板生成价格、OHLCV、分位数、排名、相关性和正负收益拆分等特征：
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

### 3.11 增强状态特征与多尺度趋势/动量相对水平特征 (Enhanced State & Relative Trend Features)

由 `time_operator_util.py:process_enhanced_state_features` 派生，包含微观订单流压力、因果无量纲市场状态锚点及尺度无关的趋势相对水平特征：

1. `log_price_slope_48` / `log_price_slope_96`: 48/96 根 Bar 对数价格 OLS 线性斜率。
2. `trend_to_noise_48` / `trend_to_noise_96`: 对数价格斜率相对对数收益率波动率的比值。
3. `signed_efficiency_48`: 窗口净价格位移与累积绝对位移之比（Kaufman 方向效率，$[-1, 1]$）。
4. `trend_r2_48`: 线性趋势拟合优度 $R^2$（$[0, 1]$）。
5. `log_return_vol_quantile_192`: 48 周期对数波动率在过去 192 步历史中的经验分位数（$[0, 1]$）。
6. `log_price_slope_quantile_192`: 48 周期对数价格斜率在过去 192 步历史中的经验分位数（$[0, 1]$）。
7. `ema_divergence_20_60` / `ema_divergence_60_120` / `ema_divergence_20_120`: 多跨度 EMA 相对扩散比率。
   $$\text{ema\_divergence}_{i, j} = \frac{\text{EMA}_i - \text{EMA}_j}{\text{EMA}_j}$$
8. `macd_histogram_norm`: 归一化无量纲 MACD 柱状图动能特征。
   $$\text{DIF}_{norm} = \frac{\text{EMA}_{12}(\text{Close}) - \text{EMA}_{26}(\text{Close})}{\text{Close}}$$
   $$\text{DEA}_{norm} = \text{EMA}_9(\text{DIF}_{norm})$$
   $$\text{macd\_histogram\_norm} = \text{DIF}_{norm} - \text{DEA}_{norm}$$
9. `price_velocity_10m`: 快慢 EMA（10/20）相对价差比率。
10. `price_acceleration_10m_norm`: 价格速度一阶差分经滚动波动率标准化后的加速度。
11. `ema_slope_96` / `ema_slope_192`: 长周期 EMA 的时间斜率。
12. `plus_di_14` / `minus_di_14` / `adx_14`: 14 周期趋向指标系列。
13. `vwap_slope_96` / `vwap_slope_192`: VWAP 时间斜率。
14. `price_to_vwap_zscore_48` / `price_to_vwap_zscore_96`: 价格相对 VWAP 偏离度的 48/96 周期滚动 Z-score。
    $$\text{dev}_t = \frac{\text{Close}_t - \text{VWAP}_t}{\text{Close}_t}, \quad \text{price\_to\_vwap\_zscore}_w = \frac{\text{dev}_t - \text{Mean}(\text{dev}, w)}{\text{Std}(\text{dev}, w) + \epsilon}$$
15. `rsi_14_norm`: 14 周期 Wilder RSI 归一化中心化振荡器（$[-1, 1]$）。
    $$\text{rsi\_14\_norm} = \frac{\text{RSI}_{14} - 50}{50}$$
16. `stoch_k_14_norm` / `stoch_d_14_norm`: 14 周期随机振荡器 K/D 归一化指标（$[-1, 1]$）。
    $$\text{stoch\_k\_norm} = \frac{K_{14} - 50}{50}, \quad \text{stoch\_d\_norm} = \frac{D_{14} - 50}{50}$$
17. `cci_14_norm`: 14 周期顺势指标归一化截断值（$[-3, 3]$）。
    $$\text{CCI}_{14} = \frac{\text{TP} - \text{SMA}_{14}(\text{TP})}{0.015 \cdot \text{MeanDeviation}_{14}}, \quad \text{cci\_14\_norm} = \text{clip}\left(\frac{\text{CCI}_{14}}{100}, -3, 3\right)$$
18. `realized_vol_zscore_192`: 12 周期已实现波动率相对 192 步历史的滚动 Z-score。
19. `realized_vol_term_ratio_12_48`: 12 周期对 48 周期已实现波动率期限结构比率（度量短中期波动率扩张/收缩）。
    $$\text{realized\_vol\_term\_ratio}_{12, 48} = \frac{\text{RV}_{12}}{\text{RV}_{48} + \epsilon}$$
20. `imbalance_1_zscore_48` / `imbalance_5_zscore_48`: 1 档与 5 档盘口不平衡度的 48 周期滚动 Z-score。
21. `imbalance_1_persistence_20m`: 1 档盘口不平衡度的 20 步指数平滑持久性特征。
22. `ofi_norm_zscore_48` / `ofi_persistence_20m`: 归一化订单流不平衡量 (OFI) 的 48 周期 Z-score 与 20 步平滑持久性。
23. `cvd_slope_96` / `cvd_slope_192`: 累积成交量差 (CVD) 相对斜率。

---

### 3.12 FineFT RL 交易过程状态特征 (Trading Process Features)

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
