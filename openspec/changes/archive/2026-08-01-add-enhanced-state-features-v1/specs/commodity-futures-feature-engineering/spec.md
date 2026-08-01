# Commodity Futures Feature Engineering Specification - Enhanced State Features

## ADDED REQUIREMENTS

### 3.12 增强状态特征 (Enhanced State Features)

#### 1. 短周期 Level-5 OFI (Order Flow Imbalance)
- **要求**: 系统必须在 5min/10min/30min 下采样 bar 上生成归一化 Level-5 订单流不平衡特征。
- **数学公式**:
  针对第 $k$ 档盘口 ($k \in \{1..5\}$)，根据相邻 Tick 挂单价变化计算买卖量变化 $\Delta V_{b,t}^{(k)}, \Delta V_{a,t}^{(k)}$：
  $$\Delta V_{b,t}^{(k)} = \begin{cases} v_{b,t}^{(k)}, & \text{if } P_{b,t}^{(k)} > P_{b,t-1}^{(k)} \\ v_{b,t}^{(k)} - v_{b,t-1}^{(k)}, & \text{if } P_{b,t}^{(k)} = P_{b,t-1}^{(k)} \\ 0, & \text{if } P_{b,t}^{(k)} < P_{b,t-1}^{(k)} \end{cases}, \quad \Delta V_{a,t}^{(k)} = \begin{cases} 0, & \text{if } P_{a,t}^{(k)} > P_{a,t-1}^{(k)} \\ v_{a,t}^{(k)} - v_{a,t-1}^{(k)}, & \text{if } P_{a,t}^{(k)} = P_{a,t-1}^{(k)} \\ v_{a,t}^{(k)}, & \text{if } P_{a,t}^{(k)} < P_{a,t-1}^{(k)} \end{cases}$$
  $$\text{level5\_ofi\_weighted\_norm}_t = \frac{\sum_{k=1}^5 \frac{1}{k} (\Delta V_{b,t}^{(k)} - \Delta V_{a,t}^{(k)})}{\sum_{k=1}^5 (v_{b,t}^{(k)} + v_{a,t}^{(k)}) + \epsilon}$$
- **约束**: 输出必须位于 $[-1.0, 1.0]$ 区间内，若双边挂单总量为 0 则输出 0.0。

#### 2. 盘口耗竭与恢复 (Depth Depletion & Replenishment)
- **要求**: 计算卖盘/买盘 Top-5 挂单深度的相对衰减率与滚动均值回复比率。
- **数学公式**:
  $$\text{depth\_depletion}_{a,t,k} = \frac{\max(0, \text{Depth5}_{a,t-k} - \text{Depth5}_{a,t})}{\text{Depth5}_{a,t-k} + \epsilon}$$
  $$\text{depth\_replenishment\_ratio}_{t,w} = \frac{\text{Depth5}_{a,t} + \text{Depth5}_{b,t}}{\text{MA}_w(\text{Depth5}_a + \text{Depth5}_b) + \epsilon}$$
- **特征命名**: `ask_depth_depletion_{k}`, `bid_depth_depletion_{k}`, `depth_replenishment_ratio_{w}`。

#### 3. 买卖价差扩大特征 (Spread Widening Dynamics)
- **要求**: 计算相对买卖价差及其历史滚动窗口 Z-Score，用于衡量市场流动性紧缺与做市商退场冲击。
- **数学公式**:
  $$\text{relative\_bid\_ask\_spread}_t = \frac{P_{ask1,t} - P_{bid1,t}}{P_{mid,t}}$$
  $$\text{spread\_widening\_zscore}_{t,w} = \frac{\text{relative\_bid\_ask\_spread}_t - \text{MA}_w(\text{relative\_bid\_ask\_spread})}{\text{Std}_w(\text{relative\_bid\_ask\_spread}) + \epsilon}$$
- **约束**: 当 `Std` 为 0 或样本量不足时，Z-Score 兜底为 0.0。

#### 4. 成交方向持续性 (Trade Directional Persistence)
- **要求**: 结合 Tick Rule 估算的买卖成交量，计算主动买卖净额比率及指数加权方向持续性。
- **数学公式**:
  $$\text{trade\_direction\_net\_ratio}_{t,w} = \frac{V_{buy,t,w} - V_{sell,t,w}}{V_{buy,t,w} + V_{sell,t,w} + \epsilon}$$
  $$\text{trade\_direction\_persistence}_{t,w} = \text{EWMA}_\alpha(\text{trade\_direction\_net\_ratio})$$
- **特征命名**: `trade_direction_net_ratio_{w}`, `trade_direction_persistence_{w}`。

#### 5. 趋势加速度 (Trend Acceleration)
- **要求**: 计算价格趋势速度的一阶导数（加速度），并使用滚动波动率归一化。
- **数学公式**:
  $$v_t = \text{EMA}_m(\text{Close}_t) - \text{EMA}_n(\text{Close}_t)$$
  $$a_t = \frac{v_t - v_{t-k}}{\sigma_{\text{Close}, w} + \epsilon}$$
- **特征命名**: `price_velocity_{k}`, `price_acceleration_{k}_norm`。

#### 6. 波动率 Regime (Volatility Regime Indicator)
- **要求**: 计算历史 Garman-Klass / Parkinson 波动率在滚动 192 周期窗口内的连续分位数百分比分值。
- **数学公式**:
  $$\text{vol\_quantile}_{t,w} = \text{PercentileRank}_w(\sigma_{GK,t}) \in [0.0, 1.0]$$
- **特征命名**: `garman_klass_vol_quantile_{w}`, `parkinson_vol_zscore_{w}`。

#### 7. 成交量/持仓量 Regime (Volume Open Interest Regime)
- **要求**: 计算价格变化方向、持仓量变化方向与相对成交量的连续三元交互乘积。
- **数学公式**:
  $$\text{price\_oi\_vol\_interaction}_{t,w} = \text{sign}(\Delta_k P_t) \times \text{sign}(\Delta_k \text{OI}_t) \times \frac{V_{t,k}}{\text{MA}_w(V) + \epsilon}$$
- **特征命名**: `price_oi_vol_interaction_{k}`, `oi_change_rate_norm_{w}`。

#### 8. 跨月价差动态变化 (Cross-Month Spread Dynamics)
- **要求**: 计算主力与次主力合约 Log 价差变化率及跨月持仓份额转移速率。
- **数学公式**:
  $$\text{cm\_spread\_velocity}_{t,k} = (\ln P_{main,t} - \ln P_{sub,t}) - (\ln P_{main,t-k} - \ln P_{sub,t-k})$$
  $$\text{cm\_oi\_shift\_speed}_{t,k} = \frac{\text{OI}_{sub,t}}{\text{OI}_{main,t} + \text{OI}_{sub,t} + \epsilon} - \frac{\text{OI}_{sub,t-k}}{\text{OI}_{main,t-k} + \text{OI}_{sub,t-k} + \epsilon}$$
- **特征命名**: `cm_main_sub_log_price_spread_velocity_{k}`, `cm_open_interest_shift_speed_{k}`。
