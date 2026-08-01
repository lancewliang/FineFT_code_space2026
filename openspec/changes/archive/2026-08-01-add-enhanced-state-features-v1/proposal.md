# Proposal: add-enhanced-state-features-v1

## 背景与目标
为了增强 Reinforcement Learning Agent 及机器学习模型在商品期货复杂市场环境中的感知能力，需要在原有 KLINE、QUOTE、时间编码、混频及基础风险/流动性特征的基础上，进一步扩展 8 类高阶增强特征（Enhanced State Features）：
1. **短周期 OFI (Short-period Level-5 Order Flow Imbalance)**
2. **盘口耗竭与恢复 (Depth Depletion and Replenishment)**
3. **价差扩大状态 (Spread Widening Dynamics)**
4. **成交方向持续性 (Trade Directional Persistence)**
5. **趋势加速度 (Trend Acceleration)**
6. **波动率 Regime (Volatility Regime Indicator)**
7. **成交量/持仓量 Regime (Volume Open Interest Regime Dynamics)**
8. **跨月价差动态变化 (Cross-Month Spread Dynamics)**

这些特征旨在补充微观订单流冲击、盘口挂单弹性、做市商退场预警、趋势二阶导数及主力持仓迁移速率等信号，从而提高特征库的信号丰富度与预测稳定性。

---

## 关键决策 (Key Decisions)

### 1. 特征定义与数学公式标准

- **短周期 OFI (`ofi_5m_norm`, `level5_ofi_weighted_norm`)**:
  采用 Cont et al. 标准公式，在 Level-1 至 Level-5 上按档位权重 $1/k$ 计算买卖单边净注入量，并用五档挂单总深度进行比率归一化，输出范围 $[-1.0, 1.0]$。
- **盘口耗竭与恢复 (`ask_depth_depletion_5m`, `depth_replenishment_ratio_20m`)**:
  - Depletion: 计算 $k$ 个 Bar 内买/卖盘口 Top-5 深度的相对衰减比例；
  - Replenishment: 计算当前深度相对历史 $N$ 周期移动平均深度的回复比率。
- **价差扩大状态 (`relative_bid_ask_spread`, `spread_widening_zscore_48`)**:
  计算买卖一价差相对中间价的比率 $\frac{P_{ask1} - P_{bid1}}{P_{mid}}$，并计算其在滚动 48 周期窗口内的 Z-Score，识别流动性枯竭状态。
- **成交方向持续性 (`trade_direction_net_ratio_5m`, `trade_direction_persistence_20m`)**:
  结合 Tick Rule 成交方向估计，计算主动买卖净额比率 $\frac{V_{buy} - V_{sell}}{V_{buy} + V_{sell}}$ 及指数衰减平滑后的连续方向持久度。
- **趋势加速度 (`price_velocity_10m`, `price_acceleration_10m_norm`)**:
  价格一阶变化速度 $v_t = \text{EMA}_m(P) - \text{EMA}_n(P)$ 的二阶差分 $a_t = v_t - v_{t-k}$，除以滚动历史波动率进行标准化，消除不同价格量级的失真。
- **波动率 Regime (`garman_klass_vol_quantile_192`, `parkinson_vol_zscore_192`)**:
  计算 Garman-Klass / Parkinson 波动率，并输出当前波动率在历史滚动 192 周期（约 1 个交易日）内的连续 Percentile Rank 分位数 $[0.0, 1.0]$，保持梯度的连续可导性。
- **成交量/持仓量 Regime (`price_oi_vol_interaction_10m`, `oi_change_rate_norm_10m`)**:
  三元连续交互特征 $\text{sign}(\Delta P) \cdot \text{sign}(\Delta \text{OI}) \cdot \frac{V}{\text{MA}(V)}$，区分多头建仓、空头建仓与平仓盘驱动。
- **跨月价差动态变化 (`cm_main_sub_log_price_spread_velocity_10m`, `cm_open_interest_shift_speed_10m`)**:
  主力与次主力合约 Log 价差变化率 $\Delta_k (\ln P_{main} - \ln P_{sub})$ 及持仓份额迁移速率 $\Delta_k \left( \frac{\text{OI}_{sub}}{\text{OI}_{main} + \text{OI}_{sub}} \right)$。

### 2. 管道与组件划分 (Architecture & Pipeline Boundaries)

- **下采样与盘口层 (`downscale.py`)**: 计算单 Bar 内的快照级基础比率、价差及五档 OFI。
- **时间算子与衍生层 (`time_operator` / 增强算子模块)**: 计算跨 Bar 历史窗口的滚动 Z-Score、Percentile Rank、二阶加速度及连贯性指标。
- **特征工程状态**: 所有增强特征作为常规 Candidate State Features 进入下游 Feature Selection（筛选）及 Scale Save（标准化缩放），不设为 mandatory，不跳过缩放。

### 3. 数值安全性与异常保护 (Data Quality & Fail-Fast)

- 所有除法运算统一增加零分母兜底偏置 $\epsilon = 1e-8$。
- 对极值实施 Winsorization 裁剪（限制在有限区间），并严格保证输出中无 `Null`、`NaN`、`Inf`。

---

## 验收标准 (Acceptance Criteria)

1. **算子正确性**：8 类增强特征在商品期货（如 `fu` 等）5min/10min/30min 下采样数据上准确生成，计算公式符合定义。
2. **数值质量**：生成数据集通过 `NaN` 校验和有限值校验，无 `NaN` 或 `Inf` 引入。
3. **流水线兼容性**：新特征能够无缝接入 `Feature Selection`（计算 RankIC / CatBoost Importance / Correlation 去重）并顺利完成 `Scale Save` 导出。
4. **测试覆盖**：编写完善的 Pytest 单元测试，覆盖各增强特征算子的边缘输入与正常输出。
