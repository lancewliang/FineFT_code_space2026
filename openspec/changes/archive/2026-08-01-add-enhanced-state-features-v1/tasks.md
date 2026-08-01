# Tasks: add-enhanced-state-features-v1

- [x] **Task 1: 扩展盘口与订单流基础算子 (`downscale.py`)**
  - 在下采样中实现五档 OFI (`ofi_5m_norm`, `level5_ofi_weighted_norm`)、买卖相对价差 (`relative_bid_ask_spread`) 及基础盘口深度特征。
- [x] **Task 2: 实现盘口耗竭与恢复算子 (`depth_depletion_replenishment`)**
  - 实现 Top-5 深度衰减率 (`ask_depth_depletion_5m`, `bid_depth_depletion_5m`) 与深度恢复比率 (`depth_replenishment_ratio_20m`)。
- [x] **Task 3: 实现成交方向持续性与价差 Z-Score 算子 (`trade_persistence_and_spread`)**
  - 实现基于 Tick Rule 估计的买卖净额比率及 EWMA 方向持久度，以及价差 48 周期 Z-Score (`spread_widening_zscore_48`)。
- [x] **Task 4: 实现趋势加速度与波动率 Regime 算子 (`trend_acceleration_and_vol_regime`)**
  - 实现标准化趋势二阶加速度 (`price_acceleration_10m_norm`) 及 Garman-Klass / Parkinson 波动率历史分位数 (`garman_klass_vol_quantile_192`)。
- [x] **Task 5: 实现成交量/OI Regime 与跨月价差动态算子 (`volume_oi_regime_and_cross_month`)**
  - 实现量价持仓三元连续交互标量 (`price_oi_vol_interaction_10m`)，以及主力/次主力 Log 价差速度与持仓迁移速率 (`cm_main_sub_log_price_spread_velocity_10m`, `cm_open_interest_shift_speed_10m`)。
- [x] **Task 6: 接入特征校验与模块集成**
  - 将新特征注册至预期特征列表并接入 `DataQualityValidator` 进行 Fail-fast 与 NaN 校验。
- [x] **Task 7: 编写单元测试与流水线验证**
  - 编写 `test_enhanced_state_features.py` 测试用例，验证特征生成正确性，并运行完整的 Feature Selection 与 Scale Save 流程。
