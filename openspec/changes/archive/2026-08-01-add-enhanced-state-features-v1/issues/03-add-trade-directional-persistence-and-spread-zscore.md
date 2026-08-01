# 03 — 成交方向持续性与价差 Z-Score 算子

**What to build:** 实现基于 Tick Rule 的主动买卖净额比率 (`trade_direction_net_ratio_5m`)、EWMA 方向持久度 (`trade_direction_persistence_20m`) 及相对价差 48 周期 Z-Score (`spread_widening_zscore_48`) 特征算子。

**Blocked by:** 01 — 盘口 Level-5 OFI 与相对价差基础算子

**Status:** ready-for-agent

- [ ] 实现目标窗口内主动买卖估计量的净额归一化比率。
- [ ] 实现指数加权衰减平滑的方向持续性算子。
- [ ] 实现相对买卖价差在 48 周期历史窗口内的 Z-Score 算子。
