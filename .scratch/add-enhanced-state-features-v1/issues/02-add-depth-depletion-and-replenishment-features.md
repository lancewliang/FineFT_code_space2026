# 02 — 盘口耗竭与深度恢复比率算子

**What to build:** 实现 Top-5 深度衰减率 (`ask_depth_depletion_5m`, `bid_depth_depletion_5m`) 与深度恢复比率 (`depth_replenishment_ratio_20m`) 特征算子。

**Blocked by:** 01 — 盘口 Level-5 OFI 与相对价差基础算子

**Status:** ready-for-agent

- [ ] 实现 Top-5 盘口卖盘与买盘深度的 k 周期相对衰减率计算。
- [ ] 实现当前总深度相对滚动均值深度的恢复比率算子。
- [ ] 确保在深度为 0 或输入短缺时返回平滑有限值，无 NaN/Inf。
