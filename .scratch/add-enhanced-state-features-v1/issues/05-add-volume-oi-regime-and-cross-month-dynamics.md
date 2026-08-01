# 05 — 成交量/持仓量 Regime 与跨月价差动态算子

**What to build:** 实现量价持仓三元连续交互标量 (`price_oi_vol_interaction_10m`, `oi_change_rate_norm_10m`) 以及主力/次主力 Log 价差变化速率 (`cm_main_sub_log_price_spread_velocity_10m`) 与持仓迁移速率 (`cm_open_interest_shift_speed_10m`) 特征算子。

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

- [ ] 实现价格符号、持仓变化符号与相对成交量乘积的三元连续交互标量。
- [ ] 在跨月特征模块中实现主力与次主力 Log 价差一阶速度及持仓量占比迁移速率。
- [ ] 保证零持仓或单合约缺失时的 Fail-fast/优雅兜底。
