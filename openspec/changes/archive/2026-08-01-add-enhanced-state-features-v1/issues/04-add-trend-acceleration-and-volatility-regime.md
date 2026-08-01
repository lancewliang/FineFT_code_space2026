# 04 — 趋势加速度与波动率 Regime 算子

**What to build:** 实现归一化趋势加速度 (`price_velocity_10m`, `price_acceleration_10m_norm`) 与 Garman-Klass / Parkinson 波动率历史 192 周期连续分位数 (`garman_klass_vol_quantile_192`, `parkinson_vol_zscore_192`) 特征算子。

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

- [ ] 实现价格 EMA 速度与二阶加速度，并除以历史波动率归一化。
- [ ] 实现 Garman-Klass / Parkinson 波动率在滚动 192 周期窗口内的连续分位数百分比分值 [0.0, 1.0]。
- [ ] 保证平滑连续可导，零波动率时安全兜底。
