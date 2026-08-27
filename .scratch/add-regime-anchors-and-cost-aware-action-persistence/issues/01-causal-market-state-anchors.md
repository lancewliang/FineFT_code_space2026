# 01: 生成因果市场状态锚点 State Feature

**What to build:** 让商品期货 State Feature 直接表达中长窗口方向、趋势纯度和相对波动位置，使 Low-level Agent 能识别低波动强趋势，而不依赖不稳定的短周期代理。

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] 生成 `log_price_slope_48`、`log_price_slope_96`、`trend_to_noise_48`、`trend_to_noise_96`、`signed_efficiency_48`、`trend_r2_48` 和 `log_return_vol_quantile_192`。
- [ ] 所有计算只使用当前及历史 Bar；合约边界重置；非正价格 Fail-fast。
- [ ] 窗口不足输出有限的中性值，成熟窗口满足定义的范围约束。
- [ ] 特征对整体价格单位缩放保持不变，并通过 NaN Validation、Scale Save 和候选特征清单。
- [ ] focused tests 覆盖平滑上涨/下跌、常数、含噪趋势、未来追加前缀不变和数值异常。

