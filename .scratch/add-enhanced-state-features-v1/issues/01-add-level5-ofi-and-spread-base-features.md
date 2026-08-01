# 01 — 盘口 Level-5 OFI 与相对价差基础算子

**What to build:** 在 `downscale.py` 下采样中计算五档订单流不平衡量 (`level5_ofi_weighted_norm`) 与相对买卖价差 (`relative_bid_ask_spread`)，并在下采样生成的 5min/10min/30min DataFrame 中正确导出这两列。

**Blocked by:** None — can start immediately

**Status:** ready-for-agent

- [ ] 实现五档价格与挂单量变化量计算公式，按档位权重 1/k 计算 Level-5 OFI 并经总深度归一化至 [-1.0, 1.0]。
- [ ] 计算相对买卖价差 (P_ask1 - P_bid1) / P_mid，零分母或缺口异常时安全兜底。
- [ ] 在 `downscale.py` 输出中包含 `level5_ofi_weighted_norm` 与 `relative_bid_ask_spread` 列。
