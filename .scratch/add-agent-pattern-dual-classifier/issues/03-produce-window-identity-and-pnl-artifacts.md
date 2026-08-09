# 03 — 生成窗口身份、PnL 和基础分析表

**What to build:** 把逐步行为事实转换为可供后续分类的形态识别窗口，为每个窗口生成区分 Initial-action 的稳定身份和唯一 PnL，并贯通固定 schema 的 Window 与 Expanded tables。此时普通窗口允许暂用未分类哨兵，涨跌停事件窗口使用 KX1，从而先完成端到端数据骨架。

**Blocked by:** 02 — 扩展到完整 Agent 形态候选全集。

**Status:** ready-for-agent

- [ ] Label 1 至 Label 5 的每条行为轨迹只生成连续、不重叠的完整 20 步窗口。
- [ ] Label 0 和 Label 6 的每条完整行为轨迹恰好生成一个 KX1 涨跌停事件窗口。
- [ ] 普通轨迹不足 20 步的尾部不生成窗口，并在 Coverage Report 记录 dropped-tail steps、gross PnL 和 net PnL。
- [ ] window id 由固定身份与边界字段的规范 JSON 计算 SHA-256，相同市场区间的不同 Initial-action 产生不同身份。
- [ ] window id 不受分类结果、PnL、阈值、绝对目录或输出位置影响。
- [ ] gross PnL 使用窗口已实现 PnL 与边界浮动 PnL，net PnL再扣手续费；slippage 仅作诊断。
- [ ] Window Table 使用固定英文列顺序，每个 window id 恰好一行，patterns 字段为合法 JSON 数组。
- [ ] Expanded Table 使用固定英文列顺序，唯一键为 `(window_id, kline_pattern, strategy_pattern)`。
- [ ] 测试证明窗口/尾部 PnL 守恒，Expanded PnL 不可跨策略形态求和为账户总额。

