# add-risk-liquidity-state-features

## 背景与目标
为了使 RL Agent 能够在不同的市场波动率机制、交易活跃度及持仓量压力下做出决策，商品期货数据预处理流程需要新增一组风险与流动性 State Feature。
本变更在 `downscale.py` 的 `BASE_FEATURE` 中导出 5min `open_interest` 列，并在 `time_operator` 中生成 6 个风险状态特征与 4 个流动性状态特征（带有窗口后缀），作为普通候选特征进入后续 Feature Selection 与 Scale Save。

## 关键决策
- **组件划分**：下采样层 (`downscale.py`) 提供 `open_interest` 基础列，历史窗口滚动特征统一由 `time_operator` 生成。
- **6 个风险特征**：`atr_pct`, `historical_volatility`, `rolling_volatility`, `parkinson_volatility`, `garman_klass_volatility`, `realized_volatility`。`historical_volatility` 的日化系数使用 `sqrt(bars_per_day)`，根据品种 Trading Session 推导。
- **4 个流动性特征**：`relative_volume`, `relative_amount` (使用 tradeval), `relative_open_interest`, `open_interest_change_ratio`。
- **常规候选特征**：不加入 mandatory 特征，不跳过 Scale Save 缩放，由 Feature Selection 自由筛选。
- **数值安全性**：零分母或非法值安全兜底，禁止 `NaN`、`inf` 或 `-inf` 写入特征文件。

## 验收标准
- `downscale.py` 输出 5min 级 `open_interest` 列；缺少 `OpenInterest` 时 fail-fast。
- `time_operator` 生成 10 个带 `{window}` 后缀的风险与流动性特征列，数值均为有限浮点数。
- 特征能顺利进入 concat、Dataset Split、Feature Selection 及 Scale Save。
