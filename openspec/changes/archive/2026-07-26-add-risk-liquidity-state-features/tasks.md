# 任务列表：add-risk-liquidity-state-features

- [x] 在 `downscale.py` 中输出 5min `open_interest` 列并在缺少源字段时 fail-fast
- [x] 在 `time_operator` 中实现 6 个风险状态特征 (`atr_pct`, `historical_volatility`, `rolling_volatility`, `parkinson_volatility`, `garman_klass_volatility`, `realized_volatility`)
- [x] 在 `time_operator` 中实现 4 个流动性状态特征 (`relative_volume`, `relative_amount`, `relative_open_interest`, `open_interest_change_ratio`)
- [x] 实现基于品种 Trading Session 的 `bars_per_day` 日化系数推导
- [x] 编写并运行测试验证特征生成与异常安全保护
