# 任务列表：add-trading-process-features-as-network-input

- [x] 在 `base_env.py` 中导出 `trading_info` 数组 (`position_exposure`, `single_holding_return_rate`, `single_holding_max_drawdown`)
- [x] 在 `replay_buffer_DQN.py` 中更新 `NETWORK_INFO_KEYS` 常量并保留 `trading_info` 采样
- [x] 在 `low_level.py` 中增加 `TRADING_INFO_DIM=3` 及 `fc_trading` 编码层
- [x] 全量更新 Stage I, Stage II, Stage III, Ablation, Baseline 脚本中的 Qnet/ensemble_Qnet 调用点
- [x] 验证环境、Buffer 和网络端到端输入测试
