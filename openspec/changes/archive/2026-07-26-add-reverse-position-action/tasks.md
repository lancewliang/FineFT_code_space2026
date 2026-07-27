# 任务列表：add-reverse-position-action

- [x] 在 `futures_util.py` 的 `change_of_wallet()` 与 `calculate_avaiable_action()` 中实现反手逻辑与开关控制
- [x] 在 `base_env.py` 中新增 `allow_reverse_position` 参数与 `step()` 持仓反向检测
- [x] 在 `demo_env.py` 及 `create_optimal_q_table` 中支持反手 Q 表计算
- [x] 在 `pretrain_qtable_diagnostics.py` 等 Stage I 文件中透传 `allow_reverse_position`
- [x] 编写并运行单元与集成测试 `tests/test_futures_reverse_position.py`
