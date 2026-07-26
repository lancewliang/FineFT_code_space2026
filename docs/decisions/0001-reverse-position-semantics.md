# Reverse Position uses best-effort semantics

Reverse Position 在一步内执行先平仓后反向开仓，采用 best-effort 语义：平仓一定成功，反向开仓可能因保证金不足而失败（position 归零）。不采用 atomic 语义（要么两步都成功要么整个动作被拒绝），因为真实交易中先平后开是两笔独立订单，不存在原子保证；agent 应通过 `calculate_avaiable_action()` 学会在保证金不足时不选择反手。

# Reverse Position truncates to position_list on insufficient depth

当 orderbook 深度不足以开目标反向仓位时，截断到 position_list 中最大可行值而非允许任意 position 值。这保持了 position 始终在 position_list 中的不变量，避免破坏下游 `calculate_avaiable_action()` 的 assert、`map_position_leverage_to_action()` 的映射和离散动作空间设计。

# Reverse Position is gated by allow_reverse_position flag (default off)

Reverse Position 功能通过 `allow_reverse_position` 参数控制，默认为 `False`。关闭时 `change_of_wallet()` 维持原有 warning + 拒绝行为，`calculate_avaiable_action()` 维持方向限制，`create_optimal_q_table()` 维持 `-max_punishment` 惩罚。这确保向后兼容——已有训练结果、Q 表和专家路径不受影响。开关需从 `Base_Env.__init__` 一路传递到 `change_of_wallet`、`calculate_avaiable_action`、`create_optimal_q_table` 和 `create_optimal_q_table_from_df`。

# DP Q table must stay consistent with environment on reverse position

`create_optimal_q_table` 中 `future_position * current_position < 0` 的处理必须与环境 `change_of_wallet` 的行为一致。当 `allow_reverse_position=True` 时，Q 表不再对反手动作赋 `-max_punishment`，改为调用 `change_of_wallet()` 计算反手后的实际 reward；反手失败（position 归零，即 `changed_position != future_position`）时仍赋 `-max_punishment`。不一致会导致 pretrain warmup 产生与环境实际行为不匹配的专家路径。
