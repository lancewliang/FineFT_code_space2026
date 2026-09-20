# Design: 非主力与次主力合约门控防御与退化消融设计 (add-non-main-contract-defensive-gating)

## Context
在商品期货多合约连续验证集中，远月合约或非活跃时段存在极低的成交量和极浅的五档盘口。在 Stage III 高层 VAE 门控评估和 Optuna 寻优时，若无差别调度投机 Agent 开仓，会引发非理性的滑点与大幅回撤，破坏跨合约平均指标与组合收益率。
基于 ADR-0017 已落地直通的 `prev_day_contract_role_tier` 特征，高层门控可在决策步进行角色感知并实施防御。

## Architecture & Data Flow

1. **状态特征映射**：
   - 门控初始化时从 `self.tech_indicator_list` 中解析 `prev_day_contract_role_tier` 的特征索引 `self.role_tier_index`。
   - 当列表中未配置该特征时，`self.role_tier_index` 确定性置为 `None`，不触发非法访问。
2. **决策前向分流**：
   ```
   [输入当前 State s]
           │
           ▼
   enable_non_main_contract_defense == True
   AND role_tier_index is not None
   AND float(s[role_tier_index]) < 0.5 ?
       ├── YES ──► 触发规则逐步平仓 (_defensive_action)
       │           追加防御槽位 slot_count 至 macro_action_history
       │           返回平仓动作，跳过底层 Q-Network 推理
       │
       └── NO  ──► 继续常规双轴 VAE 似然分位数计算
                   根据阈值判断是否 OOD 或执行选中 Agent
   ```
3. **Optuna 调优与实例复用**：
   - Optuna 主进程与 Worker 进程均解析 `--enable_non_main_contract_defense`。
   - 在 `prepare_base_args` 中透传该标志至 `base_args`。
   - Worker 循环执行 Trial 时，`reconfigure_routing` 同步更新该标志，保持不同 Trial 行为一致。

## Verification & Seams
- **Primary Seam**: `vae_risk_aware_routing.get_action(info, s, current_position, current_leverage)`。
- **Ablation Seam**: 校验 `--enable_non_main_contract_defense=False` 与 `True` 时对相同非主力状态样本的不同响应。
