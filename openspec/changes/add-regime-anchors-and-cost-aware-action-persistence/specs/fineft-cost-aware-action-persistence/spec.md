## ADDED Requirements

### Requirement: 环境提供无副作用的逐动作预计换仓成本
Futures Trading Environment SHALL 在每个决策时点提供从当前动作切换到每个可用动作的预计即时交易摩擦，并保持实际环境状态不变。

#### Scenario: 预计成本复用实际执行语义
- **WHEN** 系统估算从当前仓位和杠杆切换到目标可用动作
- **THEN** 估算 SHALL 使用当前可见盘口、仓位、杠杆、手续费率、深度截断和 Reverse Position best-effort 语义
- **AND** 成本 SHALL 为非负手续费与不利滑点之和
- **AND** 成本 SHALL 使用与 Q 值相同的钱包余额单位

#### Scenario: 保持动作成本为零
- **WHEN** 目标动作等于当前动作
- **THEN** 预计换仓成本 SHALL 为 `0`

#### Scenario: 估算不得修改环境
- **WHEN** 系统计算一个或多个目标动作的预计成本
- **THEN** 钱包、仓位、杠杆、累计手续费、累计滑点、持仓历史和随机状态 SHALL 保持不变

#### Scenario: 不可用动作不参与选择
- **WHEN** 某动作不在当前可用动作集合
- **THEN** 系统 SHALL 将其排除在预计成本候选和最终动作比较之外

### Requirement: Low-level Q 策略使用成本感知动作迟滞
系统 SHALL 提供训练贪心分支和 Low-level 推理消费者共用的确定性动作规则，在候选动作 Q 优势不足以覆盖预计成本与安全边际时保持当前动作。

#### Scenario: Q 优势严格超过门槛时换仓
- **WHEN** 可用动作中的最佳候选动作不同于当前动作
- **AND** `Q(candidate) - Q(current)` 严格大于 `cost_multiplier * estimated_cost + safety_margin`
- **THEN** 系统 SHALL 返回候选动作

#### Scenario: Q 优势未严格超过门槛时保持
- **WHEN** 最佳候选动作不同于当前动作
- **AND** `Q(candidate) - Q(current)` 小于或等于 `cost_multiplier * estimated_cost + safety_margin`
- **THEN** 系统 SHALL 返回当前动作

#### Scenario: 当前动作已是最佳动作
- **WHEN** 可用动作中的最佳候选动作等于当前动作
- **THEN** 系统 SHALL 直接返回当前动作

#### Scenario: 风险约束优先于迟滞
- **WHEN** 当前动作因强平、涨跌停、保证金或其他环境约束不再可用
- **THEN** 系统 SHALL 从可用动作中返回最佳候选
- **AND** 系统 SHALL NOT 因迟滞继续保持不可用动作

#### Scenario: 错误初始仓位可以退出
- **WHEN** 平仓或换向动作可用且其 Q 优势严格通过门槛
- **THEN** 系统 SHALL 允许退出当前仓位
- **AND** 系统 SHALL NOT 增加固定最短持仓时间

#### Scenario: 随机探索和教师轨迹不受迟滞影响
- **WHEN** 动作来自 epsilon 随机探索、固定风格 rollout 或 DP expert path
- **THEN** 系统 SHALL 保持现有动作行为
- **AND** 系统 SHALL NOT 对该动作应用 Q 优势迟滞

### Requirement: 成本迟滞配置与行为可复现
系统 SHALL 持久化并审计成本感知动作迟滞的开关、成本倍数、安全边际和逐步决策事实。

#### Scenario: 默认关闭保持旧行为
- **WHEN** 新开关未显式启用或模型产物缺少新配置
- **THEN** 动作选择 SHALL 退化为现有可用动作 argmax
- **AND** 旧模型与旧实验 SHALL 保持可复现

#### Scenario: 配置参数合法性
- **WHEN** 成本倍数或安全边际为负数，或 Q/预计成本包含非有限值
- **THEN** 系统 SHALL Fail-fast 并报告对应参数或动作

#### Scenario: 训练与推理配置一致
- **WHEN** 模型使用成本迟滞训练并被推理消费者加载
- **THEN** 模型产物 SHALL 提供训练时的开关、成本倍数和安全边际
- **AND** 推理 SHALL 使用相同配置或对显式不一致 Fail-fast

#### Scenario: 逐步诊断记录候选与最终动作
- **WHEN** 成本迟滞处理一次模型贪心动作
- **THEN** 诊断 SHALL 记录当前动作、原始候选、最终动作、Q 优势、预计成本、门槛和决策原因
- **AND** 汇总 SHALL 记录候选换仓数、执行换仓数、被抑制换仓数和估计避免成本

#### Scenario: 所有 Low-level 消费者使用共享规则
- **WHEN** 相同 Q、当前动作、可用动作、预计成本和配置分别进入训练贪心分支、独立 Low-level 测试和路由后的 Low-level 推理
- **THEN** 三者 SHALL 返回相同最终动作

