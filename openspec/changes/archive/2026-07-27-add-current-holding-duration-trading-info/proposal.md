# add-current-holding-duration-trading-info

## 背景与目标

当前 Stage I 低层 agent 的 Q 网络显式消费 `trading_info` 作为 Trading Process Feature。现有 `trading_info` 由 Futures Trading Environment 生产，包含 signed position exposure、当前持仓收益率和当前持仓最大回撤率。它能表达当前仓位方向、仓位规模、收益和风险，但不能表达当前持仓已经持续了多久。

这会让模型无法区分“刚开仓的同方向风险暴露”和“已经持续很多 env step 的同方向风险暴露”。在期货交易中，当前持仓时长是风险感知行为的重要上下文；同样的 exposure、收益率和最大回撤，在不同持仓年龄下可能需要不同动作。

本变更目标是从 `Base_Env` 和 `trading_info` 契约出发，将 `current_holding_duration_norm` 作为第四个 Trading Process Feature 传递给低层模型，而不是在某个训练脚本里临时追加一维。

## 用户场景

### 场景 1：模型能识别当前持仓年龄

Stage I 训练研究者希望低层 agent 能知道当前持仓已持续的 env step 数，并据此学习更稳健的加仓、减仓、平仓或继续持有行为。

### 场景 2：当前持仓语义在所有环境消费者中一致

维护者希望 `trading_info` 的新增字段由 Futures Trading Environment 统一维护，训练、测试、路由和 replay buffer 消费的都是同一份环境契约。

### 场景 3：同方向加仓/减仓不被误判为新持仓

用户已经确认当前持仓是一段方向性风险暴露。只要持仓方向不变，同方向加仓或减仓都不结束当前持仓，也不重置当前持仓时长。

### 场景 4：旧低层权重不需要兼容

用户已经确认不需要兼容旧的三维 `trading_info` 低层 checkpoint。本变更可以作为明确的模型输入契约升级；旧权重加载失败是可接受结果。

## 设计方向

采用“环境契约升级 + 模型维度同步”的方案。

1. Futures Trading Environment 继续作为 `trading_info` 的唯一生产者。
2. `TRADING_INFO_KEYS` 扩展为四个字段：
   - `position_exposure`
   - `single_holding_return_rate`
   - `single_holding_max_drawdown`
   - `current_holding_duration_norm`
3. `Base_Env` 维护当前持仓时长的原始 env step 计数。
4. `trading_info` 只暴露归一化后的 `current_holding_duration_norm`。
5. 归一化公式为 `min(current_holding_duration / holding_duration_norm_steps, 1.0)`。
6. `holding_duration_norm_steps` 是可配置参数，默认值为 `180`。
7. Qnet / ensemble_Qnet 的默认 `TRADING_INFO_DIM` 升级为 `4`，所有 Stage I 低层训练和测试路径与环境契约同步。

## 关键决策

- 当前持仓时长按 env step 计数，不按自然时间或 wall-clock 时间计数。
- 空仓时当前持仓时长为 `0`。
- 从空仓开到非零 position 后，第一个可观测状态的当前持仓时长为 `1`。
- `initial_state` 已经带非零 position 时，`reset()` 后当前持仓时长为 `1`。
- 同方向持仓继续累加。
- 同方向加仓或减仓不重置当前持仓时长。
- 平仓到 `0` 后当前持仓时长归 `0`。
- 反手后新方向当前持仓时长从 `1` 开始。
- `current_holding_duration_norm` 是模型侧字段名；避免使用 `holding_time`、`holding_length` 或未归一化的 `current_holding_duration` 作为 `trading_info` 字段名。
- 本变更不兼容旧三维低层 checkpoint，不实现自动迁移或兼容加载。

## 范围边界

**包含：**

- 更新 Futures Trading Environment 的 `trading_info` 契约。
- 更新 Base_Env、环境初始化 helper 和继承 Base_Env 的环境类，使 `holding_duration_norm_steps` 能传递到环境。
- 更新低层模型默认 `TRADING_INFO_DIM`。
- 更新 Stage I 低层训练、并行训练、测试、聚合环境和高层路由中依赖低层模型的维度假设。
- 更新 replay buffer 和 Qnet 相关 focused tests，使四维 `trading_info` 成为默认契约。
- 更新环境 Trading Process Feature 测试，覆盖当前持仓时长的主要边界。

**不包含（本次）：**

- 不迁移或兼容旧三维 checkpoint。
- 不新增独立模型输入通道。
- 不改变 `single_holding_return_rate` 或 `single_holding_max_drawdown` 语义。
- 不改成按自然时间、分钟数或交易 Session 内时间计数。
- 不改变数据预处理、Feature Selection、Scale Save 或 State Feature 合约。
- 不做持仓时长归一化窗口的实验调参。

## 验收标准

- [x] `TRADING_INFO_KEYS` 包含 `current_holding_duration_norm`，且 `trading_info` 返回四维数组。
- [x] 默认 `holding_duration_norm_steps` 为 `180`，并可通过环境初始化参数覆盖。
- [x] reset 空仓时 `current_holding_duration_norm == 0`。
- [x] reset 非零初始仓位时 `current_holding_duration_norm == 1 / holding_duration_norm_steps`。
- [x] 开仓后第一个可观测状态 `current_holding_duration_norm == 1 / holding_duration_norm_steps`。
- [x] 同方向持仓、加仓、减仓会继续累加当前持仓时长。
- [x] 平仓到 0 后 `current_holding_duration_norm == 0`。
- [x] 反手后新方向 `current_holding_duration_norm == 1 / holding_duration_norm_steps`。
- [x] 当前持仓时长超过归一化窗口后，`current_holding_duration_norm` 截断为 `1.0`。
- [x] Qnet / ensemble_Qnet 默认接受四维 `trading_info`。
- [x] Stage I 低层训练和测试路径与新的四维 `trading_info` 契约同步。
- [x] 相关 focused tests 和 OpenSpec strict validation 通过。
