# Design: add-current-holding-duration-trading-info

## Context

`trading_info` 是低层 Q 网络的显式输入之一。已有 ADR 要求 Qnet 和 ensemble_Qnet 调用必须显式传入 `trading_info`，避免缺少 Trading Process Feature 时静默运行。当前需求延续这个方向：不新增输入通道，而是升级现有 `trading_info` 契约。

Futures Trading Environment 是 `trading_info` 的生产者。Stage I low-level training、parallel rollout、low-level testing、high-level routing 中的低层模型推理都是消费者。为了避免不同训练脚本各自拼接字段，本变更把当前持仓时长放进环境契约。

## Decisions

### 1. `Base_Env` owns Current Holding Duration

`Base_Env` 维护一个原始 env step 计数，表示 Current Holding Duration。该计数从实际环境 position 的变化更新，而不是从目标 action 直接推断。

这样可以处理订单簿深度不足、保证金不足或 Reverse Position best-effort 导致实际 position 与目标 position 不一致的情况。

### 2. `trading_info` exposes normalized duration only

模型侧字段名为 `current_holding_duration_norm`，公式为：

```text
min(current_holding_duration / holding_duration_norm_steps, 1.0)
```

原始 `current_holding_duration` 可以作为环境内部状态存在，但不作为 `trading_info` 字段暴露。默认归一化窗口为 `180` 个 env step。

### 3. Current Holding is directional exposure, not position lot age

当前持仓是一段方向性风险暴露。同方向加仓或减仓不结束当前持仓，因此不重置持仓时长。只有空仓和平仓归零，或者方向发生反转，才结束旧的当前持仓。

### 4. Reset semantics

`reset()` 根据 `initial_state` 的 position 初始化当前持仓时长：

- position 为 0 时初始化为 0
- position 非 0 时初始化为 1

这表示 episode 初始状态已带非零仓位时，模型看到的是一个已存在的 Current Holding。

### 5. Terminal and liquidation branches must share the same shape helper

现有环境分支中存在硬编码三维零数组。实现时应收敛为统一 helper，保证 reset、普通 step、强平和终止分支都返回四维 `trading_info`。

### 6. Model contract is a breaking change

低层模型默认 `TRADING_INFO_DIM` 升级为 4。旧三维低层 checkpoint 加载失败是预期行为，不做自动补零或迁移。

### 7. Keep module boundaries

- 环境负责产生 Trading Process Feature。
- 模型只消费给定维度的 `trading_info`。
- replay buffer 只保存和采样 `info["trading_info"]`，不理解字段含义。
- 训练和测试脚本只负责把环境输出传给模型，不临时拼接持仓时长字段。

## Data Shape

`trading_info` 顺序为：

```text
[
  position_exposure,
  single_holding_return_rate,
  single_holding_max_drawdown,
  current_holding_duration_norm,
]
```

`TRADING_INFO_KEYS` 是该顺序的唯一命名来源。

## Failure Policy

- `holding_duration_norm_steps <= 0` 时 fail-fast。
- 消费者仍按三维构造低层模型时，shape mismatch 应暴露为错误。
- 旧 checkpoint 因 `fc_trading` shape 不匹配而无法加载时，不做兼容处理。

## Risks

- 这是跨环境和低层模型的输入契约升级，会影响所有低层 checkpoint 和依赖低层模型的测试/路由入口。
- 如果只改训练不改测试或高层路由，会出现训练权重无法加载或推理输入 shape 不匹配。
- 归一化窗口默认 180 是研究默认值，不代表最优参数；后续实验可能需要调参。
