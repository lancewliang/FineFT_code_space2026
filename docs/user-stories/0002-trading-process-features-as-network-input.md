---
status: accepted
source: user-request
---

# Trading Process Features as Network Input

## User Story

As a DiHFT low-level RL agent
I want to receive trading process features (position_exposure, single_holding_return_rate, single_holding_max_drawdown) as part of my network input
So that I can make risk-aware and position-aware decisions — differentiating close/hold/open actions based on current position state and risk exposure, rather than treating all positions identically

## Background

当前 Q 网络的 `forward()` 接收四路输入：`state`（市场技术指标）、`previous_action`（上一步动作编号）、`time`（资金费率倒计时）、`avaliable_action`（可用动作掩码）。模型完全不知道当前账户的交易状态（持仓、收益、风险等），也不知道交易过程中的成本和收益信息。

这里沿用现有接口名 `previous_action`，但它在代码中实际表示当前 `position/leverage` 映射回 action space 的编码，不是完整的上一条交易命令；新增 `position_exposure` 用归一化 signed exposure 补足仓位状态语义。

环境 `info` 字典中已经计算了丰富的交易过程特征，但只有 4 个键被 replay buffer 保存并传给网络。以下 3 个特征已在环境中计算但未被利用：

| 特征 | 来源 | 含义 | 理由 |
|------|------|------|------|
| `position_exposure` | `personal_state[3] / max_abs_position` | 归一化当前持仓暴露（正=多，负=空，0=空仓） | 同样的市场状态，持多仓和空仓的最优决策完全不同。当前只能通过 `previous_action` 间接推断，映射有损；归一化后与风险收益率输入尺度更接近 |
| `single_holding_return_rate` | `info["single_holding_return_rate"]` | **这笔交易**的收益率（比率） | 告诉模型"当前这笔持仓赚了多少比例"，直接服务于持仓 vs 平仓决策 |
| `single_holding_max_drawdown` | `info["single_holding_max_drawdown"]` | **这笔交易**的最大回撤（比率） | 告诉模型"当前这笔持仓的风险有多大"，是风控的关键指标 |

### 为什么不包含 `unrealized_pnl`

`unrealized_pnl`（未实现盈亏）是绝对金额，与 `single_holding_return_rate` 高度相关：

```
single_holding_return_rate = single_holding_return / require_money
```

两者都反映浮盈浮亏，但 `return_rate` 是比率、`unrealized_pnl` 是绝对金额。已知 `position_exposure`（归一化持仓暴露）和 `return_rate`（收益率）后，`unrealized_pnl` 的边际信息量有限。为避免特征冗余、减少参数量和过拟合风险，**统一使用无量纲输入，不使用绝对值**。

## Decisions

| # | 问题 | 决策 | 理由 |
|---|------|------|------|
| 1 | Actor 范围 | 每个 **子 Qnet** 独立接收 `trading_info`，`ensemble_Qnet` 广播给所有子 Qnet | 与现有 `previous_action`/`time` 的处理方式一致：ensemble 透传，每个 Qnet 独立编码 |
| 2 | 特征范围 | **1 个归一化持仓暴露特征 + 2 个比率风险收益特征**：position_exposure, single_holding_return_rate, single_holding_max_drawdown | 持仓状态 + 收益率 + 回撤率构成风险感知决策信息；不包含 `unrealized_pnl` 避免与 `return_rate` 冗余 |
| 3 | 特征量纲 | **统一使用无量纲输入**，不使用原始手数或绝对金额 | position_exposure 归一化到 [-1, 1]，收益/回撤使用比率；绝对金额（如 unrealized_pnl）与 return_rate 高度冗余 |
| 4 | 收益/回撤语义 | `single_holding_return_rate` 和 `single_holding_max_drawdown` 属于**当前持仓**，不是跨交易的累计值 | 同方向变仓不结束当前持仓；平仓或持仓方向改变会结束当前这一笔交易 |
| 5 | 新特征编码方式 | 独立线性层 `fc_trading` 将 3 维特征映射到 `hidden_nodes` 维，与 `state_hidden` 拼接 | 与现有 `fc1(state)` 和 `fc3(previous_action)` 的编码风格一致 |
| 6 | 新特征维度参数 | 新增 `TRADING_INFO_DIM` 参数，**默认值 3，全局生效** | 3 个交易过程特征；全局生效，所有使用 `ensemble_Qnet`/`Qnet` 的文件需同步更新 |
| 7 | 拼接位置 | 拼接到 `information_hidden = cat([state_hidden, previous_action_hidden, time, trading_hidden])` | 保留现有 `previous_action` 的离散 action-space 编码路径，新增连续 Trading Process Feature 编码路径；两者同属 agent-side 状态信息，但语义和尺度不同，因此独立编码后在同层融合 |
| 8 | 特征归一化 | 在环境/helper 中将 `position` 归一化为 `position_exposure = position / max_abs_position`，不在模型内归一化 | 三维输入尺度更接近；原始 position 仍保留在 `personal_state`/诊断信息中 |
| 9 | replay buffer 存储 | 将 3 个特征打包为 `trading_info` 数组存入 info 字典 | 避免在 buffer 中增加 3 个独立键，保持 info 结构简洁 |
| 10 | 核心价值 | **风险感知的仓位决策** — 模型能根据持仓状态和风险指标做出差异化的平仓/持仓/开仓决策 | 当前模型对所有仓位一视同仁，无法区分"浮盈 5% 的多头"和"浮亏 10% 的空头" |
| 11 | `previous_action` 与 `position_exposure` 边界 | 保留现有 `previous_action` 接口名；`previous_action` 表示当前 position/leverage 的 action-space 编码，`position_exposure` 表示归一化 signed exposure | 避免全局重命名带来的高风险改动，同时明确 `position_exposure` 不是冗余字段 |
| 12 | 迁移策略 | **强制迁移**：所有 Qnet/ensemble_Qnet 调用必须显式传入 `trading_info`，不提供 `None`/零向量兼容路径 | 避免遗漏调用点时静默退化为无交易过程特征输入；旧 checkpoint 不兼容，需重新训练或显式处理 |
| 13 | 字段顺序 | 使用唯一常量/小 helper 固定 `trading_info` 字段顺序：position_exposure, single_holding_return_rate, single_holding_max_drawdown | 防止环境、replay buffer、训练和测试路径中数组字段顺序漂移；shape 正确不代表语义正确 |
| 14 | replay buffer 白名单 | 在 replay buffer 内定义共享 `NETWORK_INFO_KEYS` 常量，所有采样路径复用它并包含 `trading_info` | 避免多处硬编码白名单导致 `sample()`、`sample_evaluate()` 或 sunrise 分支漏传新输入 |
| 15 | return_rate 分母 | 使用当前持仓生命周期内的累计投入/占用口径，沿用 `calculate_required_money(...)` 逻辑 | 同方向变仓延续当前持仓，分母应反映这笔持仓累计承受的资金占用；持仓结束时必须重置相关 history |
| 16 | position_exposure 分母 | 使用 `max(abs(p) for p in position_list)` 动态计算 `max_abs_position` | 不新增 CLI 参数、不硬编码 8；当 position_choices/max_holding_number 改变时仍保持输入范围稳定在 [-1, 1] |
| 17 | 验收测试范围 | 覆盖环境、replay buffer 和模型三层数据流；缺少 `trading_info` 的 Qnet/ensemble_Qnet 调用必须失败 | 验证强制迁移真实生效，避免 shape 正常但新输入未接入的静默退化 |
| 18 | 调用点迁移范围 | **一次性全量迁移** repo 内所有 `Qnet`/`ensemble_Qnet` 构造和调用点 | 与强制迁移策略一致；避免部分脚本坏掉或被迫引入默认零向量兼容 |

## Scenarios

### 环境在 reset 时输出 trading_info

Given:
- 环境刚执行 `reset()`
- 初始 position_exposure=0, single_holding_return_rate=0, single_holding_max_drawdown=0

When:
- 获取 `info["trading_info"]`

Then:
- `info["trading_info"]` = `np.array([0, 0, 0])`

### 环境在 step 正常路径输出 trading_info

Given:
- 当前持仓 position=4（多头 4 手）
- `max_abs_position=8`，因此 position_exposure=0.5
- 这笔交易的 single_holding_return_rate=0.05（收益率 5%）
- 这笔交易的 single_holding_max_drawdown=0.02（最大回撤 2%）

When:
- 执行 `step(action)` 后获取 `info["trading_info"]`

Then:
- `info["trading_info"]` = `np.array([0.5, 0.05, 0.02])`

### 环境在爆仓路径输出 trading_info

Given:
- 爆仓发生，position 归零

When:
- 获取 `info["trading_info"]`

Then:
- `info["trading_info"]` 表示动作执行后的可观测状态
- `info["trading_info"]` = `np.array([0, 0, 0])`

### 开仓时 return_rate 和 max_drawdown 重置

Given:
- 当前 position=0（空仓）
- agent 执行开仓动作，position 变为 4

When:
- 下一步获取 `info["trading_info"]`

Then:
- return_rate=0, max_drawdown=0（新交易开始，收益率和回撤从零计算）

### 平仓时 return_rate 和 max_drawdown 归零

Given:
- 当前 position=4（多头），return_rate=0.05, max_drawdown=0.02
- agent 执行平仓动作，position 变为 0

When:
- 下一步获取 `info["trading_info"]`

Then:
- position_exposure=0, return_rate=0, max_drawdown=0

### 持仓方向改变时结束当前交易

Given:
- 当前 position=4（多头），return_rate=0.05, max_drawdown=0.02
- agent 执行反向开仓动作，position 从正数变为负数

When:
- 下一步获取 `info["trading_info"]`

Then:
- 旧的多头交易结束
- 新的空头交易开始
- return_rate 和 max_drawdown 按新持仓重新开始计算

### 同方向变仓时延续当前持仓

Given:
- 当前 position=4（多头），return_rate=0.05, max_drawdown=0.02
- agent 执行同方向加仓或减仓动作，position 仍为正数

When:
- 下一步获取 `info["trading_info"]`

Then:
- 当前持仓不结束
- return_rate 和 max_drawdown 不清零，继续描述这笔多头持仓的风险收益状态
- return_rate 分母使用当前持仓生命周期内的累计投入/占用口径

### trading_info 通过 replay buffer 传递

Given:
- replay buffer 的 info 键白名单包含 `trading_info`
- 环境返回的 info 中 `trading_info` = [position_exposure, return_rate, max_drawdown]

When:
- 从 replay buffer 采样一个 batch

Then:
- `infos["trading_info"]` 的 shape 为 `(batch_size, 3)`
- `next_infos["trading_info"]` 的 shape 为 `(batch_size, 3)`

### Qnet 接收并编码 trading_info

Given:
- `Qnet.__init__` 中 `TRADING_INFO_DIM=3`
- `fc_trading = Linear(3, hidden_nodes)`

When:
- 调用 `Qnet.forward(state, time, previous_action, avaliable_action, trading_info)`

Then:
- `trading_hidden = relu(fc_trading(trading_info))` shape 为 `(batch, hidden_nodes)`
- `information_hidden = cat([state_hidden, previous_action_hidden, time, trading_hidden])` shape 为 `(batch, N_ACTIONS + 2*hidden_nodes + time_bedding)`
- 输出 Q 值 shape 不变：`(batch, N_ACTIONS)`

### ensemble_Qnet 广播 trading_info 给每个子 Qnet

Given:
- `ensemble_Qnet` 包含 N=7 个子 Qnet
- `trading_info` shape 为 `(batch, 3)`

When:
- 调用 `ensemble_Qnet.forward(state, time, previous_action, avaliable_action, trading_info)`

Then:
- 每个子 Qnet 独立接收相同的 `trading_info` 并独立编码
- 输出 Q 值 shape 为 `(batch, N, N_ACTIONS)`

### 训练脚本从 info 提取 trading_info 传入网络

Given:
- `update()` / `update_pretrain()` / `act_single_context()` 中从 info 提取 `trading_info`

When:
- 构建网络输入

Then:
- `trading_info = info["trading_info"].float().reshape(bs, -1).to(device)` 传入 `eval_net()`
- `trading_info_ = info_["trading_info"].float().reshape(bs, -1).to(device)` 传入 `target_net()`

### 不同持仓状态下 trading_info 值不同

Given:
- 同一市场状态（state 相同）

| 场景 | position_exposure | return_rate | max_drawdown |
|------|----------|-------------|--------------|
| 空仓 | 0 | 0 | 0 |
| 多头浮盈 | 0.5 | 0.05 | 0.02 |
| 多头浮亏 | 0.5 | -0.03 | 0.08 |
| 空头浮盈 | -0.5 | 0.02 | 0.01 |
| 空头浮亏 | -0.5 | -0.06 | 0.12 |

When:
- 模型接收不同的 `trading_info`

Then:
- 模型输出不同的 Q 值分布（因为 `fc_trading` 编码不同）
- 空仓时倾向于开仓动作，浮盈时倾向于持仓/平仓，浮亏且 max_drawdown 大时倾向于止损平仓

## Implementation Notes

### 涉及文件分类

#### 核心修改（4 个文件）

1. **`FineFT/env/env_class/base_env.py`**:
   - 定义唯一字段顺序常量/小 helper，例如 `TRADING_INFO_KEYS = ("position_exposure", "single_holding_return_rate", "single_holding_max_drawdown")` 和 `build_trading_info(...)`
   - `position_exposure = position / max(abs(p) for p in self.position_list)`，不新增 CLI 参数、不硬编码最大仓位
   - `reset()` 和 `step()` 的 info 字典中添加 `"trading_info"` 键
   - 值为 `np.array([position / max_abs_position, single_holding_return_rate, single_holding_max_drawdown])`
   - `trading_info` 统一表示动作执行后的可观测状态；平仓或爆仓后为空仓 `[0, 0, 0]`

2. **`FineFT/RL/util/replay_buffer_DQN.py`**:
   - 定义共享 `NETWORK_INFO_KEYS` 常量，包含 `"trading_info"`
   - 所有 `sample()` / `sample_evaluate()` / sunrise 采样路径复用该常量

3. **`FineFT/model/low_level.py`**:
   - `Qnet.__init__()`: 新增 `TRADING_INFO_DIM` 参数（默认 3），创建 `self.fc_trading = nn.Linear(TRADING_INFO_DIM, hidden_nodes)`
   - `Qnet.forward()`: 新增 `trading_info` 参数，编码后拼接到 `information_hidden`
   - `Qnet.fc2` 输入维度从 `N_ACTIONS + hidden_nodes + time_bedding` 增加到 `N_ACTIONS + 2*hidden_nodes + time_bedding`
   - `ensemble_Qnet.__init__()`: 新增 `TRADING_INFO_DIM` 参数（默认 3），透传给 `Qnet`
   - `ensemble_Qnet.forward()`: 新增 `trading_info` 参数，透传给每个子 `Qnet`
   - `ensemble_Qnet.get_best_q()`: 新增 `trading_info` 参数，透传给 `forward()`

4. **`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`**:
   - 新增 `--trading_info_dim` 参数（默认 3）
   - `Weighted_Contexts_DQN.__init__()`: 传入 `TRADING_INFO_DIM=self.trading_info_dim` 给 `ensemble_Qnet`
   - `update()`: 从 `info`/`info_` 提取 `trading_info`，传入 `eval_net()` 和 `target_net()`
   - `update_pretrain()`: 同上
   - `act_single_context()`: 从 `info` 提取 `trading_info`，传入 `eval_net()`

#### 全局生效需同步更新的文件（23 个）

因 `TRADING_INFO_DIM` 默认值为 3，以下所有使用 `ensemble_Qnet`/`Qnet` 的文件需同步更新：

**Stage I 训练（5 个）**:
- `FineFT/RL/DiHFT/low_level/trace_udpate_index.py`
- `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`
- `FineFT/RL/DiHFT/ablation/converge_steps_sun/FineFT.py`
- `FineFT/RL/DiHFT/ablation/converge_steps_sun/FineFT_without_pretrain.py`
- `FineFT/RL/DiHFT/ablation/converge_steps_sun/EarnHFT_random.py`

**Stage I 测试（5 个）**:
- `FineFT/RL/DiHFT/low_level/test_agent_index.py`
- `FineFT/RL/DiHFT/low_level/test_agent_average.py`
- `FineFT/RL/DiHFT/ablation/converge_steps_sun/FineFT_test.py`
- `FineFT/RL/DiHFT/ablation/converge_steps_sun/FineFT_wo_pretrain_test.py`
- `FineFT/RL/DiHFT/ablation/converge_steps_sun/EarnHFT_random_test.py`

**Stage II/III 高层（8 个）**:
- `FineFT/RL/DiHFT/high_level/vae_routing_util.py`
- `FineFT/RL/DiHFT/high_level/vae_routing_final_result_macro_action.py`
- `FineFT/RL/DiHFT/high_level/train_high_level_seq.py`
- `FineFT/RL/DiHFT/high_level/train_high_level.py`
- `FineFT/RL/DiHFT/high_level/test_single_agent.py`
- `FineFT/RL/DiHFT/high_level/test_high_level_rejection.py`
- `FineFT/RL/DiHFT/high_level/test_high_level.py`
- `FineFT/RL/DiHFT/high_level/high_level_heurstic.py`

**Ablation（2 个）**:
- `FineFT/RL/DiHFT/ablation/safe_routing.py`
- `FineFT/RL/DiHFT/ablation/converge_steps_sun/EarnHFT_PES.py`

**Baselines（4 个）**:
- `FineFT/RL/base/winnow_test.py`
- `FineFT/RL/base/dqn_train.py`
- `FineFT/RL/base/dqn_test.py`
- `FineFT/RL/base/sunrise_dqn_train.py`

**每个文件的更新模式相同**：
1. 构造 `ensemble_Qnet`/`Qnet` 时传入 `TRADING_INFO_DIM=3`
2. 调用 `forward()`/`get_best_q()` 时从 `info` 提取并传入 `trading_info`
3. 加载已保存模型时需确保 `state_dict` 中包含 `fc_trading` 层的权重（旧模型不兼容，需重新训练）

### 数据流

```
base_env.py reset()/step():
  info["trading_info"] = np.array([position_exposure, single_holding_return_rate, single_holding_max_drawdown])
        |
        v
replay_buffer.add(state, info, action, reward, next_state, next_info, done)
        |
        v
replay_buffer.sample() → infos["trading_info"] shape: (batch, 3)
        |
        v
update()/update_pretrain()/act_single_context():
  trading_info = info["trading_info"].float().reshape(bs, -1).to(device)
        |
        v
ensemble_Qnet.forward(state, time, previous_action, avaliable_action, trading_info)
  → 每个 Qnet: fc1(state) + fc3(previous_action) + fc_time(time) + fc_trading(trading_info)
  → concat → fc2 → out → mask
```

### 模型兼容性说明

`TRADING_INFO_DIM=3` 默认值意味着 `fc_trading` 层始终存在，且所有 `Qnet`/`ensemble_Qnet` 调用必须显式传入 `trading_info`。已保存的旧模型（无 `fc_trading` 权重）的 `state_dict` 与新模型不兼容，**需重新训练**；只有在明确迁移旧 checkpoint 时才允许显式使用 `strict=False`，不得通过 `trading_info=None` 或默认零向量让旧调用静默继续运行。

实现完成后 repo 内不得残留旧签名的 `Qnet`/`ensemble_Qnet` 构造或调用点；所有训练、测试、高层、ablation 和 baseline 路径必须一次性迁移。

### 验收测试

1. 环境测试：覆盖 `reset()`、正常持仓、同方向变仓、平仓、持仓方向改变和爆仓后的 `trading_info` 值与重置语义。
2. Replay buffer 测试：覆盖 `sample()` 和 `sample_evaluate()` 返回 `infos["trading_info"]` / `next_infos["trading_info"]`，shape 为 `(batch_size, 3)` 且 dtype 为 float。
3. 模型测试：缺少 `trading_info` 时 `Qnet`/`ensemble_Qnet` 调用失败；传入 `(batch, 3)` 时输出 shape 保持不变。

## LANGUAGE.md Sync

已同步术语：

**Trading Process Feature (交易过程特征)**:
从环境交易执行过程中产生的实时特征，包括 position_exposure（归一化持仓暴露）、single_holding_return_rate（这笔交易的收益率）和 single_holding_max_drawdown（这笔交易的最大回撤），全部为无量纲输入、不使用原始手数或绝对金额；打包为 `trading_info` 数组传入 Q 网络的 `fc_trading` 层编码；与 State Feature（市场技术指标）互补，State Feature 描述市场状态，Trading Process Feature 描述 agent 自身持仓状态和风险暴露。
_Avoid_: 交易特征、过程特征

**当前持仓 (Current Holding)**:
从 position 由 0 变为非 0 或持仓方向改变时开始，并在平仓到 0 或持仓方向再次改变时结束的一笔交易；同方向加仓或减仓不结束当前持仓。
_Avoid_: 未平仓交易、连续非零仓位

**Previous Action**:
现有低层 Q 网络输入名，实际表示当前 position/leverage 映射到 action space 的编码。
_Avoid_: 上一条交易命令、完整交易状态

## MAP.md Sync

已同步 `Stage I Low-level Training` 组件描述：Q 网络输入从四路（state, previous_action, time, avaliable_action）扩展为五路，新增 `trading_info`（Trading Process Feature）输入通道。

已同步 `Futures Trading Environment` 组件描述：`reset()`/`step()` 返回的 info 字典新增 `trading_info` 键。
