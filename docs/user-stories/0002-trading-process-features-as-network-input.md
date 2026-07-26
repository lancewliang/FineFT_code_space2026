---
status: accepted
source: user-request
---

# Trading Process Features as Network Input

## User Story

As a DiHFT low-level RL agent
I want to receive trading process features (position, single_holding_return_rate, single_holding_max_drawdown) as part of my network input
So that I can make risk-aware and position-aware decisions — differentiating close/hold/open actions based on current position state and risk exposure, rather than treating all positions identically

## Background

当前 Q 网络的 `forward()` 接收四路输入：`state`（市场技术指标）、`previous_action`（上一步动作编号）、`time`（资金费率倒计时）、`avaliable_action`（可用动作掩码）。模型完全不知道当前账户的交易状态（持仓、收益、风险等），也不知道交易过程中的成本和收益信息。

环境 `info` 字典中已经计算了丰富的交易过程特征，但只有 4 个键被 replay buffer 保存并传给网络。以下 3 个特征已在环境中计算但未被利用：

| 特征 | 来源 | 含义 | 理由 |
|------|------|------|------|
| `position` | `personal_state[3]` | 当前持仓量（正=多，负=空，0=空仓） | 同样的市场状态，持多仓和空仓的最优决策完全不同。当前只能通过 `previous_action` 间接推断，映射有损 |
| `single_holding_return_rate` | `info["single_holding_return_rate"]` | **这笔交易**的收益率（比率） | 告诉模型"当前这笔持仓赚了多少比例"，直接服务于持仓 vs 平仓决策 |
| `single_holding_max_drawdown` | `info["single_holding_max_drawdown"]` | **这笔交易**的最大回撤（比率） | 告诉模型"当前这笔持仓的风险有多大"，是风控的关键指标 |

### 为什么不包含 `unrealized_pnl`

`unrealized_pnl`（未实现盈亏）是绝对金额，与 `single_holding_return_rate` 高度相关：

```
single_holding_return_rate = single_holding_return / require_money
```

两者都反映浮盈浮亏，但 `return_rate` 是比率、`unrealized_pnl` 是绝对金额。已知 `position`（仓位大小）和 `return_rate`（收益率）后，`unrealized_pnl` 的边际信息量有限。为避免特征冗余、减少参数量和过拟合风险，**统一使用比率，不使用绝对值**。

## Decisions

| # | 问题 | 决策 | 理由 |
|---|------|------|------|
| 1 | Actor 范围 | 每个 **子 Qnet** 独立接收 `trading_info`，`ensemble_Qnet` 广播给所有子 Qnet | 与现有 `previous_action`/`time` 的处理方式一致：ensemble 透传，每个 Qnet 独立编码 |
| 2 | 特征范围 | **3 个比率特征**：position, single_holding_return_rate, single_holding_max_drawdown | 持仓状态 + 收益率 + 回撤率构成风险感知决策信息；不包含 `unrealized_pnl` 避免与 `return_rate` 冗余 |
| 3 | 特征量纲 | **统一使用比率**，不使用绝对金额 | 比率无量纲、跨品种可比较、量级可控；绝对金额（如 unrealized_pnl）与 return_rate 高度冗余 |
| 4 | 收益/回撤语义 | `single_holding_return_rate` 和 `single_holding_max_drawdown` 是**这笔交易**（从开仓到当前）的收益率和最大回撤，不是跨交易的累计值 | 环境中这两个值在每次开仓时重置为 0，在持仓期间逐步更新，平仓后归零；它们描述的是"当前这笔持仓"的表现 |
| 5 | 新特征编码方式 | 独立线性层 `fc_trading` 将 3 维特征映射到 `hidden_nodes` 维，与 `state_hidden` 拼接 | 与现有 `fc1(state)` 和 `fc3(previous_action)` 的编码风格一致 |
| 6 | 新特征维度参数 | 新增 `TRADING_INFO_DIM` 参数，**默认值 3，全局生效** | 3 个交易过程特征；全局生效，所有使用 `ensemble_Qnet`/`Qnet` 的文件需同步更新 |
| 7 | 拼接位置 | 拼接到 `information_hidden = cat([state_hidden, previous_action_hidden, time, trading_hidden])` | 所有非掩码信息在同层融合，`fc2` 输入维度相应增加 `hidden_nodes` |
| 8 | 特征归一化 | 不在模型内归一化，依赖环境输出值的合理范围 | position 在 position_list 范围内（如 [-8, 8]），return_rate 通常在 [-1, 1] 附近，max_drawdown 非负且通常 < 1 |
| 9 | replay buffer 存储 | 将 3 个特征打包为 `trading_info` 数组存入 info 字典 | 避免在 buffer 中增加 3 个独立键，保持 info 结构简洁 |
| 10 | 核心价值 | **风险感知的仓位决策** — 模型能根据持仓状态和风险指标做出差异化的平仓/持仓/开仓决策 | 当前模型对所有仓位一视同仁，无法区分"浮盈 5% 的多头"和"浮亏 10% 的空头" |

## Scenarios

### 环境在 reset 时输出 trading_info

Given:
- 环境刚执行 `reset()`
- 初始 position=0, single_holding_return_rate=0, single_holding_max_drawdown=0

When:
- 获取 `info["trading_info"]`

Then:
- `info["trading_info"]` = `np.array([0, 0, 0])`

### 环境在 step 正常路径输出 trading_info

Given:
- 当前持仓 position=4（多头 4 手）
- 这笔交易的 single_holding_return_rate=0.05（收益率 5%）
- 这笔交易的 single_holding_max_drawdown=0.02（最大回撤 2%）

When:
- 执行 `step(action)` 后获取 `info["trading_info"]`

Then:
- `info["trading_info"]` = `np.array([4, 0.05, 0.02])`

### 环境在爆仓路径输出 trading_info

Given:
- 爆仓发生，position 归零

When:
- 获取 `info["trading_info"]`

Then:
- position=0
- return_rate 和 max_drawdown 为爆仓前的最后值

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
- position=0, return_rate=0, max_drawdown=0

### trading_info 通过 replay buffer 传递

Given:
- replay buffer 的 info 键白名单包含 `trading_info`
- 环境返回的 info 中 `trading_info` = [position, return_rate, max_drawdown]

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

| 场景 | position | return_rate | max_drawdown |
|------|----------|-------------|--------------|
| 空仓 | 0 | 0 | 0 |
| 多头浮盈 | 4 | 0.05 | 0.02 |
| 多头浮亏 | 4 | -0.03 | 0.08 |
| 空头浮盈 | -4 | 0.02 | 0.01 |
| 空头浮亏 | -4 | -0.06 | 0.12 |

When:
- 模型接收不同的 `trading_info`

Then:
- 模型输出不同的 Q 值分布（因为 `fc_trading` 编码不同）
- 空仓时倾向于开仓动作，浮盈时倾向于持仓/平仓，浮亏且 max_drawdown 大时倾向于止损平仓

## Implementation Notes

### 涉及文件分类

#### 核心修改（4 个文件）

1. **`FineFT/env/env_class/base_env.py`**:
   - `reset()` 和 `step()` 的 info 字典中添加 `"trading_info"` 键
   - 值为 `np.array([position, single_holding_return_rate, single_holding_max_drawdown])`

2. **`FineFT/RL/util/replay_buffer_DQN.py`**:
   - 在所有 6 处 info 键白名单中添加 `"trading_info"`

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
  info["trading_info"] = np.array([position, single_holding_return_rate, single_holding_max_drawdown])
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

`TRADING_INFO_DIM=3` 默认值意味着 `fc_trading` 层始终存在。已保存的旧模型（无 `fc_trading` 权重）的 `state_dict` 与新模型不兼容，**需重新训练**。`load_state_dict` 时需使用 `strict=False` 或提供包含 `fc_trading` 权重的新 checkpoint。

## LANGUAGE.md Flag

新增术语建议：

**Trading Process Feature (交易过程特征)**:
从环境交易执行过程中产生的实时特征，包括 position（当前持仓量）、single_holding_return_rate（这笔交易的收益率）和 single_holding_max_drawdown（这笔交易的最大回撤），全部为比率或离散值、不使用绝对金额；打包为 `trading_info` 数组传入 Q 网络的 `fc_trading` 层编码；与 State Feature（市场技术指标）互补，State Feature 描述市场状态，Trading Process Feature 描述 agent 自身持仓状态和风险暴露。
_Avoid_: 交易特征、过程特征

## MAP.md Flag

`Stage I Low-level Training` 组件描述需补充：Q 网络输入从四路（state, previous_action, time, avaliable_action）扩展为五路，新增 `trading_info`（Trading Process Feature）输入通道。

`Futures Trading Environment` 组件描述需补充：`reset()`/`step()` 返回的 info 字典新增 `trading_info` 键。
