# weight_advantage_pretrain.py 逻辑流程说明

本文档说明 `RL/DiHFT/low_level/weight_advantage_pretrain.py` 的训练逻辑。该脚本用于训练 DiHFT 的低层交易智能体，输出目录为：

```text
result/DiHFT/low_level/<dataset_name>/weights_advantage_pretrain/
```

对应启动脚本是：

```text
script/train/DiHFT/low_level/advantage.sh
```

## 1. 脚本定位

`weight_advantage_pretrain.py` 训练的是一个带多个 context/sub-policy 的 ensemble DQN。

核心类：

```python
class Weighted_Contexts_DQN
```

核心思想可以概括为：

```text
训练数据分块 train/df_*.feather
        |
        v
构造期货交易环境 Demo_Env
        |
        v
前若干轮使用专家/规则策略预训练
        |
        v
之后让多个 context 分别探索训练
        |
        v
用 TD loss + 专家 Q 分布 KL loss 更新 ensemble Q 网络
        |
        v
按 epoch 保存 trained_model.pkl
```

## 2. 输入与依赖数据

脚本默认从 `dataset/<dataset_name>/` 读取数据：

```text
dataset/<dataset_name>/train/df_*.feather
dataset/<dataset_name>/state_features.npy
dataset/<dataset_name>/maintenance_margin_ratio_dict.npy
```

其中：

- `train/df_*.feather`：训练分块数据，每次训练随机抽一个分块。
- `state_features.npy`：状态特征列名列表。
- `maintenance_margin_ratio_dict.npy`：不同名义价值对应的维持保证金参数。

`advantage.sh` 为不同交易对设置不同的 `max_holding_number`：

```text
BTCUSDT: 8
ETHUSDT: 160
BNBUSDT: 100
DOTUSDT: 6000
```

## 3. 可调参数与含义

脚本通过 `argparse` 暴露训练参数，启动时可用命令行覆盖默认值。例如：

```bash
python RL/DiHFT/low_level/weight_advantage_pretrain.py \
    --dataset_name BTCUSDT \
    --max_holding_number 8 \
    --batch_size 512
```

在当前仓库的期货训练脚本中，通常先进入 `finetf` 环境并设置 `PYTHONPATH`：

```bash
conda activate finetf
export PYTHONPATH="${ROOTPATH}/FineFT${PYTHONPATH:+:${PYTHONPATH}}"
python -u FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py ...
```

### 3.1 Replay Buffer 参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--buffer_size` | `1000000` | 每个 replay buffer 最多保存的 transition 数量。脚本会分别创建 `buffer_pretrain` 和 `buffer_diverse`。 |
| `--n_step` | `1` | n-step TD 的步数。大于 1 时，buffer 会把连续多步 reward 按 `gamma` 折扣累计。 |

### 3.2 数据与交易环境参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--base_path` | `dataset` | 数据根目录。脚本会读取 `<base_path>/<dataset_name>/train/df_*.feather`、`state_features.npy` 和 `maintenance_margin_ratio_dict.npy`。 |
| `--dataset_name` | `BTCUSDT` | 数据集或交易对名称，同时影响输入数据目录、日志目录和模型输出目录。 |
| `--max_holding_number` | `8` | 最大持仓数量，用来生成离散持仓档位。不同交易对通常需要单独设置。 |
| `--order_book_depth` | `25` | 环境和专家 Q 表使用的订单簿深度。数据特征需要和该深度匹配。 |
| `--position_choices` | `9` | 离散持仓档位数量。代码假设它是奇数，因为中间档位表示空仓。 |
| `--leverage_choices` | `[5]` | 可选杠杆列表。该参数使用 `action="append"`，多次传入会追加，例如 `--leverage_choices 3 --leverage_choices 5`。代码注释建议通常只使用一个杠杆值。 |
| `--long_estimated_rate` | `0.0005` | 多头 funding/持仓估计费率，传入环境和专家 Q 表计算。 |
| `--short_estimated_rate` | `0` | 空头 funding/持仓估计费率，传入环境和专家 Q 表计算。 |
| `--transcation_cost` | `0.0002` | 交易手续费率。参数名沿用代码中的拼写 `transcation_cost`。 |
| `--early_stop` | `2160` | 环境单个 episode 的提前停止步数上限。 |
| `--initial_wallet_balance` | `100000.0` | 初始钱包余额，用于构造环境初始账户状态。 |
| `--initial_margin` | `0` | 初始保证金。训练循环里会根据随机初始动作和首个价格重新计算该值。 |
| `--initial_unrealized_pnL` | `0` | 初始未实现盈亏。 |
| `--initial_position` | `0` | 初始持仓。训练循环里会根据随机初始动作重新设置。 |
| `--initial_leverage` | `5` | 初始杠杆。训练循环里会根据随机初始动作重新设置。 |

动作空间由 `position_choices` 和 `leverage_choices` 共同决定：

```text
N_ACTIONS = (position_choices - 1) * len(leverage_choices) + 1
```

因此增加杠杆选项会直接扩大 Q 网络输出维度和环境动作空间。

### 3.3 网络结构参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--hidden_nodes` | `128` | 每个子 Q 网络的隐藏层宽度。 |
| `--N` | `7` | ensemble 中的 context/sub-policy 数量。正式训练阶段每个 sample 会让这 `N` 个 context 分别采样一条 episode。 |
| `--time_info_dim` | `2` | 时间信息输入维度。当前代码拼接 funding 倒计时的 hour 和 minute，因此默认是 2。 |

### 3.4 强化学习训练参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--tau` | `0.005` | target network 软更新系数。每次更新后执行 `target = tau * eval + (1 - tau) * target`。 |
| `--batch_size` | `512` | 每次从 replay buffer 采样的 transition 数量。 |
| `--update_times` | `1` | 每次满足更新条件后连续执行的梯度更新次数。 |
| `--gamma` | `0.9` | TD target 和 n-step return 的 reward 折扣因子。注意预训练阶段专家 Q 表构造时单独使用 `gamma=1`。 |
| `--epsilon_init` | `1` | 正式训练阶段 epsilon-greedy 的初始随机探索概率。 |
| `--epsilon_min` | `0.1` | epsilon 衰减下限。 |
| `--epsilon_step` | `100000.0` | epsilon 从 `epsilon_init` 线性衰减到 `epsilon_min` 需要的步数尺度。每次 `index == 0` 采样动作后衰减一次。 |
| `--rollout_steps` | `1024` | 采样步数间隔。满足 buffer 足够大且 `step_counter % rollout_steps == 1` 时触发更新。 |

### 3.5 学习率与总训练轮数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--lr_init` | `0.005` | Adam 优化器初始学习率。 |
| `--lr_min` | `0.0001` | 学习率线性衰减下限。 |
| `--lr_step` | `20000.0` | 学习率从 `lr_init` 衰减到 `lr_min` 需要的步数尺度。每次 `index == 0` 采样动作后衰减一次。 |
| `--num_sample` | `400` | 外层采样轮数，即 `for sample in range(num_sample)` 的总次数。 |
| `--seed` | `12345` | Python、NumPy、PyTorch 和 CUDA 的随机种子。 |

### 3.6 输出路径参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--result_path` | `result/DiHFT/low_level` | 模型和 TensorBoard 日志输出根目录。实际模型目录为 `<result_path>/<dataset_name>/weights_advantage_pretrain/`。 |

脚本还会把普通日志写到：

```text
log_futures/<dataset_name>/low_level/train/advantage.log
```

### 3.7 损失函数参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--outer_bond` | `4` | `calculate_partial_loss()` 的 context 选择范围参数，用于控制 TD error 动态加权时的外侧边界。 |
| `--reachout_index` | `1` | `calculate_partial_loss()` 的范围扩展参数，用于在最匹配 context 附近额外扩展可学习区域。 |
| `--if_use_hubber_loss` | `True` | 是否对 TD error 使用 Huber loss。参数名沿用代码拼写 `hubber`。注意该参数使用 `type=bool`，命令行传入字符串时 Python 会按 `bool("False") == True` 解析；如果要关闭，建议改代码或把参数定义改成明确的布尔开关。 |

### 3.8 专家监督与预训练参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--ada_init` | `256` | 专家 KL loss 的初始权重。总损失中使用 `loss = td_loss + ada * KL_div` 或 `partial_td_error_loss + ada * KL_div`。 |
| `--ada_min` | `0` | 专家 KL loss 权重的衰减下限。 |
| `--ada_step` | `500000.0` | `ada` 从 `ada_init` 衰减到 `ada_min` 需要的步数尺度。每次 `index == 0` 采样动作后衰减一次。 |
| `--pretrain_epoch` | `2` | 外层 sample 中用于预训练的轮数。`sample < pretrain_epoch` 时使用专家/规则策略采样。 |

### 3.9 代码内固定参数

以下参数没有暴露为命令行参数，当前只能改代码调整：

| 名称 | 当前值 | 位置与含义 |
| --- | --- | --- |
| `grad_clip` | `5` | `Weighted_Contexts_DQN.__init__()` 中固定设置，传给 `update_params(..., grad_cliping=5)`，用于梯度裁剪。 |
| `epoch_number` | `4` | `train()` 中固定设置。每累计 4 个 sample 聚合一次 epoch 指标并保存一次模型。 |
| `group_number` | `self.N` | `train()` 中赋值但当前未被后续逻辑使用。 |
| `max_punishment` | `1e10` | 构造环境和专家 Q 表时传入，用于不可行动作或极端惩罚值。 |
| 预训练规则策略数量 | `4` | 预训练阶段固定执行专家最优、持多、持空、空仓四种策略。 |

## 4. 初始化流程

初始化发生在 `Weighted_Contexts_DQN.__init__()` 中。

### 4.1 随机种子与设备

先调用 `seed_torch()` 固定 Python、NumPy、PyTorch、CUDA 随机种子。随后根据 CUDA 是否可用选择：

```python
self.device = "cuda" or "cpu"
```

### 4.2 日志与模型路径

模型保存路径：

```text
result/DiHFT/low_level/<dataset_name>/weights_advantage_pretrain/
```

TensorBoard 日志路径：

```text
result/DiHFT/low_level/<dataset_name>/weights_advantage_pretrain/log/
```

### 4.3 动作空间

低层动作由“目标持仓 + 杠杆”组成。

代码中：

```python
self.N_ACTIONS = (position_choices - 1) * len(leverage_choices) + 1
```

默认：

```text
position_choices = 9
leverage_choices = [5]
N_ACTIONS = 9
```

`position_list` 会按 `max_holding_number` 均匀生成负仓、空仓、正仓。例如 `max_holding_number=8` 且 `position_choices=9` 时，持仓档位大致是：

```text
[-8, -6, -4, -2, 0, 2, 4, 6, 8]
```

### 4.4 网络结构

使用 `model.low_level.ensemble_Qnet`：

```python
self.eval_net = ensemble_Qnet(..., ensemble_number=self.N)
self.target_net = copy.deepcopy(self.eval_net)
```

默认 `N=7`，表示 ensemble 中有 7 个 Q 网络，也可以理解为 7 个 context/sub-policy。

每个子 Q 网络输入包括：

- 市场状态 `state_features`
- 距离 funding 的时间信息：小时、分钟
- 上一步动作 `previous_action`
- 当前可用动作 mask `avaliable_action`

输出形状：

```text
batch_size x N x N_ACTIONS
```

也就是每个 context 都输出一组动作 Q 值。

## 5. 交易环境构造

每次训练 sample 会随机读取一个训练分块：

```python
df_index = random.choices(range(self.total_df_index_length), k=1)[0]
self.train_df = pd.read_feather("train/df_<df_index>.feather")
```

然后随机选择初始动作，并把它映射为初始持仓和杠杆：

```python
initial_action = random.choices(range(self.position_choices), k=1)[0]
self.initial_position, self.initial_leverage = map_action_to_position_leverage(...)
```

再根据初始价格计算初始保证金：

```python
self.initial_margin = abs(initial_position * current_markprice / initial_leverage)
```

之后调用 `initiate_demo_env()` 创建 `Demo_Env`。

`Demo_Env` 继承 `Base_Env`，额外会创建一个动态规划得到的专家 Q 表：

```python
self.q_table = create_optimal_q_table(...)
```

环境每一步返回的 `info` 中包含：

- `previous_action`：当前状态对应的上一动作。
- `avaliable_action`：可用动作 mask。
- `avaiable_action_list`：可用动作列表。
- `funding_count_down_hour/minute`：距离 funding 的时间。
- `q_value`：专家动态规划 Q 表中当前状态对应的动作价值。

注意代码中同时存在拼写：

```text
avaliable_action
avaiable_action_list
```

这是原代码命名，不是文档笔误。

## 6. Replay Buffer

脚本使用两个 `Multi_step_ReplayBuffer_multi_info`：

```python
buffer_pretrain
buffer_diverse
```

分别用于：

- `buffer_pretrain`：预训练阶段收集专家/规则策略轨迹。
- `buffer_diverse`：正式训练阶段收集各 context 自己探索得到的轨迹。

buffer 保存的是：

```text
state, info, action, reward, next_state, next_info, done
```

其中 `info` 和 `next_info` 会保留训练所需的辅助字段，例如 `q_value`、`previous_action`、`avaliable_action`、funding 倒计时等。

如果 `n_step > 1`，buffer 会把连续多步 reward 折扣累计成 n-step return；默认 `n_step=1`。

## 7. 训练主循环

训练入口：

```python
trainer.train()
```

主循环：

```python
for sample in range(self.num_sample):
    pretrain = sample < self.pretrain_epoch
```

默认：

```text
num_sample = 400
pretrain_epoch = 2
```

因此前 2 个 sample 是预训练阶段，之后进入正式训练阶段。

## 8. 预训练阶段

预训练阶段的目标是先让 ensemble 网络学习一些“合理动作”和专家 Q 分布，避免一开始完全随机。

### 8.1 构造专家最优动作序列

预训练时会基于当前训练分块重新计算一个动态规划 Q 表：

```python
q_table = create_optimal_q_table_from_df(...)
self.perfection_action_list = get_dp_action_from_qtable(q_table, initial_action)
```

`get_dp_action_from_qtable()` 会从初始动作出发，每一步选取专家 Q 表中的最大 Q 动作，形成一条 perfect action path。

### 8.2 4 种规则策略采样

预训练阶段固定跑 4 种策略：

```python
for index in range(4):
```

`act_multi_styles_pretrain()` 中定义：

```text
index = 0: 专家 DP 最优动作
index = 1: buy and hold，尽量保持最大多头
index = 2: sell and keep，尽量保持最大空头
index = 3: empty position，尽量保持空仓
```

这些策略产生的 transition 会写入 `buffer_pretrain`。

### 8.3 预训练更新条件

当 buffer 中步数足够，并且达到 rollout 间隔时开始更新：

```python
step_counter_pretrain > batch_size * update_times + n_step
step_counter_pretrain % rollout_steps == 1
```

默认：

```text
batch_size = 512
update_times = 1
rollout_steps = 1024
```

每次从 `buffer_pretrain` 采样 batch，调用：

```python
self.update_pretrain(...)
```

## 9. 预训练损失 update_pretrain()

`update_pretrain()` 包含两部分损失：

```text
loss = td_loss + ada * KL_div
```

### 9.1 TD loss

当前网络输出：

```text
current_sa_quantiles: batch_size x N
```

目标值来自 target network：

```text
target = reward + (1 - done) * gamma * target_net_best_q(next_state)
```

然后使用 SmoothL1Loss：

```python
td_loss = SmoothL1Loss(current_sa_quantiles, target_sa_quantiles)
```

### 9.2 专家 KL loss

网络输出所有 context 的动作分布：

```text
predict_action_distribution: batch_size x N x N_ACTIONS
```

预训练阶段先给每个 context 同等权重：

```python
batch_weights = torch.ones(batch_size, N)
weighted_action_distribution = einsum("ijk,ij->ik", ...)
```

然后和环境 `info["q_value"]` 中的专家 Q 分布做 KL divergence：

```python
KL_div = kl_div(policy_distribution, expert_q_distribution)
```

这里会先用 `recalculate_q_demonstration()` 对不可用动作施加极大惩罚，使不可用动作不会被专家分布鼓励。

### 9.3 参数更新

更新流程：

```text
optimizer.zero_grad()
loss.backward()
gradient clipping
optimizer.step()
soft update target_net
```

target network 使用软更新：

```python
target = tau * eval + (1 - tau) * target
```

默认 `tau=0.005`。

## 10. 正式训练阶段

预训练结束后，进入 diverse training。

代码中：

```python
for index in range(self.N):
```

默认 `N=7`，因此每个 sample 会让 7 个 context 分别跑一条 episode。

每个 context 的动作选择：

```python
a = self.act_multi_styles(s, info, self.epsilon, index)
```

实际调用的是：

```python
act_single_context(state, info, context_index, epsilon)
```

选择逻辑：

```text
以 1 - epsilon 的概率：选该 context 下 Q 值最大的动作
以 epsilon 的概率：从可用动作列表中随机选动作
```

只有 `index == 0` 时会衰减：

- `epsilon`
- `ada`
- `lr`

默认衰减参数：

```text
epsilon: 1 -> 0.1
ada: 256 -> 0
lr: 5e-3 -> 1e-4
```

这意味着训练早期更依赖专家 KL 约束，后期逐渐减少专家约束，让 TD 学习占主导。

正式阶段采集的 transition 写入 `buffer_diverse`，满足条件后调用：

```python
self.update(...)
```

## 11. 正式训练损失 update()

正式训练同样包含：

```text
loss = partial_td_error_loss + ada * KL_div
```

和预训练不同的是，正式训练不再简单平均所有 context，而是根据 TD error 动态计算 context 权重。

### 11.1 计算 TD error 矩阵

当前动作 Q：

```text
current_sa_quantiles: batch_size x N x 1
```

目标 Q：

```text
target_sa_quantiles: batch_size x 1 x N
```

相减后得到：

```text
td_errors: batch_size x N x N
```

可以理解为：当前 N 个 context 与目标 N 个 context 两两组合后的 TD error。

### 11.2 calculate_partial_loss()

`calculate_partial_loss()` 会：

1. 找到每个样本中 TD error 较合适的 context 区域。
2. 根据 `outer_bond` 和 `reachout_index` 扩展一个可接受范围。
3. 生成 `batch_weights` 和 TD loss 权重矩阵。
4. 得到 `partial_td_error_loss`。

默认：

```text
outer_bond = 4
reachout_index = 1
```

直观上，它不是强制所有 context 都平均学习，而是让更匹配当前样本风险/收益形态的 context 权重更高。

### 11.3 加权动作分布与专家 KL

正式阶段用 `batch_weights` 对多个 context 的动作输出加权：

```python
weighted_action_distribution = torch.einsum(
    "ijk,ij->ik", predict_action_distribution, batch_weights
)
```

然后继续与专家动态规划 Q 分布做 KL 约束：

```python
KL_div = F.kl_div(...)
```

随着 `ada` 衰减，KL 约束会逐渐变弱。

## 12. 日志与模型保存

训练过程中写入 TensorBoard：

```text
total_loss
KL_loss
td_loss
return_rate_train_<index>
reward_sum_train_<index>
epoch_return_rate_train
epoch_final_balance_train
epoch_reward_sum_train
```

代码中设置：

```python
epoch_number = 4
```

每累计 4 个 sample，会保存一次模型：

```text
result/DiHFT/low_level/<dataset_name>/weights_advantage_pretrain/epoch_<k>/trained_model.pkl
```

保存的是：

```python
self.eval_net.state_dict()
```

## 13. 整体流程图

```text
开始
 |
 v
解析参数
 |
 v
初始化 Weighted_Contexts_DQN
 |
 +--> 固定随机种子
 +--> 读取 state_features / 保证金配置
 +--> 构造 position_list / action space
 +--> 创建 ensemble eval_net 与 target_net
 +--> 创建 optimizer / TensorBoard writer
 |
 v
for sample in num_sample:
 |
 +--> 随机选择 train/df_i.feather
 +--> 随机选择初始动作
 +--> 构造 Demo_Env
 |
 +--> sample < pretrain_epoch ?
 |       |
 |       +-- 是:
 |       |     +--> 生成专家 Q table / perfect action list
 |       |     +--> 运行 4 种规则策略
 |       |     +--> transition 写入 buffer_pretrain
 |       |     +--> 满足条件后 update_pretrain()
 |       |
 |       +-- 否:
 |             +--> 依次运行 N 个 context policy
 |             +--> epsilon-greedy 选动作
 |             +--> transition 写入 buffer_diverse
 |             +--> 满足条件后 update()
 |
 +--> 写 TensorBoard 指标
 +--> 每 4 个 sample 保存一次 trained_model.pkl
 |
 v
结束
```

## 14. 关键理解

这个脚本的名字里有 `advantage_pretrain`，但代码主体不是传统 Dueling DQN 的 advantage head，而是：

```text
多个 context 的 ensemble Q 网络
+ 专家动态规划 Q 分布约束
+ 预训练阶段的规则/专家轨迹
+ 正式阶段的 TD-error 动态加权
```

其中 `weight` 主要体现在正式训练阶段：通过 TD error 计算 `batch_weights`，再用它对不同 context 的动作输出加权。

`pretrain` 体现在训练前几个 sample：先用专家最优、持多、持空、空仓四种策略生成经验，并用 TD loss + 专家 KL loss 预热网络。
