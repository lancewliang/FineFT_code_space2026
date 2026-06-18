# RL/DiHFT/high_level/vae_routing_optuna.py 逻辑流程说明

本文档说明 `RL/DiHFT/high_level/vae_routing_optuna.py` 的执行流程。这个脚本本身不直接训练交易策略，而是用 Optuna 在验证集上搜索 VAE 路由模块的超参数，并把每次 trial 的回测结果汇总保存下来。

整体流程如下：

```text
命令行参数
    |
    v
读取原始 trading 参数 parser + 搜索范围 parser_all
    |
    v
tune(args_1, args_2)
    |
    +-- 固定随机种子
    +-- 覆盖 dataset_name / max_holding_number
    +-- 定义 Optuna objective
    |       |
    |       +-- 采样 window_length / gamma / rule_base_threshold
    |       +-- 构造 vae_risk_aware_routing
    |       +-- 调用 test() 得到 return_rate
    |
    +-- create_study(maximize)
    +-- optimize(objective, n_trials=128, n_jobs=16)
    +-- 导出 trials_dataframe 到 CSV
```

## 1. 脚本定位

这个脚本的职责只有三件事：

1. 设定 Optuna 的搜索空间。
2. 反复调用 `vae_risk_aware_routing.test()` 评估候选参数。
3. 保存搜索结果表格，供后续人工挑选参数。

真正的路由逻辑不在本文件，而是在：

- [`RL/DiHFT/high_level/vae_routing_util.py`](../FineFT/RL/DiHFT/high_level/vae_routing_util.py)

那里完成 VAE 分数统计、阈值判断、规则关闭/低层网络动作选择、以及回测结果落盘。

## 2. 参数分层

脚本里有两套参数来源。

### 2.1 `args_1`

来自 `vae_routing_util.py` 里的 `parser`，主要是运行交易/回测所需的完整配置，例如：

- 数据路径
- 交易对名称
- 初始仓位和杠杆
- 低层模型路径
- VAE 模型路径
- `window_length`
- `gamma`
- `rule_base_threshold`

### 2.2 `args_2`

来自本文件的 `parser_all`，只负责超参搜索范围：

- `--window_length_min / max`
- `--gamma_min / max`
- `--rule_base_threshold_min / max`
- `--dataset_name`
- `--max_holding_number`

也就是说：

```text
args_1 = “真实执行参数”
args_2 = “搜索空间参数”
```

## 3. 入口流程

主入口在文件末尾：

```python
if __name__ == "__main__":
    from RL.DiHFT.high_level.vae_routing_util import parser

    args_1 = parser.parse_args()
    args_2 = parser_all.parse_args()
    tune(args_1, args_2)
```

执行顺序很直接：

1. 解析 `vae_routing_util.py` 里的基础参数。
2. 解析本文件定义的搜索参数。
3. 进入 `tune()`。

## 4. `tune()` 的主流程

`tune(args_1, args_2)` 是整个脚本的核心。

### 4.1 固定随机种子

先调用：

```python
seed_torch(12345)
```

它同时固定了 Python、NumPy、PyTorch 和 CUDA 的随机性，目的是让不同 trial 的结果更可复现。

### 4.2 覆盖基础参数

脚本会把搜索用的交易对和仓位上限写回 `args_1`：

```python
args_1.dataset_name = args_2.dataset_name
args_1.max_holding_number = args_2.max_holding_number
```

这一步保证后续 trial 都在同一个数据集和同一仓位尺度下比较。

### 4.3 定义 objective

`objective(trail)` 的职责是评价一个参数组合的好坏。

每个 trial 会做这些事：

1. 按 `trail.number % torch.cuda.device_count()` 分配 GPU。
2. 采样 `window_length`。
3. 采样 `gamma`。
4. 采样 `rule_base_threshold`。
5. 构造 `vae_risk_aware_routing(args_1)`。
6. 调用 `test()`。
7. 把 `test()` 返回的 `return_rate` 作为 Optuna 优化目标。

这里的优化方向是：

```python
optuna.create_study(direction="maximize")
```

所以它在找“回测表现最好的参数组”。

## 5. 单次 trial 的内部逻辑

`objective()` 实际上只是把参数喂给 `vae_risk_aware_routing`。真正的评估过程在 `vae_risk_aware_routing.test()` 里。

### 5.1 初始化阶段

`vae_risk_aware_routing.__init__()` 会：

1. 选择设备 `cuda:<gpu_index>` 或 `cpu`。
2. 创建结果目录：

```text
result/DiHFT/high_level/<dataset_name>/vae_risk_aware_routing/gamma_<g>_window_<w>_threshold_<t>/
```

3. 读取验证集 `valid.feather`。
4. 读取 `state_features.npy` 和 `maintenance_margin_ratio_dict.npy`。
5. 加载低层 ensemble Q 网络 `model.pth`。
6. 加载多个 VAE 模型：

```text
label_0 ... label_<label_number-1>
```

7. 读取各 label 的 in-distribution `log p(x)` 分布文件 `id_logpx.npy`。
8. 为每个 label 建一个固定长度的 quantile deque。

### 5.2 初始 warm-up

`test()` 开始后会先构造环境：

```python
env = initiate_base_env(...)
```

然后调用 `initial_rollout()`，连续执行 `window_length` 步 `rule_based_close()`。

这个阶段的作用是：

- 先把量化窗口填满；
- 让每个 label 的 quantile deque 有足够历史；
- 避免一开始就用空窗口做风险判断。

### 5.3 每步路由决策

warm-up 结束后，主循环每一步都会：

1. 对当前状态 `s` 计算每个 VAE 的重构损失。
2. 把损失映射到对应 label 的 in-distribution 分位数。
3. 对最近 `window_length` 个分位数做指数衰减平均，衰减系数就是 `gamma`。
4. 得到每个 label 的风险权重。
5. 比较 `max(weights)` 和 `rule_base_threshold`：

   - 如果都低于阈值，走规则关闭分支；
   - 否则选择 `argmax(weights)` 对应的 label，再交给低层 ensemble Q 网络选动作。

### 5.4 两条分支

#### 规则分支

如果 `max(weights) < rule_base_threshold`：

- 且 `single_holding_max_drawdown < 0.05` 时，沿用上一个动作；
- 否则调用 `rule_based_close()` 强制收敛到更安全的平仓/降仓动作。

这条路径本质上是在低置信度或高回撤时，把控制权切回规则策略。

#### 模型分支

如果 `max(weights) >= rule_base_threshold`：

- 选出最大权重对应的 `selected_agent_index`；
- 用这个 label 对应的低层 Q 值切片；
- 在可用动作中取最大 Q 值动作。

也就是说，高层 VAE 先做“场景路由”，低层 ensemble 再做“具体交易动作”。

## 6. 回测结果与评分

整段 episode 结束后，`test()` 会保存一批数组：

- `reward_history.npy`
- `total_asset_history.npy`
- `micro_action_history.npy`
- `trading_info.npy`
- `initial_margin_history.npy`
- `wallet_balance_history.npy`
- `unrealized_pnl_history.npy`
- `maintain_marigine_history.npy`
- `new_position_required_money_history.npy`
- `macro_action.npy`

然后计算：

```python
return_rate = reward_sum / (require_money + 1e-12)
```

这个值就是 Optuna 的目标函数返回值。

## 7. Optuna 输出

所有 trial 完成后，脚本会：

1. 打印总 trial 数。
2. 打印最佳参数：

```python
study.best_trial.params
```

3. 导出完整 trial 表：

```text
result/DiHFT/high_level/<dataset_name>/vae_risk_aware_routing_optuna/optuna_results.csv
```

这份 CSV 里包含每次 trial 的参数和结果，方便后续人工筛选。

## 8. 运行方式

对应脚本一般是：

```text
python RL/DiHFT/high_level/vae_routing_optuna.py
```

仓库里也有 shell 封装：

- [`script/test/DiHFT/high_level/vae_optuna.sh`](../FineFT/script/test/DiHFT/high_level/vae_optuna.sh)

## 9. 实现上的几个注意点

1. `objective()` 直接修改共享的 `args_1`，而 `study.optimize(..., n_jobs=16)` 会并行跑 trial，所以这个实现对共享参数对象比较敏感。
2. `gpu_id = trail.number % torch.cuda.device_count()` 默认依赖 CUDA 可用且至少有一张卡。
3. `gamma` 使用了 `suggest_float(..., log=True)`，因此搜索更偏向乘法尺度上的变化。
4. 每个 trial 都会重新实例化 `vae_risk_aware_routing`，因此开销主要来自模型加载和整段验证集回测。

