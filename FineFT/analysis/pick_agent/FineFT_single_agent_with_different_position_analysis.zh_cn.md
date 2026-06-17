# FineFT_single_agent_with_different_position.py 逻辑流程说明

本文档说明 `analysis/pick_agent/FineFT_single_agent_with_different_position.py` 的作用和执行流程。

该脚本处在 README 的阶段 II：

```text
阶段 I：训练低层 ensemble agent
阶段 II：回测低层 ensemble 的每个 context，并过滤/重组 ensemble
阶段 III：使用 VAE/启发式路由选择低层 agent
```

它本身不训练新策略，而是读取低层模型回测后的 `analysis_result.npy`，从不同 epoch、不同 context/bin、不同市场动态 label 中筛选表现较好的子 agent，最后拼成一个新的 potential ensemble model。

## 1. 输入与输出

### 1.1 输入

默认读取：

```text
result/DiHFT/low_level/<dataset_name>/<parameter>/epoch_<k>/analysis_result.npy
result/DiHFT/low_level/<dataset_name>/<parameter>/epoch_<k>/trained_model.pkl
dataset/<dataset_name>/state_features.npy
```

其中 `analysis_result.npy` 不是这个脚本生成的，而是由低层回测脚本生成：

```text
RL/DiHFT/low_level/test_agent_index.py
```

启动入口通常是：

```text
script/test/DiHFT/low_level/main.sh
script/test/DiHFT/low_level/test_util.sh
```

`test_agent_index.py` 会对每个 epoch 的 ensemble 模型做细粒度回测：

```text
市场动态 label_*
  x 初始动作 initial_action
  x ensemble context/bin_index
  x valid/label_*/df_*.feather 片段
```

每组条件保存：

```python
{
    "label": label,
    "initial_action": initial_action,
    "bin_index": bin_index,
    "reward_sum": [...],
    "df_length": [...],
    "turnover": [...],
}
```

字段含义：

- `label`：市场动态标签，例如 `label_0`。
- `initial_action`：初始动作，对应不同初始持仓。
- `bin_index`：ensemble 中第几个 context/Q 子网络。
- `reward_sum`：该 context 在多个验证片段上的累计 reward。
- `df_length`：每个验证片段长度。
- `turnover`：动作切换幅度的近似统计。

### 1.2 输出

脚本会输出三类结果。

筛选分析结果：

```text
analysis_result/DiHFT/low_level/<dataset_name>/result.csv
analysis_result/DiHFT/low_level/<dataset_name>/result_all.csv
analysis_result/DiHFT/low_level/<dataset_name>/best_index_info_by_dynamics_with_different_position.csv
```

重组后的 potential model：

```text
result/DiHFT/potential_model/<dataset_name>/model.pth
```

## 2. 参数说明

主要参数：

```python
--dataset_name       默认 BTCUSDT
--num_label          默认 5
--epoch_num          默认 50
--initial_position   默认 9
--save_path          默认 analysis_result/DiHFT/low_level
--model_save_path    默认 result/DiHFT/potential_model
--std_preference     默认 0.1
```

其中：

- `num_label=5` 表示有 `label_0` 到 `label_4` 五种市场动态。
- `initial_position=9` 实际表示初始动作数量，默认遍历 `0..8`。
- `epoch_num=50` 表示每个参数目录下扫描 `epoch_1` 到 `epoch_50`。
- `std_preference` 用来惩罚收益波动，选择时使用：

```text
score = trans_reward_mean - std_preference * trans_reward_std
```

## 3. 类初始化

核心类：

```python
class picker
```

初始化时构造：

```python
self.label_list = ["label_0", ..., "label_4"]
self.initial_position_list = range(9)
self.model_save_path = result/DiHFT/potential_model/<dataset_name>
```

这意味着脚本默认认为：

```text
每个数据集有 5 类市场动态
每个低层 agent 有 9 个动作/初始持仓状态
```

## 4. 单个 epoch 结果转换

函数：

```python
transform_single_epoch_result(result, epoch_path)
```

输入是一个 epoch 下的 `analysis_result.npy`。

它对每条记录做以下转换：

```python
normalized_reward = reward_sum / df_length
trans_reward_mean = mean(normalized_reward)
trans_reward_std = std(normalized_reward)
mean_turnover = mean(turnover)
```

然后删除原始列表字段：

```text
reward_sum
df_length
turnover
normalized_reward
```

并补充：

```text
epoch_path
```

### 为什么要除以 df_length

不同 `valid/label_*/df_*.feather` 片段长度可能不同。直接比较 `reward_sum` 会偏向更长片段，因此这里转成：

```text
单位长度 reward
```

也就是 `reward_sum / df_length`。

## 5. 单个 epoch 内筛选最佳 context

函数：

```python
pick_best_index_from_single_epoch(result, epoch_path)
```

它在单个 epoch 内，对每一种：

```text
label x initial_action
```

筛选最佳 `bin_index`。

选择标准：

```python
trans_reward_mean - std_preference * trans_reward_std
```

含义是：

```text
收益均值越高越好
收益波动越小越好
```

默认 `std_preference=0.1`，也就是对波动做较轻的惩罚。

输出是该 epoch 下每个 `label + initial_action` 的最佳记录。

## 6. 聚合一个参数目录

函数：

```python
conclude_single_parameter(parameter_path)
```

遍历：

```text
epoch_1
epoch_2
...
epoch_50
```

每个 epoch 调用：

```python
analysis_single_epoch(epoch_path)
```

返回两类结果：

- `single_parameter_result`：所有 context/bin 的完整转换结果。
- `single_parameter_best_result`：每个 epoch 内按 `label + initial_action` 选出的最佳结果。

## 7. 聚合所有参数目录

函数：

```python
conclude_all_parameter(root_path)
```

其中：

```text
root_path = result/DiHFT/low_level/<dataset_name>
```

它会遍历该目录下所有 parameter 子目录，例如：

```text
weights_advantage_pretrain
其他实验参数目录
```

对每个 parameter 调用 `conclude_single_parameter()`。

最终得到：

```text
all_parameter_result_all
all_parameter_result_best
```

## 8. 生成 result.csv 和 result_all.csv

函数：

```python
get_all_parameter_result()
```

会把聚合结果转成 DataFrame：

```python
df_best = pd.DataFrame(all_parameter_result_best)
df_all = pd.DataFrame(all_parameter_result)
```

其中：

- `result.csv`：每个 epoch 内、每个 `label + initial_action` 的最佳 context 结果。
- `result_all.csv`：所有 epoch、所有 context/bin 的完整结果。

`result_all.csv` 会按以下字段排序：

```text
epoch_number
label
initial_action
bin_index
```

输出路径：

```text
analysis_result/DiHFT/low_level/<dataset_name>/result.csv
analysis_result/DiHFT/low_level/<dataset_name>/result_all.csv
```

注意：后续真正用于重组 potential model 的是 `result_all.csv`，不是 `result.csv`。

## 9. 按市场动态筛选最终子 agent

函数：

```python
pick_best_agent_regarding_dynamics_bin_index_path(result_all)
```

这一步的目标是：

```text
为每个市场动态 label 选一个最适合的 context/bin 和对应 epoch 模型
```

对每个 `label`：

1. 取出该 label 的所有记录。
2. 按以下键分组：

```text
label, bin_index, epoch_path
```

3. 对每组计算 `trans_reward_mean` 的平均值。
4. 找到平均收益最高的一组。

最终得到：

```text
label
epoch_path
bin_index
reward_max
```

并保存到：

```text
best_index_info_by_dynamics_with_different_position.csv
```

### 这一步为什么叫 different_position

因为最终按 `label, bin_index, epoch_path` 分组时，没有把 `initial_action` 放进分组键。

也就是说，它是在所有初始动作/初始持仓条件上做平均，选择一个在不同初始持仓下整体表现较好的子 agent。

## 10. 重组 potential ensemble model

函数：

```python
create_potential_result(best_agent_df)
```

这一步会根据上一步筛出的结果，从不同 epoch 模型中抽取指定的 `qnet_list.<bin_index>` 子网络，再拼成一个新的 ensemble。

固定网络参数：

```python
n_state = len(dataset/<dataset_name>/state_features.npy)
n_action = 9
n_hidden = 128
time_info_dim = 2
```

待加载模型路径：

```python
epoch_path/trained_model.pkl
```

调用：

```python
create_new_ensemble_qnet_from_different_save_path(
    n_state,
    n_action,
    n_hidden,
    2,
    epoch_path_list,
    bin_index_list,
)
```

该函数会：

1. 新建一个 ensemble Q network。
2. 对每个被选中的 `epoch_path` 加载 `trained_model.pkl`。
3. 提取其中的 `qnet_list.<bin_index>` 参数。
4. 重命名为新 ensemble 中的 `qnet_list.<new_index>`。
5. 加载到新模型中。

保存结果：

```text
result/DiHFT/potential_model/<dataset_name>/model.pth
```

这个 `model.pth` 就是后续高层 routing/heuristic 可以使用的候选低层模型集合。

## 11. 主程序执行顺序

主程序：

```python
args = parser.parse_args()
p = picker(args)
single_parameter_result_best, single_parameter_result_all = p.get_all_parameter_result()
df = pd.read_csv("analysis_result/.../result.csv")
df_all = pd.read_csv("analysis_result/.../result_all.csv")
best_agent_info = p.pick_best_agent_regarding_dynamics_bin_index_path(df_all)
p.create_potential_result(best_agent_info)
```

完整流程：

```text
开始
 |
 v
解析参数
 |
 v
初始化 picker
 |
 v
扫描 result/DiHFT/low_level/<dataset_name> 下的所有参数目录
 |
 v
对每个参数目录扫描 epoch_1..epoch_50
 |
 v
读取每个 epoch 的 analysis_result.npy
 |
 v
计算 trans_reward_mean / trans_reward_std / mean_turnover
 |
 v
保存 result.csv 与 result_all.csv
 |
 v
按 label 聚合 result_all
 |
 v
为每个 label 选择平均 trans_reward_mean 最高的 epoch_path + bin_index
 |
 v
保存 best_index_info_by_dynamics_with_different_position.csv
 |
 v
从多个 trained_model.pkl 中抽取被选中的 qnet 子网络
 |
 v
拼成新的 ensemble Q network
 |
 v
保存 result/DiHFT/potential_model/<dataset_name>/model.pth
 |
 v
结束
```

## 12. 关键理解

这个脚本不是训练器，而是一个低层 agent 过滤器。

它做了两层选择：

第一层是在单个 epoch 内，对每个 `label + initial_action` 找到收益高且波动较低的 context。

第二层是在所有 epoch、所有初始动作上，对每个 `label` 找到平均表现最好的 `epoch_path + bin_index`。

最终得到的不是一个全新训练出来的模型，而是：

```text
从多个已训练 ensemble 模型中挑出若干优秀子网络，再重新组装成一个新的 ensemble
```

这就是 README 中“过滤集成模型”的含义。

## 13. 需要注意的实现细节

1. `epoch_num` 默认是 50，如果某个参数目录缺少 `epoch_<k>/analysis_result.npy`，脚本会直接报错。

2. `n_action` 在 `create_potential_result()` 中写死为 9。如果训练时动作空间不是 9，需要同步修改。

3. `n_hidden` 写死为 128。如果低层模型训练时使用了不同 hidden size，也需要同步修改。

4. 最终筛选 `best_index_info_by_dynamics_with_different_position.csv` 时只按 `trans_reward_mean` 均值最大选，没有继续使用 `std_preference` 惩罚。

5. `label_list = best_agent_df["label"].unique().tolist()`，而 `epoch_path_list` 和 `bin_index_list` 来自 DataFrame 原顺序。通常这里每个 label 一行，所以长度一致；如果输入表中 label 重复，可能触发 assert 或造成顺序语义不清。

