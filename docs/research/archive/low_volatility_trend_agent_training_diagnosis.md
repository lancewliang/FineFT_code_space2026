# 30 分钟商品期货低波动趋势 Regime 训练失效研究报告

> 研究日期：2026-08-27  
> 适用对象：`fu / 30min / DiHFT low-level agent`  
> 目的：保存本次研究的方法、过程数据、证据边界和实验方案，使下一次可以按同一口径复算，而不是只复用结论。

## 1. 结论先行

当前证据不支持“低波动强趋势样本整体太少”或“`ada=256` 导致 KL 项统治预训练”作为首要原因。更符合数据的解释是：

1. **目标形态在训练集并不稀缺，但独立连续片段有限。** 因果 48-bar 口径下，低两档波动率与最强负/正斜率四格合计占训练可用步的 **19.39%**；其中最稀缺的 `vol=0, slope=0` 仍有 423 步，但只有 39 个连续 run。
2. **状态表示缺少稳定、直接的“有方向趋势 ÷ 噪声”锚点。** 当前 150 个特征虽包含短期价差、趋势和波动率特征，却没有明确的长窗口有符号回归斜率、趋势效率、趋势回归 $R^2$、趋势/波动比和波动率 regime 位置；若让 MLP 间接拼出这些量，训练会更依赖合约与幅度。
3. **训练采样没有显式平衡 16 个市场状态。** 训练日程只平衡 `df_index × initial_action`，buffer 内又是均匀随机抽 transition；因而不会保护稀缺的独立 run，也不会保证 16 格覆盖。
4. **目标函数与 teacher 会鼓励频繁追逐局部最优。** reward 是原始权益变化，teacher 是用完整未来路径反向动态规划生成的 Q-table；研究时的 q-table 诊断显示 teacher 几乎不空仓且换仓率约为步数的 25%–30%。这与低波动持久趋势中“少换仓、覆盖成本”的需要不完全一致。
5. **train→valid 的总体状态分布漂移轻微，而不是全局断层。** 用训练阈值映射 valid 得到 PSI=**0.0761**；但若干单格比例变化明显，且多个当前特征的 valid IC 衰减或反号，因此主要风险是条件稳定性和表示，而不是整体分布缺失。
6. **验证选择结果证明问题是真实且范围更广。** 273 个完整候选中，低波动两档覆盖的全部 8 个斜率格都回退为 `empty_model`；用户关注的强负/正斜率四个目标格，最优被拒候选仍未通过 `mean_return,lcb,worst_initial_position`。

因此，优先级应为：**增加少量因果 Regime Anchor 特征并做条件稳定性筛选 → 加入交易成本感知的动作滞回 → 再做温和的 run-aware 重采样 → 最后才调整 teacher KL。** 不建议一开始把 16 格强行采成严格均匀，也不建议仅调低 `ada_min` 后认定问题已经解决。

## 2. 研究问题与术语边界

### 2.1 研究问题

本报告回答四个问题：

1. 训练数据中，波动率四分位 × 斜率四分位的 16 个二维组合各有多少步、多少合约、多少连续 run？
2. “高斜率、低波动”是否因为样本过少而训练失败？高波动训练表现不佳是否由 reward 尺度主导？
3. teacher、KL、采样、网络输入和当前选择特征分别提供了什么证据？
4. 下一轮应如何按最小风险的消融顺序验证改良方案？

### 2.2 两套 16 格不可混为一谈

本研究同时出现两种“16 格”：

- **因果训练诊断格**：每个时点只使用当时及之前 48 根 bar，计算滚动斜率和滚动波动率，再用训练集四分位阈值分箱。它可用于训练采样和在线特征。
- **离线验证选择格**：当前二维 selector 使用已有的波动率/斜率切片和 run 结果选择 epoch/bin；其标签依赖既有离线切片流程。选择表定义了 `slot = volatility_index * 4 + slope_index`，证据见 [selection manifest](../../analysis_result/DiHFT/low_level/fu/30min_multi/two_dimensional_selection/two_dimensional_selection_manifest.json)。

两套格子的语义相近但边界不相同。本文用前者解释训练数据覆盖，用后者证明候选模型验证失败；**不得把两张表逐格当作同一批时点做因果归因**。若下次需要严格 apples-to-apples，应在 train 上复用 selector 的同一切片实现，并只用 train 拟合任何阈值。

### 2.3 方向标签

斜率从低到高映射为：

- `slope=0`：最强负斜率；
- `slope=1`：弱负/近零；
- `slope=2`：弱正；
- `slope=3`：最强正斜率。

波动率 `vol=0..3` 从低到高。因此本报告的核心目标格是 `(v0,s0) (v0,s3) (v1,s0) (v1,s3)`。

## 3. 证据分级与可复现环境

### 3.1 当前仓库可直接复核的一手资料

| 类别 | 路径 | 用途 |
|---|---|---|
| 训练数据 | [`dataset/30min/fu/train/slice/`](../../dataset/30min/fu/train/slice/) | 14 个 `df_*.feather`，训练分箱 |
| 验证数据 | [`dataset/30min/fu/valid/`](../../dataset/30min/fu/valid/) | 12 个合约 feather，分布外推 |
| 二维选择表 | [`two_dimensional_selection.csv`](../../analysis_result/DiHFT/low_level/fu/30min_multi/two_dimensional_selection/two_dimensional_selection.csv) | 16 个 slot 最终状态 |
| 选择配置 | [`two_dimensional_selection_manifest.json`](../../analysis_result/DiHFT/low_level/fu/30min_multi/two_dimensional_selection/two_dimensional_selection_manifest.json) | 273 候选、门槛、聚合定义 |
| 训练主程序 | [`weight_advantage_pretrain.py`](../../FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py#L512) | 采样、loss、超参数更新 |
| 环境 reward | [`base_env.py`](../../FineFT/env/env_class/base_env.py#L898) | 单步 raw equity-change reward |
| replay buffer | [`replay_buffer_DQN.py`](../../FineFT/RL/util/replay_buffer_DQN.py#L128) | 均匀随机采样 |
| teacher Q-table | [`futures_util.py`](../../FineFT/env/env_class/futures_util.py#L1340) | 完整路径反向动态规划 |
| teacher 注入 | [`demo_env.py`](../../FineFT/env/env_class/demo_env.py#L79) | 将 Q-table 写入 `info["q_value"]` |
| Q 网络 | [`low_level.py`](../../FineFT/model/low_level.py#L12) | state/time/previous action/trading info 的 MLP |
| 30min 启动参数 | [`train_commodity_fu_30_half.sh`](../../FineFT/script/train/train_commodity_fu_30_half.sh#L16) | cost、n-step、gamma、pretrain epoch |
| 训练日志 | [`advantage.log`](../../log/DiHFT/fu/low_level/train/30min_multi/advantage.log) | `ada`、TD/KL 更新日志 |
| 特征算子 | [`time_operator_util.py`](../../data_preprocess/operator_futures/time_operator/time_operator_util.py#L366) | 趋势、ADX、vol regime 候选定义 |
| 风险特征算子 | [`multi_processing_util.py`](../../data_preprocess/operator_futures/time_operator/multi_processing_util.py#L678) | rolling/Parkinson/GK/ATR 定义 |
| 特征规范 | [`feature engineering spec`](../../openspec/specs/commodity-futures-feature-engineering/spec.md#L251) | 风险与流动性公式 |

### 3.2 研究时读取、当前逻辑路径已缺失的产物

下列产物在本轮前段分析时可读，并由此记录了摘要统计；报告生成时它们已不在当前仓库逻辑路径。本文**不恢复、不引用回收站路径，也不把这些统计伪装成当前可即时复核**：

- 逻辑路径 `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/30min/fu/train/feature_selection_manifest.json` 及同目录 `aggregate_metrics.csv`；
- 逻辑路径 `result/DiHFT/low_level/fu/30min_multi/weights_advantage_pretrain/qtable_diagnostics/*.csv`。

下次研究必须在命令记录中明确 `DATA_ROOT` 与 `RESULT_ROOT` 的物理路径映射，并复制 manifest 的校验和。例如：

```bash
export FEATURE_ROOT=/absolute/path/to/PREPROCESS_DATASET
export RESULT_ROOT=/absolute/path/to/result
sha256sum "$FEATURE_ROOT/commodity-futures/FEATURE_SELECTION/30min/fu/train/feature_selection_manifest.json"
find "$RESULT_ROOT/DiHFT/low_level/fu/30min_multi/weights_advantage_pretrain/qtable_diagnostics" -name '*.csv' | sort | sha256sum
```

### 3.3 环境

项目约定通过 `finetf` conda 环境执行 Python：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
python --version
python -c 'import numpy,pandas,torch; print(numpy.__version__, pandas.__version__, torch.__version__)'
git rev-parse HEAD
git status --short
```

为避免“同一脚本、不同输入”的不可追踪差异，下次还应保存：Git commit、dirty diff、所有输入文件列表与校验和、Python/包版本、随机种子、时区和运行命令。

## 4. 因果 16 格方法

### 4.1 窗口和公式

对每个合约独立排序，窗口 $W=48$（30 分钟频率约为 24 小时的观测长度；跨非交易时段仍按观测 bar 计数）。令 $p_t$ 为 `mark_price`，$y_i=\log p_i$，$x_i=0,1,\dots,W-1$。

有符号 OLS 斜率：

$$
\beta_t = \frac{\sum_{i=0}^{W-1}(x_i-\bar{x})(y_i-\bar{y})}
{\sum_{i=0}^{W-1}(x_i-\bar{x})^2}
$$

滚动波动率使用窗口内 47 个对数收益率的总体标准差：

$$
r_i=\log(p_i/p_{i-1}),\qquad
\sigma_t=\sqrt{\frac{1}{W-1}\sum(r_i-\bar r)^2}
$$

首 47 行没有完整窗口，每个合约分别丢弃。所有分位点只从 train 拟合；valid 必须使用固定的 train 阈值，以避免数据泄露。`np.searchsorted(thresholds, value, side="right")` 将每个指标分为 0–3 档。

### 4.2 连续 run

同一合约内，只要 `(vol_bin, slope_bin)` 与上一可用时点不同，就开始新 run。记录：

- `steps`：该格总步数；
- `contracts`：至少出现一次该格的合约数；
- `runs`：连续片段数；
- `mean/median run`：片段长度；
- `mean |return|`：当前单 bar 对数收益绝对值均值。

run 统计比单纯步数更接近“独立市场事件数”，也更适合后续 run-aware 抽样。

### 4.3 数据量和训练阈值

- train 原始行：15,997；合约切片：14；
- 每合约丢弃前 47 行后：15,339 个可用时点；
- train 阈值（百分数单位）：
  - slope `% log-return / bar`：`[-0.030226, 0.015009, 0.050995]`；
  - volatility `% log-return std`：`[0.311857, 0.391978, 0.472213]`。

## 5. 训练数据 16 格完整分布

单位：`slope bp/bar`、`vol bp`、`|ret| bp` 均为基点；share 为 15,339 可用步中的占比。

| vol | slope | steps | share % | contracts | runs | mean run | median run | slope bp/bar | vol bp | mean \|ret\| bp |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 423 | 2.76 | 11 | 39 | 10.8 | 7.0 | -5.185 | 26.58 | 16.13 |
| 0 | 1 | 1,235 | 8.05 | 13 | 96 | 12.9 | 8.0 | -0.260 | 26.39 | 19.33 |
| 0 | 2 | 1,443 | 9.41 | 13 | 123 | 11.7 | 7.0 | 3.162 | 26.85 | 19.03 |
| 0 | 3 | 734 | 4.79 | 13 | 57 | 12.9 | 12.0 | 7.789 | 27.50 | 18.68 |
| 1 | 0 | 763 | 4.97 | 13 | 99 | 7.7 | 6.0 | -7.271 | 35.63 | 23.14 |
| 1 | 1 | 927 | 6.04 | 13 | 136 | 6.8 | 5.0 | -0.688 | 35.47 | 23.63 |
| 1 | 2 | 1,090 | 7.11 | 13 | 166 | 6.6 | 4.0 | 3.510 | 35.05 | 25.79 |
| 1 | 3 | 1,054 | 6.87 | 14 | 123 | 8.6 | 6.0 | 8.327 | 35.15 | 23.41 |
| 2 | 0 | 1,164 | 7.59 | 13 | 114 | 10.2 | 7.5 | -8.730 | 43.02 | 29.10 |
| 2 | 1 | 881 | 5.74 | 14 | 138 | 6.4 | 4.0 | -0.782 | 43.08 | 30.43 |
| 2 | 2 | 783 | 5.10 | 14 | 145 | 5.4 | 3.0 | 3.407 | 42.86 | 29.81 |
| 2 | 3 | 1,007 | 6.57 | 14 | 134 | 7.5 | 5.0 | 8.100 | 43.04 | 27.76 |
| 3 | 0 | 1,485 | 9.68 | 13 | 80 | 18.6 | 10.5 | -10.464 | 64.84 | 36.77 |
| 3 | 1 | 791 | 5.16 | 12 | 110 | 7.2 | 4.0 | -0.744 | 55.92 | 37.49 |
| 3 | 2 | 519 | 3.38 | 13 | 109 | 4.8 | 4.0 | 3.237 | 55.01 | 41.74 |
| 3 | 3 | 1,040 | 6.78 | 12 | 75 | 13.9 | 8.0 | 10.936 | 60.38 | 31.39 |

核心观察：

- 低波动强方向四格合计 `423+734+763+1054=2,974` 步，占 **19.39%**，不能概括为整体罕见。
- 真正薄弱的是独立事件数。`v0/s0` 只有 39 个 run，远少于 `v1/s2` 的 166 个 run；用 transition 随机采样会反复看到同一长片段，并不会创造新的市场事件。
- 低波动强趋势 run 并非全都短：`v0/s0` 和 `v0/s3` 平均 run 分别为 10.8、12.9 bar，与当前 `n_step=12` 基本对齐。启动参数可见 [30min train script](../../FineFT/script/train/train_commodity_fu_30_half.sh#L19)。所以第一轮实验应先保持 n-step 不变，避免多因素混淆。
- 用“平均斜率 × 平均 run”只做趋势机会的数量级检查：`v0/s0≈-56 bp`、`v0/s3≈+100 bp`。当前单次交易成本参数为 6 bp，见 [train script](../../FineFT/script/train/train_commodity_fu_30_half.sh#L21)。这说明持续持仓可能覆盖成本，**但该乘积不是可实现净收益**，未包含回归残差、买卖价差、滑点、动作时机与往返次数。

## 6. Train → Valid 分布迁移

valid 原始目录有 12 个合约；按同样的 48-bar 因果计算、但固定使用 train 阈值后得到 9,964 个可用时点。

| vol | slope | train % | valid % | valid/train |
|---:|---:|---:|---:|---:|
| 0 | 0 | 2.76 | 1.43 | 0.52 |
| 0 | 1 | 8.05 | 6.63 | 0.82 |
| 0 | 2 | 9.41 | 9.17 | 0.98 |
| 0 | 3 | 4.79 | 6.37 | 1.33 |
| 1 | 0 | 4.97 | 5.61 | 1.13 |
| 1 | 1 | 6.04 | 8.27 | 1.37 |
| 1 | 2 | 7.11 | 7.42 | 1.04 |
| 1 | 3 | 6.87 | 4.69 | 0.68 |
| 2 | 0 | 7.59 | 6.67 | 0.88 |
| 2 | 1 | 5.74 | 4.60 | 0.80 |
| 2 | 2 | 5.10 | 3.78 | 0.74 |
| 2 | 3 | 6.57 | 4.25 | 0.65 |
| 3 | 0 | 9.68 | 12.93 | 1.34 |
| 3 | 1 | 5.16 | 4.15 | 0.81 |
| 3 | 2 | 3.38 | 3.86 | 1.14 |
| 3 | 3 | 6.78 | 10.17 | 1.50 |

采用

$$PSI=\sum_i(q_i-p_i)\log(q_i/p_i)$$

其中 $p_i$ 为 train 占比、$q_i$ 为 valid 占比，得到 **PSI=0.0761**。这属于本数据内的相对温和整体变化；不应把它解释成普适行业阈值。低波动强方向四格从 train 的 19.39% 变为 valid 的 18.10%，总量接近，但 `v0/s0` 减少到一半、`v0/s3` 增加 33%，说明**总体覆盖不差不代表逐格稳定**。

## 7. 二维 selector 的失败证据

[selection manifest](../../analysis_result/DiHFT/low_level/fu/30min_multi/two_dimensional_selection/two_dimensional_selection_manifest.json) 记录：

- 共同 epoch 为 60–98；
- 发现并纳入 273 个完整候选，无不完整候选；
- 5 个初始动作；
- `lcb_z=0`、`min_positive_contract_ratio=0`、`min_mean_return=0`、`min_lcb=0`、`min_worst_initial_position_return=0`；
- `pair_score` 是波动率边际、斜率边际和两个 joint-run 子集 LCB 的最小值；聚合先在合约内对初始仓位平均，再做合约等权。

尽管门槛已经宽松，[selection CSV](../../analysis_result/DiHFT/low_level/fu/30min_multi/two_dimensional_selection/two_dimensional_selection.csv) 中 `vol=label_0/label_1` 覆盖的**全部 8 个斜率格均为 `empty_model`**，说明失败不只发生在强方向格。用户重点关注的四个强负/正斜率格明细如下：

| selector 格 | 最佳被拒候选 | pair score | 拒绝原因 |
|---|---|---:|---|
| `vol=label_0, slope=label_0` | `epoch_94:bin_6` | -0.951747 | mean_return, lcb, worst_initial_position |
| `vol=label_0, slope=label_3` | `epoch_85:bin_4` | -0.762586 | mean_return, lcb, worst_initial_position |
| `vol=label_1, slope=label_0` | `epoch_97:bin_3` | -0.406818 | mean_return, lcb, worst_initial_position |
| `vol=label_1, slope=label_3` | `epoch_98:bin_2` | -0.491303 | mean_return, lcb, worst_initial_position |

`empty_model → flat_position` 是选择层的安全回退，只能避免部署一个已经验证亏损的模型；它不会让训练器学会这四种状态。

可复核断言：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
python - <<'PY'
import pandas as pd

p = "analysis_result/DiHFT/low_level/fu/30min_multi/two_dimensional_selection/two_dimensional_selection.csv"
d = pd.read_csv(p)
q = d[d.volatility_label.isin(["label_0", "label_1"]) &
      d.slope_label.isin(["label_0", "label_3"])]
print(q[["volatility_label", "slope_label", "kind",
         "best_rejected_pair_score", "best_rejected_reasons"]])
assert len(q) == 4
assert not q.kind.eq("model").any(), (
    "预期 RED：低波动强趋势四格不应已有通过门槛的候选"
)
PY
```

## 8. 训练机制审计

### 8.1 Reward 幅度与高波动

环境主 reward 是 `wallet_balance + unrealized_pnl - previous_margin_balance`，再叠加可选涨跌停 shaping，见 [`base_env.py`](../../FineFT/env/env_class/base_env.py#L900)。因此相同仓位下，高波动时 reward 和 TD target 的绝对值自然更大；16 格中最高波动档的 `mean |return|` 为 31–42 bp，低波动档为 16–19 bp，约为 2 倍量级。

这会使未经条件平衡的优化更重视高幅度 transition，但不是“高波动一定训练差”的充分证据：预训练 TD 使用 SmoothL1/Huber 类损失，极端误差的梯度会被限制；同时高波动格的持仓风险、错误换仓和合约差异也会变大。因此 reward scale 是**次要、待消融因素**。不建议直接使用 `reward/volatility`，因为这会在极低波动处分母放大噪声，并改变原策略的经济目标。

### 8.2 采样没有感知 16 格与 run

训练日程调用 `build_balanced_training_schedule(sample_plan, num_sample)`，随后按 `df_index` 与 `initial_action` 进入环境，见 [`weight_advantage_pretrain.py`](../../FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py#L1218)。Replay buffer 的 `sample()` 使用 `random.sample`，见 [`replay_buffer_DQN.py`](../../FineFT/RL/util/replay_buffer_DQN.py#L128)。

代码中 `self.priority_transformation = get_transformation_even_risk` 被赋值，见 [`weight_advantage_pretrain.py`](../../FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py#L512)，但本轮代码搜索未发现其在该训练脚本内被调用。故现有训练不能视为已经进行了 market-regime 风险平衡。

### 8.3 网络需要从混合特征间接推断 regime

Q 网络把 `state` 经一个线性层/ReLU，与 previous action、time 和 trading info 拼接，再输出动作值，见 [`low_level.py`](../../FineFT/model/low_level.py#L28)。它不是序列模型，无法在网络内部直接回看 48 根原始价格；能否识别“缓慢但持续”主要取决于输入特征是否显式、稳定地编码这种结构。

### 8.4 Teacher 是 future-aware，且可能过度换仓

`Demo_Env` 用完整路径构造 `create_optimal_q_table(...)`，并在 reset/step 后把对应 Q 值写入 info，见 [`demo_env.py`](../../FineFT/env/env_class/demo_env.py#L79)。预训练把网络 Q 和 teacher Q 各自 softmax 后做 KL，最终 `loss = td_loss + ada * KL`，见 [`weight_advantage_pretrain.py`](../../FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py#L814)。这说明 teacher 具有后见性，不等于在线可观测策略。

研究时读取 q-table diagnostics 的摘要如下；原 CSV 当前不在逻辑结果路径，必须按第 3.2 节重新落盘后再复核：

- 各训练切片 teacher episode reward 均为正；
- 每个切片 teacher 换仓约 130–667 次，约占步数的 25%–30%；
- teacher flat ratio 接近 0；
- 重算代表切片的 teacher action softmax：

| train slice | entropy median | best-second Q gap median | gap p10 |
|---|---:|---:|---:|
| `df_0` | 0.0230 | 5.6713 | 未记录 |
| `df_2` | 0.0403 | 5.0023 | 未记录 |
| `df_13`（较低波动） | 0.0952 | 3.9668 | 0.1956 |

低波动切片 teacher target 更模糊，但差异不是数量级变化。这支持“用 teacher 置信度调 KL”的后续实验，不支持仅据此断言 teacher 是唯一根因。

### 8.5 `ada=256` 看起来大，但 KL 没有统治已记录更新

`ada` 初始化后保持 `ada_init`，见 [`weight_advantage_pretrain.py`](../../FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py#L589)；衰减代码位于多样化训练采数分支，见 [同文件](../../FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py#L1353)，预训练阶段没有在该路径衰减。当前启动脚本的 `ada_min=0.01` 只改变衰减下限，见 [`train_commodity_fu_30_half.sh`](../../FineFT/script/train/train_commodity_fu_30_half.sh#L23)，不会自动改变预训练开始时的 256。

但对当前可读 [30min advantage log](../../log/DiHFT/fu/low_level/train/30min_multi/advantage.log) 中 355 条预训练更新记录解析后：

- 加权 KL / TD 的中位数：0.0863；
- p10：0.0382；p90：0.3606；
- `weighted KL > TD` 的比例：0.0028。

因此本轮证伪了“KL 项在数值上普遍压倒 TD”这一初始假设。`ada` 仍可能通过动作分布和 teacher 偏差产生结构性影响，但应排在表示、成本感知和采样之后。

## 9. 当前 150 特征审计

### 9.1 产物一致性与总体指标（研究时快照）

研究时对用户指定逻辑路径
`PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/30min/fu/train/feature_selection_manifest.json`
与两个 `state_features.npy` 做了精确比较，记录为：

- 候选特征 880，选中 150，选择率 17.05%；
- 42 个 mandatory 特征全部保留；
- manifest 的 150 个特征与 train `state_features.npy`、`dataset/30min/fu/state_features.npy` 的内容和顺序完全一致；当前仓库仍可读取后者，但 manifest 当前逻辑路径已缺失；
- 全候选 mean $|IC|=0.0412$，选中特征 mean $|IC|=0.0458$；
- 全候选 mean permutation importance=0.0608，选中特征为 0.0647；
- 14 个 train 合约的 cross-contract stability 检查通过；
- 32 个选中特征的 valid 绝对 IC 相比 train 下降超过 50%，产生 degradation warning。

上述数字应视作**有来源边界的研究快照**，不是当前文件系统可独立重跑的最终审计。下次必须恢复正式 `FEATURE_ROOT` 映射、保存 manifest 校验和，再运行仓库的：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
python .agents/skills/commodity-futures-feature-analysis/scripts/analyze_feature_selection.py \
  --help
python .agents/skills/commodity-futures-feature-analysis/scripts/diagnose_feature_issues.py \
  --help
```

先看 `--help` 再按当时脚本签名传入 manifest 和 train/valid 指标目录，避免报告固化过期 CLI 参数。

### 9.2 已有信息与缺口

按名称模式做重叠分类，150 个特征中约有：direction/trend 25、vol/range 23、std-normalized 13、spread/execution proxy 11。它们不是互斥类别，数量不能相加作为总数。

有用的现有特征包括：

- 短期方向：`ask1_price_log_return_2`、`ask1_price_trend_2`、`wap_2_trend_2`、`ask2_price_trend_6`；
- 稍长方向：`buy_wap_trend_24`、`price_velocity_10m`、`adx_14`、`beta_2_origin`、`beta_6_origin`；
- 波动：`rolling_volatility_2/24`、`parkinson_volatility_6/16`、`garman_klass_volatility_2`、`atr_pct_192`；
- 标准化变化：若干 `roc_*_std_norm_origin`。

但这些并不等价于目标 regime：

- `price_velocity_10m=(EMA10(close)-EMA20(close))/close`，见 [`time_operator_util.py`](../../data_preprocess/operator_futures/time_operator/time_operator_util.py#L367)，属于短期 EMA 差，不是 48/96-bar OLS 斜率；
- `adx_14` 是 $|DI^+-DI^-|/(DI^++DI^-)$ 的平滑，见 [同文件](../../data_preprocess/operator_futures/time_operator/time_operator_util.py#L388)，强度无方向；
- `roc_{window}_std_norm = close_shift / close_std`，见 [`multi_processing_util.py`](../../data_preprocess/operator_futures/time_operator/multi_processing_util.py#L486)，不是“斜率 ÷ 收益波动”；
- 风险算子虽实现 rolling/Parkinson/Garman-Klass 波动率，见 [`multi_processing_util.py`](../../data_preprocess/operator_futures/time_operator/multi_processing_util.py#L689) 和 [spec](../../openspec/specs/commodity-futures-feature-engineering/spec.md#L255)，但研究快照中未选中明确的波动率 regime quantile/z-score；
- 算子已经生成 `ema_slope_96/192`、`garman_klass_vol_quantile_192`、`parkinson_vol_zscore_192`，见 [`time_operator_util.py`](../../data_preprocess/operator_futures/time_operator/time_operator_util.py#L376)，但快照中这些特征未选中；
- 没有明确的 signed long-window normalized slope、趋势效率比、趋势回归 $R^2$、预期移动/交易成本比。`pivot_r2` 名称不能当作趋势回归拟合优度。

### 9.3 代表性 Train → Valid IC 快照

| 特征 | train IC | valid IC | 观察 |
|---|---:|---:|---|
| `ask1_price_log_return_2` | 0.20830 | 0.13719 | 绝对值降 34.1% |
| `ask1_price_trend_2` | 0.18991 | 0.13686 | 降 27.9% |
| `wap_2_trend_2` | 0.17139 | 0.13332 | 降 22.2% |
| `ask2_price_trend_6` | 0.13437 | 0.11160 | 降 16.9% |
| `buy_wap_trend_24` | 0.06069 | 0.06870 | 改善 |
| `price_velocity_10m` | -0.06988 | -0.03728 | 绝对值降 46.7% |
| `adx_14` | -0.01744 | 0.04976 | 反号、绝对值增加 |
| `beta_6_origin` | 0.03059 | -0.00680 | 反号，绝对值降 77.8% |
| `roc_6_std_norm` | 0.03354 | 0.00321 | 降 90.4% |
| `roc_12_std_norm` | 0.05331 | -0.00529 | 反号，绝对值降 90.1% |
| `roc_24_std_norm` | 0.06695 | -0.05059 | 反号，绝对值降 24.4% |
| `roc_96_std_norm` | 0.05495 | -0.04105 | 反号，绝对值降 25.3% |
| `rolling_volatility_2` | -0.03732 | 0.00949 | 反号，绝对值降 74.6% |
| `rolling_volatility_24` | -0.05134 | 0.02717 | 反号，绝对值降 47.1% |
| `parkinson_volatility_6` | -0.04146 | -0.01639 | 降 60.5% |
| `parkinson_volatility_16` | -0.06292 | -0.01592 | 降 74.7% |
| `garman_klass_volatility_2` | -0.02346 | -0.01234 | 降 47.4% |
| `atr_pct_192` | 0.02370 | 0.07090 | 改善 |
| `std_2` | -0.03086 | 0.00145 | 降 95.3% |
| `std_6` | -0.03561 | 0.00122 | 降 96.6% |

单变量 IC 反号或衰减不能证明该特征对非线性 RL 无用；valid permutation importance 也可能稳定或增加。正确动作是做组级消融和条件评估，不是依据此表批量删除。

## 10. 假设的证伪与重排

| 假设 | 证据 | 判定 | 优先级 |
|---|---|---|---:|
| 低波动强趋势样本整体太少 | 四目标格占 train 19.39% | 否定“整体稀缺”；独立 run 稀缺仍成立 | 3 |
| 严重 train-valid 状态分布断层 | PSI=0.0761，目标四格总量 19.39%→18.10% | 全局不成立；单格漂移成立 | 3 |
| `ada=256` 使 KL 数值统治 TD | weighted KL/TD 中位 0.0863，超过 TD 仅 0.28% | 数值统治假设被否定 | 5 |
| 高波动 raw reward 完全压制低波动 | 幅度约 2 倍，但 SmoothL1 限制大误差且无直接因果消融 | 有机制可能，证据不足以列首因 | 4 |
| 状态表示难以稳定识别 regime | 缺少显式 signed slope/noise、efficiency、$R^2$、vol-regime；多个代理特征 valid 衰减 | 当前最强解释 | 1 |
| 频繁换仓侵蚀低波动趋势 | cost=6 bp；teacher 高频切换；目标依赖持久持仓 | 强机制证据，需 turnover 消融确认 | 2 |
| selector 已解决问题 | 四目标格都 empty | 否；只提供安全回退 | — |

优先级表示下一轮验证顺序，不表示已经完成因果证明。

## 11. 基于数据的改良方案

### 11.1 先加入 5–8 个因果 Regime Anchor

从单一窗口 48 开始，避免一次加入大量共线变体：

1. `log_price_slope_48`：上述 OLS 斜率；
2. `return_volatility_48`：对数收益滚动标准差；
3. `trend_to_noise_48 = log_price_slope_48 / max(return_volatility_48, floor)`；
4. `signed_efficiency_48 = (logp_t-logp_{t-47}) / sum(abs(log_return), 47)`；
5. `trend_r2_48`：log-price 对时间回归的 $R^2$；
6. `volatility_quantile_192`：当前波动率在过去 192 bar 的因果位置；
7. 可选 `expected_move_to_cost_48`，但必须用当时可观测成本和 spread；
8. 只有 48-window 通过后，再加入 96-window 变体。

所有分母设训练分布确定的下限并记录，特征只使用过去数据。首次消融可保护 5 个核心 anchor 不被全局 IC 筛选立即丢弃，但保护不是永久白名单；最终仍要求条件稳定性。

### 11.2 特征筛选增加条件指标

除全局 IC/permutation importance 外，对每个候选记录：

- 16 个因果格内的 IC/收益增量；
- 合约等权 mean、标准误和 LCB；
- 每格有样本合约数与有效 run 数；
- conditional sign stability；
- worst target cell；
- train→valid 条件衰减。

稀有格不能仅按逐步样本给很高置信度。建议以合约或 run 为统计单位，必要时用 bootstrap by contract/run。

### 11.3 成本感知动作滞回

用当前持仓动作 $a_{cur}$ 和候选 $a$：

$$
\text{switch only if }Q(s,a)-Q(s,a_{cur})
> C_{switch}(a_{cur},a)+U(s)
$$

$C_{switch}$ 至少包含手续费、可估计 spread/slippage 与反手的双边成本；$U(s)$ 是不确定性/安全边际。错误初始仓位必须仍允许及时退出，所以不能简单规定最小持仓长度。首轮只比较 turnover、持仓持续时间、目标四格净收益和 worst-initial-position。

### 11.4 温和的 run-aware 重采样

不要直接让 16 格各占 6.25%。建议：

$$
w_i=\operatorname{clip}\left(\sqrt{6.25/\text{share\_pct}_i},0.8,1.5\right)
$$

采样层次为：`cell → contract → contiguous run → transition/window`。这样先保护独立市场事件和合约覆盖，再在 run 内采 transition。严格均匀会把 `v0/s0` 从 2.76% 放大约 2.3 倍，同时把 `v3/s0` 降至约 65%，可能牺牲高波动表现；上式把调整限制在 0.8–1.5。

### 11.5 Reward 与 teacher 的后续处理

- reward scaling：先报告每格 TD loss/gradient contribution，再试 clipped 或由训练期稳健尺度归一化；不要直接除以实时极低 volatility。
- teacher KL：按 teacher best-second gap 或 entropy 降权模糊样本，例如 $w_{KL}=\operatorname{clip}(gap/g_0,0,1)$；不要只改全局 `ada_min`。
- teacher 动作：考虑把切换成本/滞回加入 teacher DP，或只在 teacher 高置信度时蒸馏。
- 保持 `n_step=12` 作为第一阶段固定项，待 anchor+滞回有效后再比较 8/12/16。

## 12. 最小可归因消融阶梯

所有实验使用同一 train/valid、同一随机种子集合、同一候选 epoch 规则与二维评估：

| 实验 | 相对上一组只改变 | 目的 |
|---|---|---|
| A | 当前 150 特征基线 | 重新确认四目标格失败和方差 |
| B | A + 5 个核心 Regime Anchor | 验证表示假设 |
| C | B + 对高衰减特征做组级消融 | 验证稳定性，不逐特征追噪声 |
| D | 最佳 C + cost-aware hysteresis | 验证换仓成本假设 |
| E | D + capped run-aware sampling | 验证独立 run/覆盖假设 |
| F | E + confidence-weighted teacher KL | 最后验证 teacher 模糊度 |

不要同时改特征、采样、reward、n-step、KL 和 selector 门槛，否则成功后无法知道原因。

## 13. 验收指标

### 13.1 必须过的目标格指标

对 `(v0,s0) (v0,s3) (v1,s0) (v1,s3)` 分别报告：

- 合约等权 net return per step/run；
- contract-level LCB > 0；
- worst initial position return > 0；
- positive-contract ratio；
- 有效合约数、run 数、步数；
- turnover、动作切换率、平均持仓持续时间；
- 相对 flat 和当前 baseline 的增量。

### 13.2 防止修好低波动却破坏高波动

- `v3/*` 每格单独报告；
- natural-frequency weighted 总收益和合约等权总收益都报告；
- 高波动格或总体收益相对 A 的下降不得超过预先写入实验配置的容忍范围，建议首轮为 3%–5%，但最终阈值由部署风险预算决定；
- 最大回撤、尾部损失、turnover、交易成本和 worst contract 不得恶化到超出预注册阈值。

选择成功的定义不是“平均收益提高”，而是目标四格至少出现稳定优于 flat 的候选，同时不以不可接受的高波动/整体退化换取。

## 14. 复现代码：16 格、run 与 PSI

以下脚本只依赖当前仓库 feather，可直接粘贴运行。它打印阈值、train/valid 分布、run 统计和 PSI；为避免写入工作树，不保存文件。

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
python - <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 48
TRAIN = Path("dataset/30min/fu/train/slice")
VALID = Path("dataset/30min/fu/valid")


def ols_slope(values: np.ndarray) -> float:
    x = np.arange(values.size, dtype=np.float64)
    xc = x - x.mean()
    return float(np.dot(xc, values - values.mean()) / np.dot(xc, xc))


def causal_metrics(path: Path, contract: str) -> pd.DataFrame:
    d = pd.read_feather(path, columns=["timestamp", "mark_price"])
    d = d.sort_values("timestamp", kind="stable").reset_index(drop=True)
    logp = np.log(d.mark_price.astype(float))
    slope = logp.rolling(WINDOW).apply(
        lambda x: ols_slope(x.to_numpy()), raw=False
    )
    # 48 prices contain 47 one-bar returns; ddof=0 matches this report.
    ret = logp.diff()
    vol = ret.rolling(WINDOW - 1).std(ddof=0)
    out = pd.DataFrame({
        "contract": contract,
        "timestamp": d.timestamp,
        "slope": slope,
        "vol": vol,
        "abs_ret": ret.abs(),
    })
    return out.dropna(subset=["slope", "vol"]).reset_index(drop=True)


def load_dir(root: Path, train_slice: bool) -> pd.DataFrame:
    files = sorted(root.glob("df_*.feather" if train_slice else "*.feather"))
    return pd.concat(
        [causal_metrics(p, p.stem) for p in files], ignore_index=True
    )


def assign_bins(d: pd.DataFrame, slope_q: np.ndarray,
                vol_q: np.ndarray) -> pd.DataFrame:
    d = d.copy()
    d["slope_bin"] = np.searchsorted(slope_q, d.slope, side="right")
    d["vol_bin"] = np.searchsorted(vol_q, d.vol, side="right")
    changed = (
        d.contract.ne(d.contract.shift())
        | d.slope_bin.ne(d.slope_bin.shift())
        | d.vol_bin.ne(d.vol_bin.shift())
    )
    d["run_id"] = changed.cumsum()
    return d


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    run_len = (
        d.groupby(["vol_bin", "slope_bin", "run_id"], observed=True)
        .size().rename("run_length").reset_index()
    )
    cell = (
        d.groupby(["vol_bin", "slope_bin"], observed=True)
        .agg(steps=("slope", "size"), contracts=("contract", "nunique"),
             mean_slope=("slope", "mean"), mean_vol=("vol", "mean"),
             mean_abs_ret=("abs_ret", "mean"))
    )
    runs = run_len.groupby(["vol_bin", "slope_bin"]).run_length.agg(
        runs="size", mean_run="mean", median_run="median"
    )
    out = cell.join(runs).reset_index()
    out["share_pct"] = out.steps / len(d) * 100
    for col in ["mean_slope", "mean_vol", "mean_abs_ret"]:
        out[col] *= 10_000
    return out.sort_values(["vol_bin", "slope_bin"])


train = load_dir(TRAIN, train_slice=True)
slope_q = train.slope.quantile([.25, .5, .75]).to_numpy()
vol_q = train.vol.quantile([.25, .5, .75]).to_numpy()
train = assign_bins(train, slope_q, vol_q)
valid = assign_bins(load_dir(VALID, train_slice=False), slope_q, vol_q)

print("usable train/valid:", len(train), len(valid))
print("slope thresholds (%/bar):", slope_q * 100)
print("vol thresholds (%):", vol_q * 100)
print("\nTRAIN")
print(summarize(train).to_string(index=False))

def shares(d: pd.DataFrame) -> np.ndarray:
    idx = pd.MultiIndex.from_product([range(4), range(4)])
    return (d.groupby(["vol_bin", "slope_bin"]).size()
            .reindex(idx, fill_value=0).to_numpy(dtype=float) / len(d))

p, q = shares(train), shares(valid)
eps = 1e-12
psi = np.sum((q - p) * np.log((q + eps) / (p + eps)))
comparison = pd.DataFrame({
    "vol": np.repeat(range(4), 4), "slope": np.tile(range(4), 4),
    "train_pct": p * 100, "valid_pct": q * 100,
    "ratio": np.divide(q, p, out=np.full_like(q, np.nan), where=p > 0),
})
print("\nTRAIN -> VALID")
print(comparison.to_string(index=False))
print("PSI:", psi)
PY
```

## 15. 复用本方法的标准流程

1. **冻结证据**：记录 commit、dirty diff、`DATA_ROOT/RESULT_ROOT/FEATURE_ROOT`、文件清单和校验和。
2. **定义标签边界**：明确 causal rolling grid 还是 offline segmentation grid；不混用。
3. **train-only 拟合阈值**：保存窗口、价格列、ddof、分位点和缺失值处理。
4. **统计覆盖**：同时报告 steps、contracts、runs、run length，不只看行数。
5. **检查迁移**：valid 使用 train 阈值；报告逐格 ratio 和 PSI。
6. **验证模型失败**：每格列出 mean、LCB、worst initial position、合约覆盖和 flat baseline。
7. **审计训练机制**：采样单位、reward 尺度、TD/KL 相对量级、teacher entropy/gap、动作切换率。
8. **审计特征**：核对 manifest 与实际 state feature 顺序；做 global + conditional + cross-contract + train-valid 稳定性。
9. **预注册消融**：一次只改一个因素，固定 seed、epoch 与选择门槛。
10. **按风险验收**：目标格必须优于 flat，同时约束高波动和总体退化。

## 16. 局限性与未完成的因果证明

1. 因果 48-bar grid 与 selector 离线切片不是同一标签生成过程，不能进行严格逐时点映射。
2. 当前 train 的 `df_*.feather` 是切片文件；若一个真实合约被拆为多个 `df`，本报告的 `contracts` 实际更接近“训练切片数/标识数”。下次应从 manifest 保存真实 symbol 映射。
3. OLS 窗口按观测 bar，不按连续墙钟时间；节假日和夜盘间隔不会额外加权。
4. `slope × mean run` 只是趋势幅度数量级，不是策略可实现收益。
5. IC 是单变量相关性，不能替代策略条件 PnL 或非线性增量价值。
6. q-table diagnostics 和 feature-selection manifest 的正式逻辑路径在报告生成时缺失；对应数字是研究时快照。重跑前必须恢复正式数据映射并校验输入，不能以回收站文件充当正式证据。
7. 现有分析是观察性诊断。状态表示、滞回、run-aware sampling、reward scale 和 teacher KL 的相对因果贡献，最终只能由第 12 节的受控消融确定。

## 17. 最终决策

下一轮不应从“全面加大低波动样本权重”开始。先实现并验证一个小型、因果、可解释的 Regime Anchor 特征组，同时保持 `n_step=12`、reward、采样和 KL 不变；若 B 相对 A 改善四目标格，再依次做不稳定特征组消融、成本感知滞回和 capped run-aware sampling。只有在这些步骤完成后，才测试 confidence-weighted teacher KL。

该顺序同时解释了两个看似矛盾的现象：低波动强趋势在数据中并不少，却没有候选通过；高波动 reward 虽更大，却不能单独解释所有失败。真正需要优化的是**模型能否稳定辨识趋势相对于噪声的结构，以及是否能在可覆盖交易成本的时间尺度上保持正确仓位**。
