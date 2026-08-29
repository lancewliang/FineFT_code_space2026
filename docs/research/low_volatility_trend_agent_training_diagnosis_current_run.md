# 30 分钟商品期货低波动趋势 Agent 本轮训练退化诊断

> 研究日期：2026-08-28  
> 对象：`fu / 30min_multi / DiHFT low-level agent`  
> 结果目录：[`two_dimensional_selection`](../../analysis_result/DiHFT/low_level/fu/30min_multi/two_dimensional_selection/)  
> 方法基线：[`low_volatility_trend_agent_training_diagnosis.md`](./low_volatility_trend_agent_training_diagnosis.md)

## 1. 结论先行

这次结果并非“所有低波动格都比上次差”，也不能被当成一次严格的增强版对基线 A/B。四个重点格中，`v0/s0`、`v0/s3` 的最优分数比旧报告改善，`v1/s0`、`v1/s3` 明显退化；本轮还首次有一个低波动格 `v0/s1` 通过选择。真正恶化的是：`v1` 全部未过门槛，`v3` 出现极端负值，总体只选出 4/16 个模型。

按证据强弱，退化由以下因素叠加造成：

1. **实验没有隔离，且选择口径发生变化。** 两次训练使用同一个 `experiment_name=30min_multi`，后一次覆盖了前一次 checkpoint；同时 epoch 范围和 selector 门槛也变了。因此当前目录不能提供严格的旧模型与新模型逐候选配对证据。
2. **本轮只加入了 Regime Anchor，没有启用成本感知动作滞回。** 日志明确为 `enable_action_persistence=False`，测试动作 100% 来自 `q_argmax`。低波动候选换手率约 18%–21%；尤其 `v1` 最好候选毛收益为正，但手续费和滑点后转负。
3. **Anchor 保留规则过于宽松，并重新引入了高度共线的特征。** 7 个 anchor 全部进入最终 152 维状态，其中 4 个是在相关性过滤后强制加回；多组两两相关系数达到 0.89–0.96。当前规则只需“任意目标格 × 任意 horizon”在 train 通过，未要求 valid 稳定，也未对齐 agent 的 `n_step=12`。
4. **新增表示显著增加了 teacher 模仿冲突。** 加权 KL/TD 的中位数由旧训练的 `0.0863` 升至本轮 `0.4775`，约为 5.5 倍。KL 很少直接超过 TD，但已经不再是可忽略项，说明新状态表示下 teacher 与网络拟合之间的冲突明显增强。
5. **`v3` 的巨大负分既有真实方向错误，也被短合约等权放大。** selector 先在单合约内求均值，再对合约等权；9–16 步的短切片与 359–681 步的长合约权重相同。短切片的极端亏损会把 `v3` 分数放大到 `-4` 至 `-6`，但其毛收益本身也为负，因此不能只归咎于聚合方式。

综合判断：**这轮更像一个未完成且未隔离的 B 实验，而不是“Regime Anchor 方案已被证伪”。** Anchor 的选择规则和冗余需要收紧；旧报告优先级第二的动作滞回实际没有被测试；选择器口径和短样本支持又进一步放大了表面退化。

## 2. 新旧结果对比

旧报告保存的四个重点格最优被拒分数，与当前 [`two_dimensional_selection.csv`](../../analysis_result/DiHFT/low_level/fu/30min_multi/two_dimensional_selection/two_dimensional_selection.csv) 对比如下：

| 目标格 | 旧报告 | 本轮 | 变化 | 判定 |
|---|---:|---:|---:|---|
| `v0/s0` | -0.951747 | -0.581176 | +0.370571 | 改善但仍未过门槛 |
| `v0/s3` | -0.762586 | -0.728155 | +0.034431 | 小幅改善但仍未过门槛 |
| `v1/s0` | -0.406818 | -0.803426 | -0.396608 | 明显退化 |
| `v1/s3` | -0.491303 | -0.857654 | -0.366351 | 明显退化 |

本轮最终选择 4/16 个 slot：

| slot | epoch/bin | pair score |
|---|---|---:|
| `v0/s1` | 54/0 | 0.161095 |
| `v2/s1` | 83/4 | 0.380678 |
| `v2/s2` | 72/2 | 0.052413 |
| `v2/s3` | 63/6 | 0.279492 |

各 volatility 档所有候选的 pair score 显示，`v1` 和 `v3` 才是主要退化区：

| volatility | 均值 | 中位数 | 最优值 | eligible 比例 |
|---|---:|---:|---:|---:|
| `v0` | -1.5305 | -1.7009 | 0.1611 | 0.3% |
| `v1` | -2.4873 | -2.3416 | -0.8034 | 0.0% |
| `v2` | -1.3958 | -1.4598 | 0.3807 | 3.9% |
| `v3` | -7.7494 | -6.7785 | -4.2395 | 0.0% |

因此，“四个重点格全都更差”与当前数据不符；准确描述是：**低波动第一档略有改善并选出一个模型，第二档显著退化，高波动第四档出现新的极端失败。**

## 3. 为什么不能直接归因于新增特征

### 3.1 基线被覆盖

[`advantage.log`](../../log/DiHFT/fu/low_level/train/30min_multi/advantage.log) 中有两次启动：

- 2026-08-27 08:15：150 个特征；
- 2026-08-27 22:16：152 个特征。

两次都写入 `30min_multi`。当前 `result/.../30min_multi` 的 checkpoint 时间属于第二次训练，旧 checkpoint 已不在同一路径。这样无法对相同 epoch、相同 selector、相同数据做模型级配对。

### 3.2 selector 同时改变

旧报告与当前 [`two_dimensional_selection_manifest.json`](../../analysis_result/DiHFT/low_level/fu/30min_multi/two_dimensional_selection/two_dimensional_selection_manifest.json) 的选择条件并不一致：

| 配置 | 旧报告 | 本轮 |
|---|---:|---:|
| epoch 范围 | 60–98 | 50–100 |
| 完整候选 | 273 | 357 |
| `min_positive_contract_ratio` | 0.0 | 0.4 |
| `min_worst_initial_position_return` | 0.0 | -1.0 |
| missing joint policy | 未记录该回退 | `slope_marginal_best` |

本轮对“正收益合约覆盖面”更严格，对“最差初始仓位”更宽松。空 slot 数量同时反映模型表现和门槛变化，不能作为纯训练退化指标。

## 4. 特征增强为何没有按预期生效

### 4.1 7 个 anchor 中有 4 个绕过相关性过滤

当前候选 887 个、最终选择 152 个；7 个 anchor 全部进入状态，其中仅 3 个自然通过相关性过滤，另外 4 个由 conditional retention 强制加回。关键相关性为：

| 特征对 | train 相关系数 |
|---|---:|
| `log_price_slope_48` / `trend_to_noise_48` | 0.9435 |
| `trend_to_noise_48` / `signed_efficiency_48` | 0.9150 |
| `log_price_slope_48` / `signed_efficiency_48` | 0.8898 |
| `log_price_slope_96` / `trend_to_noise_96` | 0.9572 |

这不是七个独立信息源，而是多个近似重复的趋势方向表达。把被相关性过滤删除的特征全部加回，会增加网络辨识和 teacher 拟合难度，却不保证增加有效信息。

### 4.2 conditional retention 与实际决策 horizon 不对齐

[`regime_audit.py`](../../data_preprocess/operator_futures/feature_selection/muti_contract/regime_audit.py) 当前逻辑在“任意目标格 × 任意 window”上通过 train LCB 即可保留，并不要求 `n_step=12` 下 train→valid 符号稳定。

在 window=12 的四个关注格上，train/valid IC 符号一致数为：

| anchor | 符号一致格数 |
|---|---:|
| `log_price_slope_48` | 0/4 |
| `trend_r2_48` | 0/4 |
| `trend_to_noise_48` | 1/4 |
| `signed_efficiency_48` | 1/4 |
| `log_price_slope_96` | 3/4 |
| `trend_to_noise_96` | 3/4 |
| `log_return_vol_quantile_192` | 3/4 |

例如，`v0/s0` 中 `log_price_slope_48` 的 RankIC 从 train `-0.3581` 变为 valid `+0.2980`，`trend_r2_48` 从 `+0.3924` 变为 `-0.2189`。这里的 causal audit 格与 selector 的离线格并非同一批时点，所以它不能直接证明某个 slot 的因果来源；但它足以说明当前“train 任一窗口通过就强制保留”的规则会接受决策尺度上不稳定的 anchor。

### 4.3 不是整体数据漂移

当前 train/valid 可用 mark-price 样本分别约 15,309/9,993，旧报告为 15,339/9,964；当前 16 格 PSI 为 `0.0749`，旧报告为 `0.0761`。总体分布漂移几乎没有变化，不能解释本轮突然恶化。

## 5. KL 冲突显著上升

训练 loss 为 `TD + ada × KL`。从两段日志解析得到：

| 指标 | 旧训练 | 本轮 |
|---|---:|---:|
| 加权 KL/TD 中位数 | 0.0863 | 0.4775 |
| P10 | 0.0384 | 0.2168 |
| P90 | 0.3571 | 0.7262 |
| 前 20 次更新中位数 | 0.5474 | 0.8851 |
| 后 20 次更新中位数 | 0.1042 | 0.2285 |

两轮中该比值大于 1 的比例都只有约 0.28%，所以不能说 KL “统治”了 loss；但本轮典型相对贡献上升约 5.5 倍，旧报告中“KL 暂非首因”的结论不能原样外推。更合理的解释是：冗余且条件不稳定的 anchor 改变了表示，使 teacher 目标与 TD 目标更难同时拟合。这个解释仍需冻结其他变量后的消融验证。

## 6. 未启用动作滞回，收益被交易成本吞噬

本轮日志明确记录 `enable_action_persistence=False`；测试明细的 `decision_reason` 100% 为 `q_argmax`。因此本轮验证的是“新增 anchor”，没有验证旧报告建议的“成本感知动作滞回”。

按 selector 的合约等权口径，几个 volatility 档最好候选的成本拆解为：

| volatility | 最好 epoch/bin | 毛收益（费前） | 净收益 | 毛收益为正合约 | 净收益为正合约 | 换手率 |
|---|---|---:|---:|---:|---:|---:|
| `v0` | 54/0 | 1.6107 | 0.1611 | 4/5 | 2/5 | 约 18.2% |
| `v1` | 50/2 | 1.2511 | -0.8032 | 6/6 | 2/6 | 约 20.9% |
| `v2` | 83/4 | 1.9627 | 0.3807 | 6/6 | 4/6 | 约 19.5% |

`v1` 是最清楚的反例：方向层面 6/6 个合约毛收益为正，但成本后只有 2/6 为正，最终分数转负。也就是说，本轮最明显的低波动退化并不是“完全学不到趋势”，而是“预测优势不足以覆盖频繁换仓成本”。

## 7. `v3` 极端负分的支持度问题

[`FineFT_two_dimensional_agent_selector.py`](../../FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py) 对每个合约先求均值，再对合约等权。这能避免长合约完全主导，但当合约极短时方差很大。

以 `v3` 最好候选之一 epoch 98/bin 5 为例，部分合约只有 9、10、13、16 步，单步净收益分别约为 `-5.75`、`-23.24`、`-11.77`、`-6.26`；其他长合约有 359–681 步。极短切片与长合约等权，显著放大了负尾部。与此同时，该候选费前合约等权收益仍约为 `-2.73`，说明模型确有方向性失败，不能通过改聚合方式把问题消掉。

因此应把 `v3` 结论拆成两部分：**方向泛化真实变差；`-4` 至 `-6` 的幅度受短样本等权放大。** `v3/s2` 连 joint rows 都不存在，也说明该区域的评估支持不足。

## 8. 受控复现实验

下一轮不要继续在 `30min_multi` 上覆盖训练。按以下顺序一次只改变一个因素：

1. **重建可比基线 A。** 使用独立 experiment name，冻结 commit、数据/feature manifest 校验和、seed、TF32 设置、epoch 范围和 selector 门槛；重新训练 150 特征基线。验收：能够在同一选择脚本下复现旧结果量级。
2. **做最小 Anchor B。** 不再强制加入全部七个特征，先只保留一个非冗余长窗口方向 anchor，优先测试 `log_price_slope_96`；其余逐组加入。conditional audit 固定对齐 `n_step=12`，将 valid 稳定性用于实验验收，而不是回写 train-only 特征选择。
3. **在 B 上单独启用 Persistence C。** 固定模型、成本和 selector，只切换 `enable_action_persistence`。重点观察四目标格的费前/费后收益、换手率、持仓持续时间和最差合约；`v1` 应是首要验收格。
4. **再处理 KL。** 若最小 anchor 后加权 KL/TD 仍远高于基线，再单独比较 `ada` 或 confidence weighting；不要与 anchor 数量、采样策略同时改变。
5. **修正评估支持度。** 为 joint cell 增加最小合约数和每合约最小 step/run 数；同时报告 contract-equal 与 step-weighted 结果，或对极短合约使用 shrinkage。该改动应作为评估敏感性分析，不能用来掩盖真实毛收益为负。
6. **最后检查 epoch 101–125。** 当前训练走到 125，但 selector 只检查 50–100。可在冻结门槛后补测，不能在看过测试收益后挑 epoch。

最低建议实验矩阵：

| 实验 | 状态特征 | persistence | 目的 |
|---|---|---|---|
| A | 原 150 维 | off | 重建可比基线 |
| B1 | A + `log_price_slope_96` | off | 验证最小 anchor 增益 |
| B2 | A + 精简且低冗余 anchor 组 | off | 验证互补信息 |
| C | 最优 B | on | 验证成本感知持仓 |
| D | 最优 C | on，调整 KL | 最后验证 teacher 冲突 |

每个实验至少使用相同的多个 seed，并固定候选 epoch 与 selector 配置。主验收指标应是四目标格的合约等权净收益和正收益合约比例；辅助指标包括费前收益、费用侵蚀、turnover、加权 KL/TD、最差合约和有效 steps/runs。

## 9. 最终判定

| 假设 | 当前判定 | 证据强度 |
|---|---|---|
| 本轮所有低波动重点格都更差 | 不成立 | 高 |
| 全局 train→valid 分布漂移导致退化 | 不支持 | 高 |
| selector 变化与基线覆盖使比较失真 | 成立 | 高 |
| persistence 未启用、成本侵蚀导致 `v1` 失败 | 成立 | 高 |
| anchor 冗余与条件不稳定增加训练难度 | 高度可疑，需消融确认 | 中高 |
| KL 冲突比旧训练显著增强 | 成立；是否为根因待消融 | 高/中 |
| `v3` 纯粹由短合约统计噪声造成 | 不成立；但幅度被放大 | 中高 |
| 当前结果足以否定 Regime Anchor 方向 | 不成立 | 高 |

本轮最重要的工程结论是：**先恢复严格可比的实验，再测试“精简 anchor”和“成本感知 persistence”各自的增量。** 在此之前继续增加特征、改变 selector 或调 KL，只会让下一次结果更难解释。
