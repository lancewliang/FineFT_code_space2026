# 低层候选模型选拔证据与当前 Slot 持仓切片

日期：2026-10-28  
范围：`analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection` 的既有选拔结果、所选/备选 epoch 原始评估文件，以及冻结 `expert_hysteresis` 结果中的 77 个入场回合。只读取既有产物；没有替换模型、改配置或改冻结结果。

后续进展：原各槽位第二、第三名的18组完整路由对照已完成。更换其中3位专家的组合取得40.643%收益、3.583%最大回撤，仍属valid开发研究。见[最新结果、盈亏分解和优化顺序](2026-10-28-low-level-optimization-results.md)。

## 结论

当前九个 Slot 都是从有效候选中按联合评分选出的第一名；manifest 记录 418 个完整候选、没有不完整候选被排除，候选排名表对每个 Slot 都有 418 行。第二、第三名并非臆测：下表及附带 CSV 逐一列出了排名、模型权重路径、`bin_index`、分数与门禁。所有表中前三名均通过当前门禁。

Slot 6 当前 `epoch_60:bin_9` 的 PairScore 为 0.79525，第二名 `epoch_58:bin_3` 为 0.75031；在选拔评估的 volatility/slope 两种上下文，第二名止损 episode 比例为18.55%/20.47%，当前模型为25.20%/16.80%。它在一个上下文更低、另一个更高，不能仅凭其中一个比例说第二名更好。Slot 8 当前 `epoch_35:bin_4` 分数0.74000，第一名对应的两个上下文止损率38.55%/39.13%，高于manifest中配置的30%门槛；这是因为硬止损门禁当前明确关闭。第二、第三名的两种上下文止损率也在35%–43%之间，均没有通过30%门禁。若仅按现有排名和门槛启用这条门禁，Slot 8 会改选第四名 `epoch_71:bin_2`（分数0.63411；26.09%/29.41%），而 Slot 3 也会由 `epoch_55:bin_8` 改为 `epoch_60:bin_7`（0.03543；27.50%/21.21%）；九个 Slot 仍都有候选。该计算只是在既有表上应用门禁，不是生产变更或真实回放结果。

77 个开仓回合按入场时 `queried_slot` 切片：Slot 6 有24个回合（含1个期末未平），23个已平中5个止损、9个盈利，已平净额+¥365.41；Slot 8有9个已平，1个止损、4个盈利，净额+¥228.91；Slot 5有5个已平，2个止损、3个盈利，净额-¥42.60。此切片描述“开仓时由哪个低层 Slot 被查询”，不是模型在全时段独立造成的收益：模型会在持仓及高层状态变化中被不同频率调用，入场后持仓跨越多个路由状态，余额/止损/费用也共同决定回合损益。不能据此断言 Slot 6 的模型贡献了+¥365，或 Slot 5 模型造成了亏损。

## 按 Slot 排名的已选与可替代候选

`PairScore = min(LCB_volatility_run, LCB_slope_run)`，是两个候选评估上下文中较低的置信下限，不是人民币收益、收益率或胜率。表中每项均为 `候选名 (PairScore)`；rank 2/3 是可复核的合格候选，不是建议必换的模型。具体路径、合约支持数和两侧止损率见[候选证据 CSV](low_level_candidate_evidence_20261028/top_candidates_by_slot.csv)。

| Slot | 流量层级 | 当前已选 | 合格第 2 名 | 合格第 3 名 | 当前选择门禁 | 当前持仓入场切片：已平 / 止损 / 盈利 / 净额 |
|---:|---|---|---|---|---|---|
| 0 | core | `epoch_55:bin_6` (.26556) | `epoch_60:bin_4` (.25889) | `epoch_55:bin_5` (.23229) | 全部通过 | 2 / 1 / 1 / -¥54.12 |
| 1 | core | `epoch_38:bin_0` (.29152) | `epoch_46:bin_0` (.25409) | `epoch_48:bin_6` (.25179) | 全部通过 | 2 / 0 / 2 / +¥99.22 |
| 2 | core | `epoch_39:bin_1` (.37300) | `epoch_39:bin_5` (.37175) | `epoch_40:bin_5` (.36804) | 全部通过 | 3 / 0 / 3 / +¥591.27；另1个期末未平 |
| 3 | core | `epoch_55:bin_8` (.16156) | `epoch_53:bin_8` (.15935) | `epoch_55:bin_5` (.13419) | 全部通过 | 10 / 1 / 6 / +¥289.60 |
| 4 | core | `epoch_66:bin_1` (.29143) | `epoch_69:bin_1` (.26470) | `epoch_48:bin_7` (.26298) | 全部通过 | 5 / 1 / 3 / -¥42.99 |
| 5 | peripheral | `epoch_46:bin_7` (.34608) | `epoch_47:bin_7` (.31229) | `epoch_49:bin_7` (.28209) | 全部通过 | 5 / 2 / 3 / -¥42.60 |
| 6 | core | `epoch_60:bin_9` (.79525) | `epoch_58:bin_3` (.75031) | `epoch_59:bin_9` (.74469) | 全部通过 | 23 / 5 / 9 / +¥365.41；另1个期末未平 |
| 7 | peripheral | `epoch_60:bin_0` (.39292) | `epoch_47:bin_5` (.38978) | `epoch_57:bin_0` (.38197) | 全部通过 | 16 / 0 / 11 / +¥551.10 |
| 8 | core | `epoch_35:bin_4` (.74000) | `epoch_46:bin_5` (.67462) | `epoch_35:bin_5` (.67002) | 当前选拔门禁通过；**硬止损门禁未启用** | 9 / 1 / 4 / +¥228.91 |

路径模板为 `result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/epoch_{epoch}/trained_model.pkl`，bin 取候选 ID 中的整数。附带 CSV 已逐行校验所有前 3 候选的权重文件和 `volatility/analysis_result.csv`、`slope/analysis_result.csv` 均存在。

## 门禁含义与只开现有止损门禁的敏感性

manifest 配置 `selection_method=two_dimensional_marginal_and_dual_context_lcb`、`lcb_z=1`、按合约等权聚合，收益指标为 `reward`。候选要有不少于3个联合合约；两个联合上下文平均收益和LCB都须大于0；最差初始仓位收益及正收益合约比例要达线；volatility 与 slope 边际均值须大于0；当主力评估步数达到100时，主力平均净收益须大于0。Slot 5、7属 peripheral，其余属 core；两种层级在当前 manifest 都要求至少3个联合合约和至少60%正收益合约。各 Slot 的最差初始仓位门槛随 volatility label 变化，精确配置见 selection manifest。全部前三名 `eligible=True`，因此在当前选择器规定下是确实可替代的模型。

止损门禁虽然记录 `hard_stop_loss_threshold=0.03` 和 `max_hard_stop_loss_episode_ratio=0.3`，但 `enable_hard_stop_loss_gate=false`，所以止损率不参与当前排名淘汰。只对已有排名加上“两上下文 stop episode ratio 均不超过30%”，并在剩余合格模型里按原分数排序，得到：

| Slot | 当前选择 | 仅加30%门禁后的首选 | 新模型分数 | 当前→新模型两上下文止损率 | 变化 |
|---:|---|---|---:|---|---|
| 0 | `epoch_55:bin_6` | 不变 | .26556 | 12.09%/14.61% | 无 |
| 1 | `epoch_38:bin_0` | 不变 | .29152 | 10.47%/3.09% | 无 |
| 2 | `epoch_39:bin_1` | 不变 | .37300 | 21.57%/9.30% | 无 |
| 3 | `epoch_55:bin_8` | `epoch_60:bin_7` | .03543 | 50.00%/33.96% → 27.50%/21.21% | 换至最低合格模型；分数降.12613 |
| 4 | `epoch_66:bin_1` | 不变 | .29143 | 26.19%/21.95% | 无 |
| 5 | `epoch_46:bin_7` | 不变 | .34608 | 22.58%/10.00% | 无 |
| 6 | `epoch_60:bin_9` | 不变 | .79525 | 25.20%/16.80% | 无 |
| 7 | `epoch_60:bin_0` | 不变 | .39292 | 12.12%/2.70% | 无 |
| 8 | `epoch_35:bin_4` | `epoch_71:bin_2` | .63411 | 38.55%/39.13% → 26.09%/29.41% | 分数降.10589 |

两种新模型路径为 `epoch_60/trained_model.pkl`（bin 7）和 `epoch_71/trained_model.pkl`（bin 2）；它们通过其余已有联合门禁。该表仅是候选排名过滤敏感性，没有验证路由重构后的收益、MDD或持仓状态效应。所选模型的静态止损比率偏高并非当前生效的拒绝条件；manifest 中这项门禁当前关闭。选择器里只有在 `enable_hard_stop_loss_gate` 打开时才把 max ratio 纳入 eligible 条件。[候选 manifest](../../analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_selection_manifest.json)、[候选排名](../../analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_candidate_rankings.csv)、[门禁实现](../../FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py)

## 与 77 个入场回合的关联

如下为冻结基线中“开仓时 queried Slot”的 episode 汇总，来自此前审计关联后留存的 [entry feature CSV](fu_10min_entry_20261028/entry_features.csv)。净额只计已平回合；期末未平的 mark-to-market 单列，避免将浮动 PnL 混成已实现。数据能回答哪些 Slot 的开仓回合中出现了止损、赢家或亏损，但不能把后续持仓损益单独归给开仓时模型。

| Slot | 开仓回合 | 已平 | 止损 | 盈利回合 | 其它非盈利已平 | 已平净额 | 期末未平浮动额 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 2 | 1 | 1 | 0 | -¥54.12 | — |
| 1 | 2 | 2 | 0 | 2 | 0 | +¥99.22 | — |
| 2 | 4 | 3 | 0 | 3 | 0 | +¥591.27 | +¥104.85 |
| 3 | 10 | 10 | 1 | 6 | 3 | +¥289.60 | — |
| 4 | 5 | 5 | 1 | 3 | 1 | -¥42.99 | — |
| 5 | 5 | 5 | 2 | 3 | 0 | -¥42.60 | — |
| 6 | 24 | 23 | 5 | 9 | 9 | +¥365.41 | -¥9.90 |
| 7 | 16 | 16 | 0 | 11 | 5 | +¥551.10 | — |
| 8 | 9 | 9 | 1 | 4 | 4 | +¥228.91 | — |

Slot 6/5/8 runner-up 与 Slot 3/8 止损门禁的状态机回放已完成，汇总见 [runner-up replay_summary.csv](fu_10min_selection_20261028/runner_up_probe/replay_summary.csv) 和 [stop-gate replay_summary.csv](fu_10min_selection_20261028/stop_gate_probe/replay_summary.csv)。Slot 6 第二名收益较高但回撤升至5.39%；Slot 5 第二名收益下降；Slot 8 第二名收益近似但回撤升至5.10%。应用止损门禁的 Slot 3/8 单独或组合回放均低于原基线。这里不再等待或重跑这些实验。

## 原始产物与限制

- 真实开仓标签和 PnL 来自冻结 `expert_hysteresis` run 的 `valid/main_sub/contracts/*/audit_trades.json`、`audit_routing.json`、`audit_predictions.json`，与 `portfolio_action_timeline.csv` 连接后生成 entry 特征表；原始研究过程和字段说明见[入场归因报告](2026-10-28-expert-hysteresis-entry-research.md)。此处真实归因按 episode 开仓时 `queried_slot` 切片，不是对所有逐步调用低层网络的因果收益分解。
- 选择候选和 77 笔策略表现都使用 valid 数据。它们并非未见样本，排名可能已对 valid 做过选择；不能把 PairScore 或候选筛选结果当成独立的实盘期望值。所有候选的三个排名前位均通过门禁，只能说明在既定评估与门槛下合格，不代表替换能提高目标收益/回撤。
- 复现分析 CSV 的脚本为 [`build_candidate_evidence.py`](low_level_candidate_evidence_20261028/build_candidate_evidence.py)，输出 [前三名候选表](low_level_candidate_evidence_20261028/top_candidates_by_slot.csv) 和 [入场切片表](low_level_candidate_evidence_20261028/realized_entry_attribution_by_slot.csv)。脚本按仓库要求在 `finetf` conda 环境运行；不修改业务代码和冻结结果。
- 选择与门禁来源：[selection manifest](../../analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_selection_manifest.json)、[selection CSV](../../analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_selection.csv)、[rankings CSV](../../analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_candidate_rankings.csv)、[joint metrics](../../analysis_result/DiHFT/low_level/fu/10min_parallel/two_dimensional_selection/two_dimensional_joint_metrics.csv)、以及组装的 `model.pth`。manifest 指明 `high_level_model_change=not_performed`，即低层选择产物没有改高层模型。
- 单个候选原始评估来自候选 epoch 目录下的 `volatility/analysis_result.csv` 和 `slope/analysis_result.csv`；例如当前 Slot 6 的 [`epoch_60` volatility eval](../../result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/epoch_60/volatility/analysis_result.csv)、[`epoch_60` slope eval](../../result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/epoch_60/slope/analysis_result.csv)，候选 bin 是9。对应的模型为 `epoch_60/trained_model.pkl`。其它所选/替代模型的 epoch/bin、两侧原始评估路径在候选证据 CSV。

## 附录：实际问到的专家与离线市场标签是否一致

### 通俗结论

在冻结配置 `7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb` 的 valid 记录中，离线标签能够覆盖全部 **2,219 次实际专家查询**和全部 **77 次真实开仓**。把波动标签和趋势标签合成九宫格后：

- 每次查询都逐时刻完全一致的有 **292 / 2,219（13.16%）**；
- 77笔开仓中完全一致的有 **11 / 77（14.29%）**；
- 11笔硬止损中一致 **1笔（9.09%）**；42笔盈利回合中一致 **6笔（14.29%）**。

这不是“专家只有13%的预测准确率”，也不能据此说“86%的时候派错了专家”。这只是两套不同东西在同一时间点的标签相同率：离线标签按历史整段走势切片而来，专家选择则看模型当前分数，并由迟滞规则允许当前专家继续工作一段时间。两者设计目标不等同。特别是生成离线段落标签会用双向滤波、转折点识别和段落合并，不能保证这些标签在当时已经可知；它们只能用于事后诊断，不能拿来直接改成实时路由信号。[ADR-0013](../../adr/0013-unified-segment-regime-calibration.md) [迟滞路由实现](../../FineFT/RL/DiHFT/high_level/gating/expert_hysteresis.py)

输赢两组的完全匹配比例很接近（止损9.09%、盈利14.29%），且样本数不大；仅凭这个差别不能认为“匹配就更容易盈利”。匹配的10笔已平回合合计净额 **+¥213.79**，未匹配的65笔已平回合合计 **+¥1,772.01**。未匹配组里也包含10笔止损和36笔盈利。这个分组描述已发生的结果，不代表离线标签有预测作用，也不证明应该强制切换专家。

### 分轴看

| 比较范围 | 波动轴标签相同 | 趋势轴标签相同 | 九宫格完全相同 |
|---|---:|---:|---:|
| 全部2,219次专家查询 | 696（31.37%） | 1,031（46.46%） | 292（13.16%） |
| 全部77次开仓 | 26（33.77%） | 31（40.26%） | 11（14.29%） |
| 11笔硬止损 | 3（27.27%） | 4（36.36%） | 1（9.09%） |
| 42笔盈利回合 | 18（42.86%） | 15（35.71%） | 6（14.29%） |

分轴数字说明完整九宫格相同要求两边都对齐，所以比例会低于任意单轴的比例。它们仍是事后描述，不是模型准确率。

### 对齐方法与复核

将每个合约、每个时间戳的 `volatility_label` 与 `slope_label` 从 `dataset/10min/fu/valid/{volatility,slope}/<contract>/label_i/*.feather` 合并。对齐键为 **合约 + 时间戳**；两轴组成的离线 Slot 按冻结 manifest 和路由代码的规则计算：`volatility_label * 3 + slope_label`。每个输入 feather 的目录 `label_i` 还与行内标签列做了核对。实际路由使用 `audit_routing.json` 中的 `queried_slot`；这2,219行的 `queried_slot` 与 `active_slot_after` 全部一致。77笔真实开仓再按 `entry_features.csv` 里的 `entry_decision` 与审计记录核对，合约、时间戳、Slot 均一致。冻结配置哈希已写入机器可读汇总。

| 样本 | 有离线双轴标签 | 完全匹配 | 不匹配 |
|---|---:|---:|---:|
| 实际专家查询 | 2,219 / 2,219 | 292 | 1,927 |
| 真实开仓 | 77 / 77 | 11 | 66 |
| 硬止损开仓 | 11 / 11 | 1 | 10 |
| 盈利开仓 | 42 / 42 | 6 | 36 |

专家会因迟滞保持原槽，也会等候候选连续确认后再切换；相关状态逻辑见 [`expert_hysteresis.py`](../../FineFT/RL/DiHFT/high_level/gating/expert_hysteresis.py)。因此离线标签短期变化时，当前专家继续工作并不自动等于路由缺陷。诊断没有重放“如果每根K线都切去离线标签对应槽位会怎样”，因为该标签可能含未来信息，这种重放会把不可实时获得的信息当成策略输入。

完整逐次结果及脚本在 [`label_alignment/`](fu_10min_selection_20261028/label_alignment/)：

- [`alignment_summary.json`](fu_10min_selection_20261028/label_alignment/alignment_summary.json)：含 run ID、冻结配置哈希、各组覆盖率、分轴匹配率及 PnL 汇总；
- [`all_expert_queries_with_offline_labels.csv`](fu_10min_selection_20261028/label_alignment/all_expert_queries_with_offline_labels.csv)：2,219次查询的逐行对照；
- [`entry_episodes_with_offline_labels.csv`](fu_10min_selection_20261028/label_alignment/entry_episodes_with_offline_labels.csv)：77笔开仓的逐笔对照；
- [`analyze_label_alignment.py`](fu_10min_selection_20261028/label_alignment/analyze_label_alignment.py)：可复算脚本。按项目要求使用 `conda activate finetf` 执行，未改生产代码、冻结结果或 test 数据。
