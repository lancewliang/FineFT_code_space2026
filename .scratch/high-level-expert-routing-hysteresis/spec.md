# 高层专家路由迟滞与步骤 7–10 全链路升级实施规格

Status: ready-for-agent
Date: 2026-10-24
Updated: 2026-10-25
Decision: [ADR-0062](../../docs/adr/0062-stateful-expert-routing-hysteresis-and-frozen-score-calibration.md)
Pipeline decision: [ADR-0063](../../docs/adr/0063-in-place-expert-routing-pipeline-upgrade.md)
Design: [详细设计与 Q1–Q11 共识](../../docs/train3/high_level_routing_hysteresis_design.zh_cn.md)

## Problem Statement

研究人员希望高层路由对继续使用当前专家采用较宽松条件，对切换过去采用更严格条件，减少临界分数造成的专家反复切换，同时保留专家对最新行情的响应。

目标项目现有双轴门控只看各轴最高分，并按是否持仓决定宽松阈值；普通动作持续性还会跳过策略推断。它们不能表达“当前专家仍有效就每步重新决策”，也不能防止一个已经失效的专家被其他专家的高分间接保活。旧研究工作树的单轴示例与本项目的二维 Slot、风险分层和空仓降级槽位不兼容。

本规格保留已经完成的 Q1–Q11 行为共识。2026-10-25 用户将范围调整为步骤 7–10 全面升级、不再向后兼容，并明确确认“解除冻结，原文件直接升级”。此前的旧三文件冻结、现有生产源码不改、禁止导入两个旧模块、必须另建 expert 三文件及维护旧策略兼容路径的约束全部被本轮决定取代；不把文档更新当作已完成实现。

同日用户进一步要求：expert_hysteresis 使用的两轴窗口、两轴 gamma 依然需要一起搜索。本条取代原 Q9 中“校准期间固定窗口和衰减”的限制；模型/参考分布仍冻结，评分参数与迟滞参数在独立校准段联合选定，最佳 Trial 确定后全量冻结供步骤 8–10 使用。

## Solution

将现有高层流水线统一升级为 expert_hysteresis 门控策略，围绕当前专家身份保存状态。当前专家对应双轴评分达到保持门槛时继续每步决策；新候选达到更高进入门槛，并在发生改变的轴足够领先、连续确认后接管。当前专家任一轴失效立即撤销资格，没有完成确认的替代者则规则接管。

初次启用、规则恢复、专家切换共用 K；恢复不额外要求领先第二名。候选仍连续合格时，其证据可跨普通专家失效延续。两轴 Slot 原子切换，保留既有硬风控、空槽 No New Risk、缓存、轻量调参和最终回放约束。

## User Stories

1. As a strategy researcher, I want steps 7–10 to use one explicit expert-hysteresis strategy and versioned experiment contract, so that calibration, visualization and final evaluation execute the same configured behavior.
2. As a strategy researcher, I want the active expert identified independently of the current position, so that Flat output does not erase expert ownership.
3. As a strategy researcher, I want the retained expert to evaluate each fresh observation, so that route stability does not freeze an old trade action.
4. As a strategy researcher, I want both slope and volatility evidence checked for the active Slot, so that another expert cannot conceal its loss of applicability.
5. As a strategy researcher, I want lower holding thresholds and higher entry thresholds, so that continued use and new control require different evidence.
6. As a strategy researcher, I want an expert invalidated on the first smoothed-score breach, so that candidate confirmation does not keep an invalid expert alive.
7. As a strategy researcher, I want a candidate to clear both entry thresholds, so that a relatively best but unsuitable expert cannot take control.
8. As a strategy researcher, I want each changed axis to lead its incumbent counterpart by Δ, so that small score fluctuations do not replace the current Slot.
9. As a strategy researcher, I want unchanged axes exempted from self-comparison, so that a legitimate one-axis transition can complete.
10. As a strategy researcher, I want one candidate Slot to qualify for K consecutive decisions, so that unrelated experts cannot pool evidence.
11. As a strategy researcher, I want Slot changes to be atomic, so that independently confirmed axes cannot synthesize an unconfirmed expert.
12. As a strategy researcher, I want candidate changes and interruptions to restart confirmation, so that old evidence is never reused incorrectly.
13. As a strategy researcher, I want valid candidate evidence retained across ordinary incumbent invalidation, so that loss of the old expert does not impose an artificial delay.
14. As a strategy researcher, I want a fully confirmed candidate to take over on the same invalidation step, so that a redundant rule-only step is unnecessary.
15. As a strategy researcher, I want the rule branch used when no expert is ready, so that the system does not continue an invalid expert.
16. As a strategy researcher, I want entry and recovery to require no runner-up margin, so that close but suitable regimes can regain control after persistent evidence.
17. As a strategy researcher, I want first entry, recovery, and switching to share K, so that the initial search space remains manageable.
18. As a strategy researcher, I want the active expert retained when tied for best, so that ties cannot trigger replacement.
19. As a strategy researcher, I want an existing tied-best candidate preferred, so that index ordering cannot repeatedly destroy its confirmation.
20. As a strategy researcher, I want deterministic ordering when no tied candidate exists, so that identical replay inputs produce identical routes.
21. As a risk reviewer, I want empty-model Slots to retain No New Risk semantics, so that hysteresis cannot turn an unavailable expert into permission to trade.
22. As a risk reviewer, I want hard stops, cooldowns, circuit breakers, and non-main restrictions preserved, so that route memory cannot bypass risk control.
23. As a risk reviewer, I want no stale expert inference in an empty-model interval, so that graceful holding is not confused with expert authorization.
24. As a backtest operator, I want invalid scores and incompatible Slot mappings to stop the run, so that scoring failures cannot masquerade as low confidence.
25. As a backtest operator, I want route state reset between independent contract runs, so that one contract cannot influence another's confirmation.
26. As an experiment owner, I want the two scoring windows and two gamma values jointly searched with hysteresis parameters while model versions and reference distributions remain fixed, so that the selected trial contains the complete scoring and routing configuration actually used.
27. As an experiment owner, I want the current train/valid/test data contract to be explicit, so that valid-based joint selection and routing calibration are not mislabeled as independent periods.
28. As an experiment owner, I want test parameters frozen, so that final performance is evaluated without further selection.
29. As an experiment owner, I want quantile caches reused without sharing mutable route state, so that memory savings do not cause cross-run contamination.
30. As an experiment owner, I want lean trials and one complete best-trial replay, so that calibration preserves the existing resource model.
31. As an auditor, I want every routing event linked to its Decision ID and genuine prediction provenance, so that route changes can be separated from execution changes.
32. As an auditor, I want holding, confirmation, replacement, invalidation, and rule causes distinguished, so that replay explains each chosen action.
33. As an evaluator, I want switching, delay, friction, net results, and drawdown reported together, so that reduced switching is not mistaken for guaranteed improvement.
34. As a developer, I want tests at the existing public router boundary, so that requirements are verified through observable decisions and actions.

## Implementation Decisions

- Upgrade the existing routing implementation and steps 7–10 in place to expert_hysteresis. Existing filenames may retain vae because VAE scoring remains part of the method; they must execute the upgraded implementation. Update shared factories, parameter definitions, result consumers and tests as required. Maintaining absolute/hierarchical execution paths or legacy parameter/artifact parsers is not an acceptance requirement; legacy or incomplete inputs must fail explicitly rather than silently map to the new strategy.
- The state belongs to one independent contract rollout. It contains active Slot identity, pending candidate identity, candidate count, stable axis/Slot identities, and a decision reason. Shared read-only quantile data never carries this mutable state.
- The target architecture remains dual-axis. A Slot is the existing ordered volatility/slope pair from the Selection Manifest; do not introduce a single-VAE expert list or invent a joint profitability score.
- For each axis define H as the holding threshold and E as the entry threshold, with 0 <= H < E <= 1. Reuse the existing exit-ratio expression H_axis = exit_ratio × E_axis, requiring 0 < exit_ratio < 1 for the new strategy. Persist both configured and derived values.
- Validate public inputs as nonempty finite quantile vectors in [0, 1], with stable axis lengths, label ordering and Slot mapping throughout the independent rollout. Invalid input raises an explicit error; no silent clipping, NaN fallback or continuation as an ordinary defensive result.
- Evaluate current applicability using the active Slot's own two scores. Both must be at least their H. A score below H on either axis revokes the active expert on that decision step, independently of other experts' scores and independently of the account being Flat.
- Build the highest-score Slot set from the Cartesian product of the two axes' maximum-score sets. Prefer the valid active Slot, then the existing candidate if still a member, then deterministic Slot order. Do not switch tied active experts.
- An ordinary replacement candidate must clear both E thresholds. Every changed axis must strictly exceed its active counterpart and meet the same nonnegative Δ. An unchanged axis does not compare against itself, but must still meet its entry requirement.
- Without a valid incumbent, first entry or recovery only requires both entry thresholds and continuity of the same highest-score Slot. Do not apply a runner-up margin gate from the hierarchical strategy to this path.
- First entry, recovery, and replacement use the same positive integer K. Count only consecutive eligible decision steps of the same candidate Slot; a different eligible candidate starts at 1 and an ineligible step clears the pending evidence.
- Ordinary incumbent invalidation does not itself clear still-valid candidate evidence. If the same candidate completes K on that step, it takes control immediately without a forced intermediate rule action.
- Completed confirmation clears the pending candidate state. If confirmation is incomplete, retain a valid incumbent; otherwise route to the target project's existing rule branch.
- Exact threshold equality is inclusive: score equal to H remains valid, score equal to E can qualify, and a positive lead equal to Δ qualifies. Δ = 0 still requires strict superiority. Numerically boundary-focused tests must verify the chosen comparison contract.
- Retaining an ordinary model Slot must query the selected expert with the latest observation, true previous action, account state and available-action mask on every eligible step. A nonzero legacy persistence counter must not skip that query or force repetition of the previous micro action in this strategy.
- This requirement partially supersedes legacy nondefensive inference-skipping only for the new strategy. It does not retrain low-level models, rewrite Reward Shaping, or change DP Q tables or physical account balances.
- A selected dominant empty_model Slot is not an eligible ordinary expert. Preserve its existing No New Risk and finite graceful-hold-exit behavior; never fit a second-best model merely to fill the Slot, run a stale expert during the interval, or refresh the empty-slot hold buffer indefinitely.
- Hard stops, circuit breakers, directional cooldown, unavailable-action handling and non-main restrictions retain their existing priority. No K-step wait delays their required action. Risk/empty-slot blocking does not count as a qualifying candidate step and cannot silently join confirmation streaks across the interruption.
- Independent rollout initialization and reset clear route state. Existing score warm-up remains; after warm-up the initial candidate must complete normal entry confirmation.
- Calibration freezes VAE and specialist versions, Selection Manifest and reference score distributions. Jointly search slope_window_length, volatility_window_length, slope_gamma, volatility_gamma, both axis entry thresholds, the exit ratio, Δ and K: nine effective dimensions. H is derived from E and the exit ratio, not sampled separately. The four scoring parameters must actually affect the trial's scoring/smoothing path; do not silently fix them or sample unused parameters. Search ranges and validity constraints are explicit configuration, not invented optimal values.
- Use the existing three-way contract: train for fitting, valid for low-level selection plus routing calibration, and test for frozen final evaluation. Do not describe the valid-based selection/calibration result as independent; final test outcomes cannot update chosen parameters.
- Parameter parsing, trial records, best-parameter replay and final evaluation must preserve strategy identity/version, each trial's four scoring parameters, all thresholds, Δ and K without reverting to defaults. Once the best trial is selected, its complete scoring and routing settings are frozen for steps 8–10.
- Reuse quantile caching, lean trials and best-trial replay. The routing search-space manifest declares only the nine search dimensions; runtime input discovery and internal source identity calculation remain in code. Calibration may suppress large histories, while the best-trial replay persists complete routing/audit evidence.
- Extend existing routing attribution using the same Decision ID and Prediction Record rules. Record before/after active Slot, candidate, count before/after update, the just-completed candidate when count resets, per-axis scores, thresholds, event reason, genuine inference source and final action.
- Distinguish observed dominant Slot from active expert and defensive state in the audit output. Do not count an expert switch as a trade, fabricate a Prediction Record during a rule-only step, or charge an additional switching fee outside actual simulated execution.
- Persist candidate ordering and experiment identities so that fixed input replay is deterministic. Score generation variability must be controlled with the same cache or a frozen scoring protocol when comparing router variants.
- This spec is the behavioral reference for a separately initiated future implementation task. The current authorization remains documentation only: no Python migration, training, calibration, test run or live deployment.

### 2026-10-25 已确认的原地升级范围

用户明确要求所有后续步骤升级，并确认解除此前冻结。本节取代此前独立新建 expert 三文件方案；保留脚本路径作为操作入口，不保留旧执行语义或旧格式兼容分支。

| 步骤 | 原地升级的 shell 入口 | 必须同步升级的实现与职责 |
| --- | --- | --- |
| 7. 高层 Optuna 并行寻优 | `FineFT/script/test/DiHFT/high_level/vae_optuna_fu_10.sh` | `vae_routing_optuna.py`、`vae_routing_util.py`、门控工厂与公共参数定义：联合搜索两轴窗口、两轴 gamma 与迟滞参数，导出完整实验配置、Trial 选择证据与最优完整回放 |
| 8. 高层路由筛选与可视化 | `FineFT/script/analysis/pick_agent/high_level_heurstic_fu_10.sh` | `FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py`：读取新配置及 Trial 身份，完整导出、诊断与绘图，不能仅替换输入目录 |
| 9. 主副合约时间同步评估 | `FineFT/script/test/DiHFT/high_level/final_result_fu_10_main_sub.sh` | `vae_routing_final_result_macro_action.py` 与时间同步组合诊断：重载同一冻结配置，报告主/次主力、降级与路由事件，结果按阶段独立保存 |
| 10. 最终测试与评估 | `FineFT/script/test/DiHFT/high_level/final_result_fu_10.sh` | 同一升级后的最终评估实现及诊断/审计消费者：只在 test 上评估冻结配置，不重新选参，输出可追溯的最终证据 |

- 三个原 VAE 文件可以直接修改；不再要求内容哈希不变或另建 `expert_routing_util.py`、`expert_routing_optuna.py`、`expert_routing_optuna_fu_10.sh`。后续实现选择拆分新模块也允许，但不是并行维护旧版的兼容方案。
- 共享门控工厂、公共参数常量、筛选器、最终评估与诊断/审计模块属于必要的传递升级范围；不因此扩大到低层训练、专家选拔门槛、Reward Shaping、DP 表或账本语义变更。
- 不再要求新执行链禁止导入名为 `vae_routing_util`、`vae_routing_optuna` 的模块；要求这些模块已经升级并执行 expert_hysteresis，不回退到旧分支。
- 历史产物仍须保留。取消代码兼容不授权删除历史数据库、覆盖旧结果或把旧缓存当作新评分协议；使用新的 schema/实验身份及输出空间，发现不兼容数据明确报错。
- 本轮继续只更新规格与设计文档；实现、训练、校准、回测及部署仍是后续单独任务。

### 步骤 7–10 共用配置与产物契约

- 使用一份带 `schema_version` 的完整已解析配置作为执行依据，不能继续把参数目录名或不完整 `high_level_agent_para.txt` 当作唯一配置。配置包含策略标识/版本、两轴 E/H、退出比率、Δ、K、该 Trial 实际使用的两轴窗口/gamma、模型/Manifest/参考分布/缓存身份、数据时间边界、账户资金与单位、费用、风险约束、候选排序，以及 study、Trial、选参目标及配置哈希。CSV 与可读文本是该配置的投影，重载不得丢值或重新猜测；最佳 Trial 选定后，四项评分参数与迟滞参数一起冻结。
- 后续 shell 的内置默认值不得覆盖已选配置；正式评估发现费用、初始资金、硬止损、冷静期、断路器、非主力限制等与校准协议不一致时，明确拒绝或要求以不同实验身份另行评估，不能冒充同一冻结结果。步骤 9/10 当前分别存在 6000/5000 初始资金和不同风险默认值，需在此次升级中消除静默漂移，不在文档中捏造新最优数值。
- 按实验身份、阶段（校准/valid 与 test）、用途（best replay、筛选展示、main_sub、最终评估）隔离输出；步骤 8 不能把验证轨迹复制到最终测试根目录，步骤 9 切换阶段也不能覆盖另一阶段结果。步骤 10 只消费已冻结配置，不从 test 指标推导新参数。
- 步骤 8 显式校验 study/Trial/数据段/配置身份，不能只匹配 `trial_number` 或扫描现存轨迹目录。FAILED、PRUNED 或不完整结果不能作为正常候选。lean Trial 没有全量轨迹是正常情况，不能因此把它排除出有效选参证据。
- 用户已确认步骤 8 直接沿用步骤 7 按预先固定目标选出的最佳 Trial，只做校验、导出与展示。步骤 7 明确记录目标版本、优化方向、回撤惩罚/交易次数 guard 等有效定义及原始绩效，步骤 8 不再按原始 tr、Sharpe 或其他指标换选 Trial；多指标排名只作诊断，不能改变配置发布结果。缺少选定 Trial 的完整回放时明确报告不完整，不从其他现存目录挑替代者。
- 保留既有 `contracts/<contract>/` 的历史与审计语义，并增加独立的路由事件证据，区分主导 Slot、当前专家、候选、K 计数、规则接管、风险/空槽与真实成交。收益/B&H 图之外，要能沿同一事件时间轴检查上述迟滞状态、双轴分数与 H/E 边界；不能只画 macro argmax。
- 行情、收益与路由事件通过明确的合约、时间/步身份对齐；预热和尾部结算的差异需显式记录，不能用最短长度静默截断掩盖错位。诊断与审计使用实际阶段的行情及 TradingDay，资金、费率、合约单位来自冻结配置。
- 冻结模型、Selection Manifest、参考分布及身份匹配的评分缓存继续只读复用，不共享路由状态；缓存缺失或不兼容时在新实验空间生成或明确失败，不覆盖历史缓存。
- 当前缓存保存逐时点原始 VAE 分位数，窗口/gamma 平滑在各 rollout 内执行；来源身份一致时可以跨九维搜索的 Trials 只读复用原始缓存。不得共享平滑 deque 或其状态；若保存平滑结果，缓存身份还必须包含两轴窗口/gamma、平滑算法及边界规则。重配置时应用该 Trial 的窗口、权重与预热规则并清空平滑/路由状态，不能携带上一 Trial 的窗口或历史证据。
- 原始缓存身份核验覆盖运行时发现的模型/参考分布、数据内容、特征/标签顺序、Selection Manifest/Slot 映射及内部生成协议/schema，不能只凭缓存文件存在判定兼容。各 Trial 使用 valid 分区并记录窗口相关的预热覆盖，不把少计分的差异隐入目标值。

阶段约束：步骤 9 默认用于已指定校准/验证段的主副合约同步诊断；如显式运行 test，同样只能读取冻结配置并输出独立的 test/main_sub 证据，不成为步骤 10 之前的第二次调参机会。

### 2026-10-25 Shell 与总控维护补充

用户再次明确列出的 `vae_optuna_fu_10.sh`、`high_level_heurstic_fu_10.sh`、`final_result_fu_10_main_sub.sh` 都是必须维护升级的操作入口，连同已确认的 `final_result_fu_10.sh` 一起验收。所列“名称|脚本|日志”三行来自根目录 `run_fu_10min_pipeline.sh`，该总控的步骤清单、调用、日志提示和帮助文档也纳入维护范围，不能只升级 Python。

本轮用户进一步明确“第10步 FineFT/script/test/DiHFT/high_level/final_result_fu_10.sh 也需要升级”，并接受 Q2 推荐：四个子 shell 统一以前台完成态返回，需要后台运行时由使用者从外部启动整个总控或整个子 shell。

已核实的维护缺口与实施要求：

- 总控目前只有九步，且步骤/区间解析限定单个数字。用户已确认升级后无参数默认执行 1–10；只跑验证时显式指定 `1-9`。总控须能显式调用第 10 步，并支持 `7-10`、`-s 7,8,9,10`；帮助、步骤清单和 README 同步说明该默认行为。步骤 10 仍只消费冻结配置，最终 test 不反馈选参。
- 总控当前默认导出 `FOREGROUND=1`，但可被环境覆盖；步骤 8 的后台分支启动后立即返回，总控的同步 `bash` 调用不能据此判断实际完成。按已确认 Q2，升级后四个子 shell 统一等待寻优、配置发布、评估及必需诊断/审计完成，再返回真实状态；不保留仅启动内部后台任务便成功退出的分支。内部 worker 并行可保留，但必须等待并核验其终态；失败或中断不能继续消费旧产物。
- 总控日志清单写死 `fu/10min_parallel` 和 `final_result_valid.log`，Optuna 子脚本日志还写死品种 `fu`；步骤 8 前台分支没有写入其清单所示日志。运行输出和清单必须使用同一解析后的实验、阶段、用途与日志路径，遵守已有新产物隔离约束，不能指向旧路径或另一阶段。
- 子 shell 与总控必须实际使用 `finetf` 环境；环境不可用时明确失败，不吞掉激活错误后调用其他 Python。参数应无损传递，包含空格的路径在前台/后台执行时不能因字符串拼接改变含义。
- 总控须获得真实任务终态及失败状态，INT/TERM 后终止并回收本次启动的任务，不留下仍在写产物的后台子进程；必需审计缺失不算完整成功。
- `--list` 与 `--dry-run` 不启动任务、不修改执行权限或实验产物；解析并展示真实计划、输入配置、阶段与目标日志。按步骤续跑要验证前置配置和产物身份，不自动补跑训练或选择现存旧结果。

本节仅补充设计与实施票。核心迟滞及 01–09 的已确认决定保持有效；第 10 步升级范围、总控默认执行 1–10 及 Q2 运行方式均已确认，规格与 [10: Shell 与总控升级](issues/10-upgrade-shell-entrypoints-and-pipeline.md) 恢复 ready-for-agent，实施依赖保持不变。本轮访谈无剩余待决项；未修改 shell/Python 或启动任何实验。

## Testing Decisions

- The main business test seam is the upgraded router's public get_action entry point. Use fixed dual-axis score sequences, a controlled account and changing fake specialist outputs; assert the selected expert, returned action and public audit artifacts. This follows the already discussed fixed-score and end-to-end validation plan.
- Exercise all four existing shell entry points through controlled integration scenarios and verify that each calls the upgraded implementation, reloads the same complete config and uses the intended stage and isolated output destination. Existing filenames are not evidence of legacy behavior or correctness.
- Exercise the controller's default 1–10 execution, explicit 1-9 validation-only selection, step 10, ranges/lists, log resolution and read-only preview with controlled command substitutes. Verify each child shell waits for completion, and delayed completion, failure, interruption or missing required evidence cannot be reported as success or start a dependent step. External background launch must preserve these semantics.
- Verify legacy/incomplete schemas, missing Δ/K, config hash or study/Trial mismatches and old strategy settings fail explicitly; do not maintain fallback parsing or require historical strategy execution regressions.
- Verify expert valid/test outputs, Optuna storage/study, best parameters, logs and audit artifacts use isolated destinations. Exercise a conflicting output/storage configuration and verify it fails before deleting or writing old artifacts. Verify shared frozen models and matched caches are read-only and unchanged; cache identity mismatch must not overwrite the old cache.
- Prefer the existing router, action-persistence, gating, non-main-defense, quantile-cache and final-result fixtures. Do not introduce a new private-helper test seam merely to check counters.
- A retained expert must emit updated actions when the fake policy's output changes across observations, even with a nonzero legacy persistence setting. Expert ID alone is insufficient proof of Q1.
- Cover one-axis replacement, two-axis replacement, both-axis entry qualification, per-changed-axis Δ, subthreshold current scores hidden by a high rival, exact threshold equality and strict nonreplacement on ties.
- Replay interrupted and changing candidates, K-th-step handover, ordinary invalidation at K-1, direct same-step takeover, and rule-only recovery. Assert that candidates and axes cannot pool confirmation evidence.
- Cover all tied-best cases: active tied-best, an existing higher-ID candidate tied by a newly best lower-ID Slot, and a tie without previous state. Assert externally stable selection.
- Test initial warm-up followed by K-step entry and state reset between independent contract rollouts.
- Cover empty-model Flat and held positions, finite graceful-hold-exit, no stale ordinary inference, hard-stop and circuit-breaker preemption, cooldown and non-main restrictions. Route memory must never broaden available risk.
- Invalid-input cases cover empty arrays, NaN/Inf, out-of-range values, changed axis dimensions or Slot identity mapping, and invalid configuration. Assert an explicit failed run rather than a plausible continued return.
- Use public tuning and final-evaluation preparation/replay entry points for the configuration integration seam. Assert that all nine effective dimensions are sampled and used, changing scoring windows/gamma changes the expected smoothing/warmup behavior under controlled inputs, and the selected values survive export, best replay and steps 8–10 reload unchanged. Model/reference identities stay fixed during search; scoring parameters stay fixed only within a trial and after selection.
- Compare cached and the same frozen scoring protocol through the router; vary only the source of identical quantile inputs, not stochastic model outputs.
- Adapt relevant existing fixtures to the upgraded contracts. Assert genuine inference rather than copying legacy ACTION_PERSISTENCE expectations; preserving absolute/hierarchical production paths is not required.
- Verify calibration export → step 8 publication/plots → step 9 synchronized diagnostics → step 10 test evaluation preserves config identity and all routing/risk settings. Check missing lean-trial histories do not alter the chosen candidate, valid/test outputs cannot overwrite one another, historical input schemas are rejected, and route plots align with actual decisions.
- Backtest comparisons report replacement and rule transitions separately, short-term switchbacks, confirmation waiting in steps and time, rule occupancy, real turnover, fees, net metrics and drawdown. Reduced switching alone is not a success criterion.
- Report the result as joint scoring-and-hysteresis optimization. If attributing an effect specifically to hysteresis, compare router variants under the same explicitly fixed scoring settings/input protocol; do not attribute the effect of different searched windows/gamma entirely to the state machine.
- Implementation acceptance requires the observable state/risk/replay contracts above. Numeric performance tolerances and final calibrated values require actual independent evidence; this spec does not invent thresholds or guarantee returns.

## Out of Scope

- Maintaining legacy strategy entry paths, old parameter-name/dirname inference, incomplete text parameter packages or old artifact schemas as compatibility branches in the upgraded steps 7–10.
- Copying the obsolete single-axis Python router from the old worktree into this project.
- Rewriting archived historical experiment evidence or silently reinterpreting it as a completed expert_hysteresis run. Comparative evidence may come from archived results or a separate historical code revision; the upgraded production pipeline need not execute the old strategies.
- Changing low-level model training, the action space, turnover Reward Shaping, DP expert tables, transaction fees or account accounting.
- Relaxing Selection Manifest evidence requirements, filling empty Slots with inferior models, or changing ADR-0061's No New Risk behavior.
- Weakening hard stops, circuit breakers, cooldowns, non-main restrictions or execution masks.
- Defining new joint class probabilities, multiplying raw scores into a profitability estimate, or mixing uncalibrated specialist Q values.
- New fast/slow degradation classifiers or specialist-performance retraining triggers; this feature is expert-selection hysteresis.
- Live order execution, deployment, large training jobs or running calibration as part of this documentation update.
- Claiming optimal numeric settings or improved future profits without validation.
- An unrelated OpenSpec migration or implementation work in place of the requested local spec/design update. The existing local implementation tickets are extended to cover all four pipeline steps.

## Further Notes

ADR-0062 explicitly scopes the updates to ADR-0031, ADR-0045 and ADR-0049. Their historic benchmark behavior and unrelated risk/reward decisions remain documented rather than erased. ADR-0046 caching, lean tuning and best-trial replay remain applicable; ADR-0047/0057/0060/0061 risk and contract semantics remain constraints.

The single-axis H/E/Δ/K discussion has been projected onto existing per-axis thresholds and one atomic Slot identity. The per-axis exit ratio already used by the target can express the new H/E relationship without adding an arbitrary joint score. This adaptation and its acceptance examples are recorded in the linked design.

The implementation starts with a public-router tracer scenario proving fresh decisions under retained expert identity, then completes state/risk behavior, configuration/calibration/replay integration, downstream publication/evaluation, and finally independent retrospective research.

当前实施依赖为 01 → 02 → 03 → 04；05 在 01 后提供统一配置；06 在 04/05 后交付步骤 7；08 在 06 后交付步骤 8；09 在 08 后交付步骤 9–10；10 在 09 后完成四个 shell 与总控联调验收；07 最后执行实际独立研究评估。实施票 10 的编号不等于流水线第 10 步。新增票不代表提前运行回测。

Formal parameter values, calibration outcome and performance are not yet available. The Q1–Q11 behavior and the latest in-place, non-backward-compatible upgrade of steps 7–10 are the current scope. Documentation status does not imply completed code, successful calibration or approval for live use.
