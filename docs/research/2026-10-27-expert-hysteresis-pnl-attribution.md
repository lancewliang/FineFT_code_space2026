# expert_hysteresis valid 盈亏归因与后续优化

日期：2026-10-27（本地复核于 2026-10-28）  
范围：配置哈希 `7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb`；`valid/main_sub`；燃料油 10min。  
目的：解释本次有效验证回放中盈利、亏损、风险来源，并提出最有价值的三个后续优化步骤。

## 结论先行

这次结果在**本次结果文件自己的口径**下表现为：初始本金 6,000 元，净赚 2,080.74 元，收益率 34.68%，步级最大回撤 4.47%（日末回撤 3.99%）；距用户希望的 60% 收益尚差 25.32 个百分点，而回撤距离 5% 上限只有 0.53 个百分点。它因此是“收益不足且风险余量很薄”，不能据此声称已达到 60%/5%。[summary_metrics.json](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/summary_metrics.json)、[portfolio_summary.md](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/portfolio_summary.md)

但需先解决一个**实质性口径/账户配置问题**：仓库 `CONTEXT.md` 与 [ADR-0060](../adr/0060-contract-boundaries-and-tonne-rmb-accounting.md) 将目标账户初始本金定义为共享 12,000 元；本次 frozen config 和结果使用 6,000 元，时间线也以 6,000 元开局。`audit_account_snapshots.json` 还把账户范围标成 `independent_contract_wallet`，因此主次组合时间线虽然同步汇总，但独立合约快照本身不能证明所有开仓确实受同一个共享钱包资金约束。诊断发布报告另将 12 个合约各自按 6,000 元相加到 72,000 元分母，故报告的 2.89% 是逐合约独立账户并集/平均展示口径，不是主次同步共享组合的收益率。+34.68% 又是结果记录的 6,000 元分母口径。账户配置未按 ADR-0060 对齐前，以上结果都不能作为目标 12,000 元共享账户的正式验收值；按相同净利润简单除以 12,000 只能得到 17.34%，不能替代共享账户重放，因为保证金、资金可用性、准入和并发仓位约束会改变实际交易路径。[CONTEXT.md](../../CONTEXT.md)、[frozen_config.json](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/frozen_config.json)、[diagnostics_summary.md](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/diagnostics/diagnostics_summary.md)

对组合时间线做的线性规模压力计算也显示，若保持原收益路径并把仓位同比放大到 60% 收益，最大回撤约为 6.65%，超过 5% 限制；该演算不含保证金/交易成本随仓位变化，不能当作策略回放。有效结论是当前需要改善净收益路径本身，单纯放大仓位不满足目标。[可复核计算证据](#当前冻结搜索的目标可行性信号)

亏损不是主要由手续费单独造成，而是**少数合约方向/时机选择失利与有限毛利被交易摩擦削弱的叠加**。毛利 2,292 元、净利 2,080.74 元，差额 211.26 元，其中已记录手续费 142.26 元，逐笔审计记录的滑点 69 元；总摩擦占毛利 9.22%。其中 fu2505 是唯一净亏合约；fu2507 的毛利几乎被成本抹平。下一步应先改善有主力准入资格时的入场质量与持仓管理。非主力合约的零交易符合当前限制，不能将其区间涨幅当作可回收收益。

## 证据与口径

### 本次结果身份

- 结果收据记为 `eval_stage=valid`、`evaluation_purpose=main_sub`、Trial 508、配置哈希与目标目录哈希一致；`summary_metrics.json` 同样标记 `valid`。指定目录只有 `valid/main_sub`，没有 `test` 或 `main` 子目录。不能把此次 valid 回放写成 test 或最终样本外结论。[frozen_evaluation_receipt.json](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/frozen_evaluation_receipt.json)、[summary_metrics.json](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/summary_metrics.json)
- 所有定量指标来自冻结结果、时间线及诊断产物；相关代码只用于解释字段定义与约束。未重新训练/回放，也没有使用未来信息构造新表现数据。
- frozen config 的 `initial_wallet_balance=6000`、`leverage_choices=[1]`、`max_holding_number=1`、`strategy_position_unit=tonne`、交易成本参数 `transcation_cost=0.0003`、动作持久 6 步、止损阈值 3%、MDD 目标 5%、惩罚权重 4、最小交易数 guard 60。[frozen_config.json](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/frozen_config.json)
- `CONTEXT.md` 定义策略仓位数量为吨、人民币收益为有符号吨数乘元/吨价格变化，并定义共享组合账户和 12,000 元目标本金。应避免把保证金杠杆、资金敞口和收益倍数混为一谈。[CONTEXT.md](../../CONTEXT.md)

### 主次同步组合（结果记录口径）

| 指标 | 数值 | 来源字段 |
|---|---:|---|
| 起始本金 | ¥6,000 | `portfolio.initial_capital` |
| 毛盈亏 / 净盈亏 | ¥2,292 / ¥2,080.74 | `total_gross_pnl` / `total_net_pnl` |
| 累计收益率 | 34.679% | `tr` |
| 步级 MDD / 日末 MDD | 4.466% / 3.987% | `mdd` / `daily_mdd` |
| 交易日 / 时间步 | 225 / 8,683 | `trading_days_count` / `total_timeline_steps` |
| 开平成交事件数 / 换手额 | 152 / ¥474,199 | `total_trades` / `total_turnover_volume` |
| 手续费 | ¥142.26 | `total_commission` |

`portfolio_action_timeline.csv` 的最大回撤峰值是 ¥7,700.71（2025-02-19 22:20），至 2025-04-02 23:00 谷值 ¥7,356.80，回落 ¥343.92，峰谷回撤 4.466%；期末权益 ¥8,080.74。全周期绝对最高权益 ¥8,111.55 出现在回撤谷值之后，不能把它当作这次最大回撤的峰值。峰谷区间的组合持仓只含 fu2505，共 964 个 bar，单一方向或空仓，没有双持仓；该区间净值下降来自 fu2505 的实际持仓路径，不是主副合约同时亏损。[portfolio_action_timeline.csv](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/portfolio_action_timeline.csv)

### 盈利与亏损集中度

| 合约 | 毛利 | 净利 | 费用 | 多/空/空仓 bar 占比 | 交易数 | 步级 MDD |
|---|---:|---:|---:|---:|---:|---:|
| fu2411 | ¥875 | ¥826.37 | ¥31.63 | 27.92% / 31.91% / 40.17% | 36 | 2.81% |
| fu2509 | ¥561 | ¥546.55 | ¥10.45 | 3.03% / 8.08% / 88.89% | 11 | 4.22% |
| fu2503 | ¥378 | ¥338.59 | ¥27.41 | 22.05% / 16.23% / 61.72% | 27 | 3.81% |
| fu2501 | ¥332 | ¥299.77 | ¥20.24 | 14.41% / 20.20% / 65.40% | 22 | 2.18% |
| fu2505 | -¥15 | -¥55.72 | ¥30.72 | 13.18% / 11.31% / 75.50% | 32 | 5.63% |
| fu2507 | ¥30 | ¥3.10 | ¥15.90 | 23.37% / 11.07% / 65.56% | 18 | 4.81% |

表中来源：`diagnostics/contract_pnl_friction.csv` 的 `gross_pnl`、`net_pnl`、`est_commission`、`est_slippage`、`trades`、`mdd_pct`；`diagnostics/contract_behavior_risk.csv` 的方向/空仓百分比。所有合约合计毛利 ¥2,292，净利 ¥2,080.74。四个正净利主贡献（fu2411/2509/2503/2501）净利约 ¥2,011.27；fu2505 回撤最大且净亏 ¥55.72。fu2507 已经毛赚 ¥30，却因 ¥26.90 摩擦仅剩 ¥3.10，说明低毛利、频繁切换的合约容易变成“做对但没赚到”。

| 合约 | 本段 buy-and-hold 变动 | 策略持仓事实 | 解释限制 |
|---|---:|---|---|
| fu2412 | +17.42% | 0 交易、100% flat，net 0 | 结果显示全段非主力/被防御在场，不能把它当成可合法开仓却错过的机会 |
| fu2503 | +36.68% | 策略净利 ¥338.59，MDD 3.81% | 区间风险/资金口径不同，不把涨幅差额当作可回收利润 |
| fu2505 | +2.17% | 策略净亏 ¥55.72，MDD 5.63% | 持仓捕获与进入/退出 timing 可能不佳，且费用进一步扩大净亏 |
| fu2507 | +3.98% | 策略净利 ¥3.10，MDD 4.81% | 毛利微弱，¥26.90 摩擦吞噬约 89.7% 毛利 |
| fu2509 | -8.75% | 策略净利 ¥546.55，88.89% flat | 策略抓住少数有效方向/时段；不是持续暴露带来的收益 |

市场期间变动与策略净值不具同风险、同仓位可比性。并且按既有主力准入设计，非主力合约（角色低于准入等级）禁止开新仓，因此 fu2412/2508 等零交易合约不是简单的盈利机会漏失；buy-and-hold 只描述行情端点变化，不能解释策略 Alpha、也不能作为可交易反事实。[contract_pnl_friction.csv](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/diagnostics/contract_pnl_friction.csv)、[ADR-0057](../adr/0057-graceful-downgrade-and-synchronized-main-sub-portfolio.md)

### 交易方向、退出与费用

诊断轨迹表明有 3,336 个正净 PnL 时间步、3,286 个负值时间步、2,061 个零步；对收益贡献关键的是盈亏幅度和持仓方向/持续时间，而非正负步数本身。方向占比是**仓位 bar 占比**，不能当作开平仓方向或胜率。可见合约中 fu2409 仅做多，fu2411 多空时间相近，fu2509 仅 11.11% 的时间处于多/空仓位，却贡献 ¥546.55 净利。[portfolio_action_timeline.csv](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/portfolio_action_timeline.csv)、[contract_behavior_risk.csv](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/diagnostics/contract_behavior_risk.csv)

决策原因码历史仍是步骤级而非逐笔标签；更强的一手归因来自各合约 `contracts/<contract>/audit_trades.json`、`audit_account_snapshots.json` 与 `audit_routing.json`。按成交数量大于 0 将每次 filled `open` 配对到下一个 filled `close`，把开/平手续费计入回合净值，并对期末仍持仓的两笔计入 `unrealized_pnl_history.npy` 末值，重建出 77 个持仓回合（75 已平、2 期末未平），总净额精确对上 ¥2,080.74。已平回合胜率 56.0%（42/75），盈利因子 2.21。按回合平仓原因统计：普通 `POLICY_INFERENCE` 61 回合合计 +¥2,958.42；11 个 `HARD_STOP_LOSS` 回合合计 -¥1,188.90；2 个 `DEFENSIVE_PREEMPTION` +¥297.09；1 个 `DEFENSIVE_RULE_CLOSE` -¥80.81；2 个期末未平回合的已实现加未实现净额 +¥94.94。所有负回合累计 -¥1,649.45，由正回合抵消后才形成正总收益。硬止损回合净亏均值约 ¥108，次数少但损失集中，优先复盘其触发前方向与止损后反转/延续路径。

按方向归集 77 个回合（含期末浮动）净额：多头 38 个，+¥1,083.995；空头 39 个，+¥996.745。两边都有贡献，不能仅凭总计决定禁做多或禁做空；合约分层更有用，例如 fu2505 多头回合净额 -¥189.28、空头 +¥133.56（各 8 回合），与其最大回撤期方向反复相符。该结果是持仓回合的描述性归因，不能证明把 fu2505 改成只做空会改善其他月份或未来表现。

再将每一回合的 `reward_history` 区间和回合净值交叉核对，两者逐回合一致；在 34 个负回合中，18 个回合在持仓期间曾出现至少 ¥20 的浮盈后转为亏损或较低收益。示例包括 fu2505 2025-02-18 多头回合曾有约 +¥47.97 的浮盈，最终 -¥111.03；fu2409 2024-07-30 多头回合曾有约 +¥90.01 浮盈，最终 -¥107.94。这个事后 MFE（最大有利变动）只说明退出路径值得检查，无法视作可实现利润或证明追踪止盈能保留这些浮盈；必须用训练/校准期制定退出规则，再在未参与选择的时间段核验。[回合路径核对产物](fu_10min_expert_hysteresis_20261027/fu_hysteresis_episode_paths.csv)

最大回撤峰谷区间内，fu2505 有 12 个完成回合净额合计 -¥260.96；其中两笔硬止损回合净额 -¥218.99，另外 10 笔普通策略退出净额合计 -¥41.97。这个 closed-episode 合计没有覆盖峰谷端点的未实现损益，因此小于组合净值回撤 ¥343.92；差额由回合边界之外的浮动损益变化构成，不能把 12 个平仓回合合计直接等同为 MDD。此回撤阶段的唯一持仓标的是 fu2505，故风险优化优先级高于讨论 fu2412 行情错失。

成交审计直接记录手续费合计 ¥142.26、滑点字段合计 ¥69；因此 69 元已有逐成交 `slippage` 字段来源，不应笼统称为完全未知残差。逐成交对照字段包括 `reference_price`、`fill_price`、`commission`、`realized_pnl`、`slippage`、`decision_reason`、`decision_id`、`leg_index` 与前后 `snapshot_id`。全组合毛净差额 ¥211.26 与手续费加滑点汇总吻合。解释仍须谨慎：交易回合净值按成交实际 realized PnL 减开平手续费，`slippage` 用于成交价相对参考价的执行归因；不得又从基于实际 fill 的回合收益重复扣一次滑点。

总摩擦 ¥211.26 = 审计手续费 ¥142.26 + 逐笔 `slippage` 字段合计 ¥69。`contract_pnl_friction.csv` 的 `est_slippage` 碰巧通过毛净残差计算得到同一汇总；逐笔审计提供了直接的滑点字段，但它是否完整刻画真实市场冲击成本，仍取决于成交模拟对参考价和 fill price 的定义。配置交易成本率 0.0003 不代表每笔实际成本是常数费率。[audit_trades.json](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/contracts/fu2505/audit_trades.json)、[trading_diagnostics.py](../../FineFT/analysis/diagnostics/trading_diagnostics.py)、[futures_util.py](../../FineFT/env/env_class/futures_util.py)

### 最优 Trial 的选择目标

`frozen_evaluation_receipt.json` 显示 Trial 508 的校准目标值为 0.350493，而此次组合原始收益指标为 0.346790；二者不同。代码 `vae_routing_util.py` 将正收益 objective 按 MDD factor 与交易数 guard 调整，阈值 MDD 5%、最少交易 60、MDD 权重 4。此回放 152 次非零开/平成交事件（75 个已平回合加 2 个未平回合），guard 饱和为 1，MDD 为 4.466% 时低于 5% 门槛，factor 为 `1 + 2*(0.05-0.04466) ≈ 1.01068`，故 objective 大于 raw return。也就是说这次寻优目标不是直接最大化“60% 收益且回撤不超过 5%”的硬约束目标；仅设置 5% target/penalty 不会确保每个 Trial 的回撤低于 5%，也不会使收益达到 60%。[vae_routing_util.py](../../FineFT/RL/DiHFT/high_level/vae_routing_util.py)、[frozen_config.json](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/frozen_config.json)、[frozen_evaluation_receipt.json](../../result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/frozen_evaluation_receipt.json)

### 当前冻结搜索的目标可行性信号

本地 Optuna 搜索证据（800 个 complete Trial 的原始 metrics）显示：17 个 Trial 在当前 6,000 元路径上 MDD ≤ 5%，没有 Trial 达到 60% raw return，也没有同时满足两者。最好的回撤合格结果就是选中的 Trial 508：raw return 34.679%、MDD 4.466%；其 Trial 目标分因回撤奖励调整到 35.049%。把所需收益缺口以 6,000 元固定本金换算为增量净利约 ¥1,519.26。这个搜索只说明当前参数空间、当前模型/valid 时间段和账户口径下的结果分布，不证明策略的理论上限，也不能用同一 valid 扩大搜索后声称样本外达标。[本地搜索核对摘要](../../result/DiHFT/high_level/fu/10min_parallel/expert_hysteresis/fu-10min-expert-hysteresis-v1/valid/calibration/20261026)

最大回撤窗口 2025-02-19 22:20 至 2025-04-02 23:00 的净值回落 ¥343.92；这段窗口的持仓记录只包含 fu2505（964 个时间步，其中 551 个多头、413 个空头；另有 210 个组合空仓时间步；未出现双持仓），因此最新回撤应优先复盘 fu2505 的逐笔进出/止损/持有方向。虽然两笔硬止损回合对该区间已平回合净亏贡献很大，但 MDD 还包含未实现 PnL，不能将整段回撤等同于止损金额。此点也修正容易从旧双主力设计假设产生的误判，不能将回撤归咎于两合约同步加仓。

## 最值得做的三个优化步骤

这三步是研究/验证顺序建议，不是收益承诺。每步都须用冻结配置、共同评估区间和完全一致的资金/费用假设比较；先修复口径，再动策略参数。

### 1. 先统一 12,000 元共享账户，再建立可重复的有效对照

把本次 frozen config / 汇总定义与 ADR-0060 的 12,000 元共享组合账户、每份结果只投入一次本金、吨与人民币记账规则对齐；检查主次时间线中的保证金、可用资金、并发开仓、手续费预留是否真的由同一账户约束。然后在同一 valid 区间重跑当前冻结版本，保存 main_sub 和各合约结果，确认净收益、回撤、开仓数和权益轨迹可复算。随后才用固定 config 在保留的 test 上做一次最终确认，test 不能反馈参数选择。该步价值最高，因为本金分母/账户约束现在和项目决策定义直接冲突；如果路径因资金约束变化，当前 34.68%/4.47% 并非目标组合的基线。ADR-0063 也明确 test 消费冻结配置，不反馈选参。[ADR-0060](../adr/0060-contract-boundaries-and-tonne-rmb-accounting.md)、[ADR-0063](../adr/0063-in-place-expert-routing-pipeline-upgrade.md)

### 2. 建立成本后交易准入，减少错误方向与低净优势交易

当前 valid 证据支持三个可检验方向：fu2505 造成全部组合 MDD 区间回落，并有两笔约 -¥219 的硬止损回合；fu2507 毛利不足以覆盖交易摩擦；趋势机会分析只对主力准入时段与合规仓位有效，不以非主力合约的 Buy-and-Hold 区间收益定义“漏失”。先从审计成交回合和路由 `decision_id` 出发，将止损前后价格路径、active/queried Slot、进入时仓位资格与后续走势对齐，再在校准窗口做少量预注册风控/准入消融。

注意本次配置 `action_persistence=6`，当前实现会在剩余持久步内沿用 `current_action` 跳过 agent 推理（见 `vae_routing_util.py` 2049–2058），而 ADR-0062 记载普通专家每步重新决策；此前“调整7”记录又明确要求动作保持。未厘清该已确认语义前，不能将专家 Slot 切换次数当成真实 expert 推理次数或把持仓缓存当成 bug；任何改变持久行为的实验要作为用户已确认策略语义的显式重开/消融。对“评估期累计净收益率 60%”设为目标、步级 MDD 5%设为最终硬筛选条件（或显式报告不可行）；不能在同一 valid 上反复选择后再把它称作样本外结果。

### 3. 改善持仓回吐与专家交接，保留趋势赢家

结果已保存逐笔成交价、数量、时间、开平腿、参考价、手续费、滑点和拒单/风控原因。应先将这些 `audit_trades`、账户快照与 `decision_id` 对齐，核实模拟滑点字段是否覆盖预期执行摩擦，并对实际费用做保守上浮情景；再根据负回合路径检验退出/冷却规则，避免在 MFE 事后信息上过拟合。valid 汇总里摩擦占毛利约 9.2%，它会吞噬边际低的交易，但不是唯一问题。仓位加大/提升暴露只能在 12,000 共享账户和保证金约束下做预注册阶梯实验，并按 MDD 预算自动拒绝超过 5% 的方案；按当前基线路径同比放大到 60% 已对应约 6.65% MDD，不能把加杠杆视为达标捷径。

建议每个候选报告至少给出：净收益、步级/日末回撤、日度序列风险指标、毛净差额、成交成本分解、最大资金敞口、活跃持仓数、逐合约收益和跨窗口分布。现有数据可支持有效段描述性归因；它不能证明三步会把收益推到 60%。

## 限制与后验归因

1. 本次目录只有 `valid/main_sub`；没有 test 结果。Valid 已参与参数选择/发布流程，适合诊断，不是独立样本外验证。
2. 6,000 元结果本金与项目 12,000 元决策口径冲突；`analysis_result/.../selected_trial_publication/summary.json` 的 `initial_equity=72000` / TR 2.8899% 只是 12 份每合约 6,000 元独立评估汇总。不能把这两种数字拼接或用一条简单比例换算代替共享账户重放。
3. 回合重建依赖 filled `open`/`close` 事件严格配对，并将期末未平持仓按最后未实现 PnL 计入；转换回合的交易费用按审计记录加总。它按决策原因归集已实现回合结果，不能把原因标签解释为策略原因的因果效果。
4. MDD 是连续盯市权益峰谷变化；回合平仓净额只覆盖 realized trades 和期末持仓，二者时间边界不同，因此峰谷区间平仓净额不等于组合回撤金额。
5. `contract_behavior_risk.csv` 的多/空/flat 是持仓步占比；Buy-and-hold 是端点变化，不具备相同的主力准入、择时、资金和风控限制，不能当作可实现策略收益。对未满足主力准入的合约，不将行情收益称作策略漏失。
6. 逐笔成交/账户快照/路由均有 audit 文件，可用 `decision_id`、`trade_id`、snapshot 和时间关联。Slot 归因须进一步考虑每步持久动作缓存和持仓准入，单看宏观 Slot 次数仍不能证明因果收益。

## 主要一手来源

- 结果：`result/DiHFT/final_result/fu/10min_parallel/expert_hysteresis/7620ac5ca9979f71bd0152761fc593693546dd8d02b61b6829a8d7c474db02eb/valid/main_sub/` 下的 `summary_metrics.json`、`portfolio_summary.md`、`portfolio_action_timeline.csv`、`frozen_config.json`、`frozen_evaluation_receipt.json`、`diagnostics/diagnostics_summary.{md,json}`、`diagnostics/contract_pnl_friction.csv`、`diagnostics/contract_behavior_risk.csv`、`diagnostics/macro_routing_distribution.csv`、`diagnostics/<contract>_step_pnl_audit.csv` 与 `contracts/<contract>/audit_trades.json`、`audit_account_snapshots.json`、`audit_routing.json`。
- 对应 high-level 发布摘要：`analysis_result/DiHFT/high_level_heurstic/fu/10min_parallel/selected_trial_publication/summary.json`（`config_hash` 与结果一致；重点区分其 `aggregate.initial_equity=72000` 独立合约汇总与组合结果）。
- 项目口径：`CONTEXT.md`；`docs/adr/0060-contract-boundaries-and-tonne-rmb-accounting.md`、`docs/adr/0062-stateful-expert-routing-hysteresis-and-frozen-score-calibration.md`、`docs/adr/0063-in-place-expert-routing-pipeline-upgrade.md`。
- 字段和算法定义：`FineFT/analysis/diagnostics/trading_diagnostics.py`、`FineFT/RL/DiHFT/high_level/vae_routing_util.py`、`FineFT/common/artifacts.py`、`FineFT/env/env_class/futures_util.py`。

Python 复核按项目指令在 `finetf` conda 环境运行；本次只做读取和计算，没有修改任何已有策略或结果文件。


## 可执行的下一轮研究设计

以下均为待验证方案，本轮没有改动交易代码、没有训练模型、没有声称实验改善了收益。默认将“盈利60%”解释为同期间累计成本后组合收益率；如用户指年化收益或逐笔胜率，应独立重定义验收，不能混用。

| 顺序 | 具体工作与 architecture 对应 | 对照与验收 | 证据及价值 |
|---|---|---|---|
| 1 | 深化共享组合账户 module，统一资金、保证金、费用、成交与权益；优化目标与最终报告使用同一真实路径，步级 MDD>5% 的候选不得认证达标 | Trial508 保留历史；账户变更使用新实验身份。全局同一时间先盯市、减仓、再加风险，验证资金守恒和拒单；报告可行候选数及未达收益目标的事实 | 当前独立钱包与 ADR-0060 冲突；校准目标只是软回撤折扣；先确保度量可信 |
| 2 | 深化扣费交易准入与归因 module，把 active/queried 专家、拟议/最终动作、成交、持仓回合按 decision_id 对齐；研究成本后预期收益及尾部损失能否被历史特征排序 | 先检验排序稳定性，再以训练/校准历史确定准入；记录少亏金额与误拒盈利金额。相同执行模型做配对回放，禁止按 fu2505 或未来损益直接封禁 | 11 次止损净亏 1188.90 元，约占全部负回合损失 72.08%；fu2505 多头−189.28，空头+133.56；fu2507 成本吞噬89.67%毛利 |
| 3 | 深化持仓回合 module，分开检验动作保持、专家交接和退出退化。先做“保持6步”与“每步推理”单项对照，再做使用过去波动和已观察浮盈的退出实验；保留硬止损优先级 | 固定入场与其他风控，一次改一个机制，确认大赢家没有被过早截断；再组合通过的机制。观察净收益、MDD、费用、负尾与最大赢家保留情况 | 34负回合中18曾浮盈≥20元；既存在回吐也存在+334元赢家，退出不能只追求少亏；动作缓存与ADR-0062描述不同须记录现行语义 |

三个 module 都应通过现有公开回放 interface 测试，seam 只设置在实际变化处；真实基线策略和候选规则是有意义的两个 adapter，不为测试额外引入空转 adapter。目标是增加 depth：把当前散落的时序和资金 implementation 集中，从而获得 locality 与跨寻优/评估调用的 leverage。删除测试（deletion test）应检查删掉新 module 是否会迫使调用方重建这些规则，而非只增加转发。

验证采用完整时间顺序：各外层验证折中，低层选择、高层校准和准入/退出参数都只能使用该折之前的数据，并对标签/持仓跨期实施 purge/embargo。**已有 low-level agent 在全 valid 上选择过，不能把当前 valid 再切几段就称为全流程未见验证**；正式外推比较需在每折重做选择，必要时重训。当前 valid 的消融只能作为探索。候选全部冻结后才做一次保留 test 的完整评估，test 不反馈选参；若 test 已被其他研究查看过，应另找真正未见时间段。对保守费用和执行延迟场景分别报告，不用加权分数掩盖超限。

风险预算应按“每笔可能损失占账户权益”管理，区别于当前按标的价格不利变化3%触发的止损；这与 CME 关于仓位取决于止损距离和账户风险额度的原则一致，具体预算须由本项目校准，不照搬其示例比例。[CME：Proper Position Size](https://www.cmegroup.com/education/courses/trade-and-risk-management/proper-position-size)

在同一历史段上多次选参会放大回测过拟合风险；论文给出了这一问题和评估框架，不能因为 800 次里找到好结果就当作新的独立证据。本报告不声称已经计算本项目 PBO。[Bailey 等：The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)

## 目标算术敏感性与复核产物

| 固定原逐步净损益路径的情景 | 累计净收益率 | 步级 MDD | 达60%所差净利润 |
|---|---:|---:|---:|
| 原始6,000元本金 | 34.679% | 4.466% | 1,519.26元 |
| 12,000元本金，仅改权益起点 | 17.3395% | 2.5102% | 5,119.26元 |
| 原路径同比放大至60% | 60% | 6.6540% | 回撤超限 |

原路径放大系数分别是6,000元本金时1.73015倍、12,000元本金时3.46031倍。放大计算假设逐步净PnL线性扩展，**不是实际加仓回放**，没有证明成交深度、资金和保证金允许。12,000元基准行也只是算术反事实，不替代共享账户执行。当前800次搜索的最高原始收益47.3913%，MDD6.3618%；17/800（2.125%）满足5%回撤，不代表参数空间已被穷尽，更不证明60%理论上不可能。

复核数据及脚本在 [fu_10min_expert_hysteresis_20261027](fu_10min_expert_hysteresis_20261027/)：

- [组合权益与800 Trial复核](fu_10min_expert_hysteresis_20261027/fu_hysteresis_target_evidence.json)
- [逐笔回合归因汇总](fu_10min_expert_hysteresis_20261027/fu_hysteresis_episode_evidence.json)
- [77个回合明细](fu_10min_expert_hysteresis_20261027/fu_hysteresis_episodes.csv)
- [逐回合净值路径/MFE核对](fu_10min_expert_hysteresis_20261027/fu_hysteresis_episode_paths.csv)

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh
conda activate finetf
python docs/research/fu_10min_expert_hysteresis_20261027/target_audit.py
python docs/research/fu_10min_expert_hysteresis_20261027/episode_attribution.py
python docs/research/fu_10min_expert_hysteresis_20261027/episode_path_check.py
```

已验证：初始6000加逐步净PnL与保存权益逐点一致；77回合净额与2080.74元一致；每个回合的reward区间累计与成交/费用/期末浮盈重建值一致；审计手续费142.26元和滑点69元与毛净差额211.26元一致。
