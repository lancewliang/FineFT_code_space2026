# DiHFT 高频交易策略全维度诊断与归因分析方法论

- **方法论版本**：v1.2（2026-10-12）；诊断计算版本 v2.0。
- **变更**：在 v1.1 研究阈值不变的前提下，加入审计资格门禁、Prediction / Trade / Account Record 三类逐事件证据链，以及模型决策、执行、账户分层归因流程；同步扩展代码批量审查工具的 audit profile。没有为当前成绩降低通过标准。

## 1. 概述与核心目标

本方法论旨在为 FineFT 框架下的 DiHFT（分层深度强化学习高频交易系统，涵盖 VAE 宏观体制路由、低层微观 Q-Net、三级风控及执行层）提供一套系统化、严密且高效的量化策略诊断体系。

针对实验产物目录（例如 `analysis_result/DiHFT/high_level_heurstic/fu/10min_parallel/` 及对应的 `diagnostics/` 和 `result/DiHFT/final_result/`），本方法论解决以下核心问题：
1. **策略好坏评判**：如何在考虑多合约横截面、基准对比（Buy & Hold 1x/5x）、风险调整后收益与尾部回撤的前提下客观评判策略优劣？
2. **盈利成因解耦**：策略的净利润究竟来源于真实的盘面择时预测能力（Alpha），还是运气？
3. **失效根因诊断**：若策略亏损或不及预期，根本原因究竟是择时反向、摩擦损耗吞噬、持仓周期过短、宏观路由崩塌还是极端行情踩踏？
4. **针对性调优路线**：在确定根因后，应该调整高层路由门控、动作持续性、三级风控还是底层选模与训练奖励？
5. **高效分析实施与 Token 优化**：规划精准的文件读取清单与依赖链路，并依托一次性批处理工具（`batch_diagnostics_loader.py`）避免多文件串行逐个读取带来的上下文膨胀与分析延迟。

---

## 1.1 数据用途、冻结与独立测试

训练、选模、超参数寻优、特征筛选、标准化及 VAE 拟合均属于模型开发。用于其中任一环节的数据，不能再作为独立泛化评价依据。400 轮寻优不自动证明严重过拟合，但最佳验证成绩也不能独立证明泛化能力；仅称“当前验证集与回测假设下表现较好的候选配置，等待冻结后的独立测试”。这一限制与 [scikit-learn 官方模型选择示例](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)一致；金融时间序列应采用时间顺序切分、标签边界净化，不能直接照搬随机交叉验证。

独立测试前固定：代码提交、数据文件及哈希、特征清单、所有 checkpoint/变换、参数、手续费/成交/资金规则、评价版本。保存逐合约清单，字段至少包括：合约 ID、文件路径及哈希、数据起止时间、训练/选模/寻优/测试用途、前向标签成熟时间、每个模型/特征选择/变换的拟合截止时间。未知字段写“待核对”，不能据文件夹名称认定盲测。检查合约及时间交集、标签跨界、拟合截止晚于测试起点等情况；同一合约后续时段称“时间外测试”，完全未参与开发的合约才称“未见合约测试”。测试结果一旦用于调整方案，该测试数据转为开发数据，下一次独立验收另留数据。

## 1.2 审计级逐事件证据链

新回测应按 TradingDay 发布三类相互关联的逐事件记录，并以 `audit_manifest.json` 声明 schema、运行身份、模型与特征版本、账户范围、TradingDay 来源、分区行数和校验结果：

- **Prediction Record**：只在真实模型推理发生时记录 Action Value、最优与次优 Action Value margin、路由证据、模型建议动作和版本信息。它是策略推理记录，不是“今天收盘/下一 TradingDay 收盘”的概率预测；Q 值及其差值不得解释为盈利概率或校准置信度。
- **Trade Record**：按执行 leg 或执行尝试记录请求动作、实际仓位、数量、成交状态、参考价、成交价、费用、已实现盈亏、滑点证据、决策原因及交易前后账户快照关联。反手拆为平旧仓和开新仓两个 leg；拒绝与部分成交不得丢弃。
- **Account Record**：保存初始账户快照及每个环境 step 后的独立合约账户快照，覆盖钱包、浮盈亏、权益、保证金、可用资金、仓位、敞口、杠杆、峰值权益及回撤。审计历史在平仓后不得重置。

三类记录以稳定 ID 连接：Trade Record 的 `source_prediction_id` 可以引用真实 Prediction Record；warm-up、硬止损、冷静期、熔断等规则动作允许该字段为空，但必须记录决策原因。Trade Record 还应连接执行前后的 Account Record。账户范围固定声明为 `independent_contract_wallet` 时，不得把多合约独立账户求和称为共享资金组合。

审计级资格要求主键唯一、非空外键有效、TradingDay 明确、账户恒等式成立、权益变化与冻结奖励口径可对账，且全部分区经过原子发布。任一条件失败，总体状态优先 `NEEDS_AUDIT`，并停止模型预测力、执行优势或盈利成因的因果归因。旧实验缺少这些原始事实时可继续做描述性复算，但不能被追认成审计级结果。

## 2. 策略好坏的量化评价标准 (Evaluation Standards)

以下是本项目的**研究描述性阈值**，尚未校准为生产验收标准；收益水平受回测跨度、风险预算和资金口径影响。阈值须在看独立测试成绩前冻结，修改时单列版本和理由，并按同一新口径重算所有对比版本。实现以 `FineFT/analysis/diagnostics/evaluation_standards.py` 为统一来源，批处理不得另设 HEALTHY 条件。

| 指标 | Strong | Warning | Fail | 字段 |
| :--- | :--- | :--- | :--- | :--- |
| 独立账户汇总净回报率 | >20% | (0%,20%] | ≤0% | `portfolio_return_pct` |
| 独立账户参考价毛回报率 | >40% | [10%,40%] | <10% | `portfolio_gross_return_pct` |
| 毛净差额 / 正毛利 | [0%,50%) | [50%,75%] | >75% | `friction_consumption_ratio_pct` |
| 本金换手倍数 | <30x | [30x,80x] | >80x | `portfolio_turnover_multiple` |
| 逐合约平均分组年化夏普 | >2.0 | [1.0,2.0] | <1.0 | `mean_annual_sr` |
| 逐合约平均最大回撤 | <8% | [8%,15%] | >15% | `mean_mdd_pct` |
| 逐合约平均分组卡尔玛 | >2.5 | [1.0,2.5] | <1.0 | `mean_daily_cr` |
| 盈利合约比例 | ≥85% | [70%,85%) | <70% | `win_rate_pct` |
| 最差合约期末亏损比例 | <5% | [5%,12%] | >12% | `worst_contract_loss_pct` |

毛利非正时吞噬率为 `null/Unavailable`，不能用 0% 表示无成本；负毛净差额或负费用残差标 `NeedsAudit`，不能凭低吞噬率验收摩擦控制。缺少数据或成交审计时，总体状态优先 `NEEDS_AUDIT`，逐指标等级仍保留；否则任何 Fail → `RESEARCH_FAIL`，任何 Warning → `RESEARCH_WARNING`，全部 Strong → `RESEARCH_STRONG`。这些状态均不代表生产就绪。

`portfolio_*` 为兼容保留的独立合约账户汇总字段，回报率为总净值变化 / 各合约初始资金之和，不是共享资金账户收益。`mean_*` 是逐合约算术平均，不是组合最大回撤或组合夏普；`portfolio_mdd_pct`、`portfolio_annual_sr` 在无共同时间轴与共享资金净值时置空。另报最差单合约路径回撤 `worst_contract_mdd_pct`；期末亏损率使用该合约自身初始资金作分母，不能除以全体资金稀释风险。

当前 `annual_sr/daily_cr/daily_sor` 沿用 `calculate_metric` 的固定 bars 分组（默认12）与365年化，属于旧计算约定，并非按交易日聚合的日频指标，也不能解释为明日收益预测。生产评价需另行冻结 TradingDay 聚合、交易年天数及风险预算后重算；本版不追溯更改选优目标。

---

## 3. 盈利能力的成因与解耦框架 (Profitability Decomposition)

ADR 0048 要求逐步成本归因，但早期实现实际只有请求动作、奖励及资产历史，不能据此测量真实手续费和盘口滑点。本版修正其“精准解耦真实费用”的解释；历史 ADR 的研究数字不追溯改写，复算以新诊断版本为准。

### 3.1 参考价毛盈亏与实际净值变化

环境 N 个 step 对应 N 个动作/净值增量和 N+1 个行情/初始及后续净值观察。第 i 步在参考价 p[i] 执行后持有 q[i] 至 p[i+1]：

$$G=\sum_{i=0}^{N-1}q_i(p_{i+1}-p_i)u,\qquad N=(E_N-E_0)u.$$

优先读取 `market_index_history.npy` 对齐每步起止行情；即时强平未推进行情时使用重复索引，不能虚构后续价格区间。旧产物缺少索引时标为顺序步假设，不能自动认证时间对齐。优先读取 `executed_position_history.npy`；旧产物只含 `micro_action_history.npy` 时按 0/1/2 → -1/0/+1 重建**请求仓位代理**，仅支持初始空仓、单位三动作环境。涨跌停或成交约束可能使请求与实际仓位不同，必须标记此限制。净盈亏含期末浮动盈亏，不自动称已实现利润。合约单位 u 同时缩放毛利、净利与初始资金，避免把收益率放大 u 倍。

参考价与盘口应使用同一 bar 末端事件时点。2026-10-12 审查发现旧下采样参考价取 `.first()`、盘口取 `.last()`；本次修复统一到末端。旧 Feather 和模型并未自动重建，旧成绩不能证明修复后数据上的表现。

### 3.2 毛净差额与费用核对

$$R=G-N,\qquad U=R-C.$$

R 是毛净差额；C 优先读 `commission_fee_history.npy`，缺失时按初始空仓及逐步仓位变化的参考价名义额估算手续费；U 是**未解释执行残差**。兼容字段 `friction`/`total_friction` 表示 R，`est_slippage`/`total_slippage` 表示 U，不能直接称真实订单簿滑点。实际成交、估算手续费、参考价时点差异、仓位代理误差、资金费用或记账差异均可能影响 U。只有逐笔成交价、数量、方向、费用及同一时点参考价对账后，才能测量执行滑点；已计入净值的费用不能再扣一次。

每个合约输出 `<contract>_step_pnl_audit.csv`，检查首尾价格区间、首次开仓、净值增量、仓位来源、手续费来源及负残差步骤。负残差不能被截零，也不能自动归为执行优势。毛利非正时比率不可定义。

Buy & Hold 的 1x/5x 涨跌率只是价格基准，未自动匹配策略资金、杠杆、持仓规模、保证金及强平规则。相对基准少亏或毛利更高是描述性差异；真实 Alpha 需要同资金/风险/时间口径对照、冻结后的时间外结果及不确定性评价。

### 3.3 微观持仓姿态与执行机制 (Micro Posture & Execution)
- **多空平比例 (`long_pct`, `short_pct`, `flat_pct`)**：
  - 优秀策略在强趋势中坚定持仓（多头或空头占比鲜明），在横盘震荡或高不确定性时果断平仓观望（`flat_pct` 上升）。
  - `flat_pct` 高于80%或1~2步频繁反手可作为排查假设；是否错失机会或受到噪声影响，应结合风险预算、行情与对照实验判断。
- **持仓平均周期 (`mean_holding_bars`, `median_holding_bars`)**：
  - 旧字段实际统计同方向仓位状态连续段，包含空仓；不等于成交持仓回合。较短状态段可能伴随较高换手，但是否覆盖费用须逐笔核对；
  - 8～15 bars仅是待检验的经验启发式，不是验收阈值或择时能力证明。
- **动作持续性 (`action_persistence_steps`) 与推理跳过率**：
  - 依据 ADR 0045，非 Flat 持仓应物理锁定固定步数，跳过微观噪声推理，强行降低无效换手。

### 3.4 宏观 VAE 体制路由分流 (Macro Regime Routing)
- **体制九宫格覆盖度**：
  - 考察宏观路由在 9 个动力学槽位（Slot 0~8，由 `slope_bin` 0~2 与 `volatility_bin` 0~2 笛卡尔积构成）以及保底槽位（Slot 9 / Default）的步数分布。
  - **健康状态**：各槽位有合理的激活比例，顺应行情的动态迁移；
  - **退化状态**：单一 Slot 垄断超过 80% 步数（需排查 VAE 分辨率、门控边界及样本体制集中），或者大量步数跌入 Default/OOD Reject（需结合门控变化与独立校准检验漂移）。拒识比例下降只说明行为变化，不能直接证明置信度提高。宏观记录可能不含初始热身步数，分母应明确为有记录步数，同时报告覆盖率。

### 3.5 三级风控干预质量 (Risk Control Safeguards)
- **硬止损 (`hard_stop_losses`)**：标的价格相对持仓成本反向波动突破阈值（如 1.5%）时请求平仓；涨跌停、盘口、跳空和离散检查时点可能延迟成交或越过阈值。最终亏损率、持仓止损阈值与期间最大回撤不同，事件次数不能证明损失被锁定在某个比例内。
- **单向冷静期 (`stop_loss_cooldown_steps`)**：被止损后短时间内禁止同向开仓，防止在瀑布行情中逆势抄底接飞刀。
- **断路器熔断 (`circuit_breaker_suspension_steps`)**：单合约连续止损达到设定次数（如 2 次）后进入强制休眠（如 72 步），保护账户本金。

---

## 4. 盈利不佳的候选原因排查树 (Hypothesis Diagnosis Tree)

以下分支用于提出假设，描述性统计不能单独完成根因确认。先核对执行及时间账，再用同数据、同环境的消融/对照实验验证风控、路由、迟滞等机制的增量效果。

当策略表现不佳时，按照如下决策树自顶向下分级诊断：

```text
[诊断起点: 输入 diagnostics_summary.json / contract_pnl_friction.csv]
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   Gross PnL <= 0                Gross PnL > 0
 [场景 A: 盘面择时完全失效]               │
         │                             │
         ├─ 原因 A1: VAE严重OOD漂移     ├─────────────────────────────┐
         ├─ 原因 A2: 底层QNet反向预测   ▼                             ▼
         └─ 原因 A3: 选模指标过拟合  Net PnL <= 0                  Net PnL > 0
                                 [场景 B: 摩擦完全吞噬]                │
                                       │                             │
                                       ├─ 原因 B1: 调仓频次过高       ├─────────────────────────┐
                                       ├─ 原因 B2: 持仓周期过短       ▼                         ▼
                                       ├─ 原因 B3: 盘口滑点失控    MDD过大 / SR极低          表现优异 / 持续监控
                                       └─ 原因 B4: 对称换手未抑制 [场景 C: 尾部风险失控]     [场景 D: 达标策略]
                                                                     │
                                                                     ├─ 原因 C1: 单边行情扛单 (未设硬止损)
                                                                     ├─ 原因 C2: 止损后立即反扑 (无冷静期)
                                                                     └─ 原因 C3: 单合约严重失血 (熔断缺失)
```

### 场景 A：盘面择时毛利为负 ($Gross \le 0$) —— “方向看反了”
- **现象**：`total_gross_pnl <= 0`，策略即使在零手续费下也是亏钱的。
- **根因定位**：
  1. **低层模型能力缺失**：训练集未能学习到有效的短期动量与订单流特征；
  2. **选模指标陷阱**：在 `two_dimensional_selection` 阶段过度依赖未正则化的训练集回报率，选中了过度拟合的 checkpoint；
  3. **特征分布漂移 (OOD)**：验证集/测试集的波动率或偏度超出 VAE 训练时的极值范围，导致高层把“下跌”判定为“上涨”。
- **确认路径**：查看 `macro_routing_distribution.csv` 和 `two_dimensional_selection_manifest.json`，核对各槽位选出的模型及其在对应体制下的胜率。

### 场景 B：毛利丰厚但净利亏损 ($Gross > 0, Net \le 0$) —— “打工全给交易所”
- **现象**：`total_gross_pnl > 0` 且数值很大，但 `friction_consumption_ratio_pct > 100%`，最终 `total_net_pnl < 0`。
- **根因定位**：
  1. **微观高频倒手（Over-trading）**：`mean_holding_bars < 5`，每次价格稍有扰动就频繁平仓反手；
  2. **缺乏换手惩罚机制**：环境训练未接入换手惩罚，模型把微观噪声误当成盈利空间；
  3. **吃单滑点惩罚巨大**：在盘口价差大、深度浅的时段频繁下市价单。
- **确认路径**：查看 `contract_pnl_friction.csv` 中的 `est_commission`、`est_slippage` 与 `turnover_ratio`，以及 `contract_behavior_risk.csv` 中的 `mean_holding_bars`。

### 场景 C：整体盈利但回撤巨大 / 夏普过低 ($Net > 0, MDD > 20\%, SR < 1.0$) —— “赚小钱亏大钱”
- **现象**：大部分合约表现尚可（胜率可能达到 80%+），但个别月份合约出现断崖式下跌，拉垮整个组合夏普，MDD 飙升。
- **根因定位**：
  1. **单点灾难（如 ADR 0047 中的 `fu2411` 暴跌）**：模型在极端单边大跌中持续逆势抄底多头，一路扛单；
  2. **止损机制缺失或过宽**：未开启 `--stop_loss_return_threshold` 或阈值设得过大；
  3. **反复触碰止损后未冷却**：刚止损平仓，下一帧模型又发出做多指令，陷入“接飞刀-止损-再接飞刀-再止损”的死循环。
- **确认路径**：查看 `contract_pnl_friction.csv` 按 `net_pnl` 升序排列，找出垫底的 Worst Contracts；查看 `contract_behavior_risk.csv` 中的 `hard_stop_losses` 与 `circuit_breaker_suspension_steps`。

### 场景 D：收益极低且持仓极度被动 ($Net \approx 0, Flat\% > 75\%$) —— “过于保守踩空”
- **现象**：毛利和净利都在 0 附近徘徊，换手率极低，`flat_pct > 75%`。
- **根因定位**：
  1. **拒识门槛过严**：`ood_threshold` 设置过小，高层把正常行情误判为 OOD，长期强制 Flat；
  2. **边缘差值过大**：`slope_margin_threshold` / `volatility_margin_threshold` 设得过高，导致置信度不足而无法激活任何专业槽位；
  3. **断路器过于脆弱**：连续触发 1~2 次微小亏损即被长周期熔断。
- **确认路径**：查看 `macro_routing_distribution.csv` 中 `dynamic_default` 比例，以及 `high_level_agent_para.txt` 中的门控超参。

---

## 5. 策略调优与参数调整指南 (Actionable Tuning Playbook)

| 诊断结论 / 瓶颈 | 推荐调优杠杆 (Knobs) | 目标调整方向与操作建议 | 涉及代码与配置文件 |
| :--- | :--- | :--- | :--- |
| **摩擦吞噬过高 ($> 75\%$)** | 动作持续性 (`action_persistence`) | 从默认 `3` 调大至 `6` 或 `12`（锁仓 $1 \sim 2$ 小时，强制降低换手） | `final_result_fu_*.sh`<br>`high_level_heurstic_fu_*.sh` |
| **摩擦吞噬过高 ($> 75\%$)** | 非对称调仓惩罚 (ADR 0050) | 开启 `--turnover_adverse_ratio 6.0`，对逆势与震荡调仓施加 6 倍阻尼 | `parallel_diverse_train.py`<br>`futures_util.py` |
| **微观倒手 / 持仓过短** | 滞回退出比例 (`hysteresis_exit_ratio`) | 调大该参数（如从 `0.1` 提升至 `0.3`），防止在相邻状态间振荡切换 | `DiHFT_high_level_heurstic.py`<br>`gating.py` |
| **单合约极端暴跌扛单** | 标的硬止损 (`stop_loss_return_threshold`) | 缩紧止损阈值（如从 `0.02` 收窄至 `0.012`~`0.015`，即亏损 1.2%~1.5% 强制离场） | `final_result_fu_*.sh`<br>`vae_routing_final_result_macro_action.py` |
| **止损后频繁接飞刀** | 单向冷静期 (`stop_loss_cooldown_steps`) | 从 `6` 步延长至 `12`~`18` 步（禁止同向开仓 2~3 小时） | `final_result_fu_*.sh` |
| **单合约连续巨亏踩踏** | 断路器熔断 (`circuit_breaker_cooling_steps`) | 连续 2 次止损后，将休眠时间设为 `72` 步（12 小时交易时间不参与该合约） | `final_result_fu_*.sh` |
| **盘面毛利为负 ($Gross < 0$)** | 选模评价指标 (`selection_metric`) | 在 `high_level_heurstic` 中将 `--selection_metric` 从 `tr` 更改为 `annual_sr` 或 `daily_cr` | `high_level_heurstic_fu_*.sh` |
| **过于保守 / 长期 Flat** | 异常检测阈值 (`ood_threshold`) | 适当放宽门限（如从 `0.003` 调宽至 `0.008`~`0.010`），降低拒识率 | `high_level_agent_para.txt`<br>Optuna 搜索空间 |
| **宏观槽位单一垄断** | 动量/波动率边界阈值 (`margin_threshold`) | 减小边界裕度（如从 `0.15` 下调至 `0.08`），促使各槽位均衡激活 | `high_level_agent_para.txt` |

---

## 6. 分析必须读取的文件清单与依赖关系 (File Inventory)

为避免遗漏信息或无目的地反复查阅文件，建立如下标准文件清单：

```text
[实验根目录: analysis_result/DiHFT/high_level_heurstic/<dataset>/<exp>/]
    │
    ├── diagnostics/
    │   ├── diagnostics_summary.json      (核心：组合级总览、胜率、毛利、净利、摩擦拆解)
    │   ├── diagnostics_summary.md        (辅助：Markdown 全景简报)
    │   ├── contract_pnl_friction.csv     (核心：逐合约毛利、净利、佣金、滑点、夏普、回撤)
    │   ├── contract_behavior_risk.csv    (核心：持仓长短平分布、平均持仓周期、风控干预计数)
    │   └── macro_routing_distribution.csv(核心：九宫格 Slot 0~8 及 Default 分流分布)
    │
    ├── best_result.csv                   (核心：按 TR/SR/CR/MDD 选出的最优超参记录)
    ├── result.csv                        (参考：所有探索 Trial/Epoch 的全量指标)
    │
[执行与配置目录: result/DiHFT/final_result/<dataset>/<exp>/]
    │
    ├── high_level_agent_para.txt         (核心：当前正在运行的最优门控超参数配置)
    ├── contracts/<contract>/             (原始：模型、执行与账户审计工件)
    └── daily_audit/
        ├── audit_manifest.json           (核心：schema、运行身份、账户范围、TradingDay 来源与校验结果)
        └── trading_day=<YYYY-MM-DD>/
            ├── prediction_records.csv    (核心：真实模型推理、Action Value、路由与动作来源)
            ├── trade_records.csv         (核心：请求、执行 leg、成交状态、费用与账户关联)
            └── account_records.csv       (核心：独立合约账户初始及逐 step 快照)
    │
[选模元数据清单: analysis_result/DiHFT/low_level/<dataset>/<exp>/two_dimensional_selection/]
    │
    └── two_dimensional_selection_manifest.json (核心：九宫格各槽位选出的具体模型 Checkpoint 及指标)
    │
[相关 Python 核心代码]
    ├── FineFT/analysis/diagnostics/trading_diagnostics.py        (诊断指标计算与解耦实现)
    ├── FineFT/analysis/diagnostics/batch_diagnostics_loader.py   (批处理提取与自动诊断脚本)
    ├── FineFT/analysis/diagnostics/batch_code_loader.py          (strategy / audit 代码语义批量审查)
    ├── FineFT/analysis/pick_agent/DiHFT_high_level_heurstic.py   (高层选优与 best_result 生成)
    ├── FineFT/RL/DiHFT/high_level/vae_routing_final_result_macro_action.py (最终回测执行与风控逻辑)
    ├── FineFT/RL/DiHFT/high_level/vae_routing_util.py           (分级自适应门控与风控逻辑)
    ├── FineFT/env/env_class/base_env.py                          (账户推进、逐 step 审计快照与执行结果)
    └── FineFT/env/env_class/futures_util.py                      (钱包变化、成交数量、费用与盈亏语义)
```

---

## 7. 结构化六阶段分析实施路径 (Analysis Pipeline)

面对一个新完成的实验结果时，应按以下顺序执行。前一阶段的证据资格是后一阶段归因的前提；不能先看收益，再用不完整记录解释原因。

### 阶段 0：审计资格门禁 (Audit Eligibility Gate)
- **目标**：确认本次实验是否具备逐事件归因资格。
- **动作**：读取 `audit_manifest.json` 及三类日记录，检查 schema 和 run identity、模型/特征版本、TradingDay 来源、`independent_contract_wallet` 账户范围、分区完整性、主外键、账户恒等式、权益变化与原子发布状态。
- **判定**：任一硬条件失败即标 `NEEDS_AUDIT`，仅允许描述性汇总；不得继续声称模型预测有效、真实滑点优势或某机制产生增量收益。旧产物不因能够估算部分字段而自动通过此门禁。

### 阶段 1：宏观总览与研究等级 (Macro Health Check)
- **目标**：按冻结版本生成研究等级；不据验证集判定生产就绪。
- **动作**：通过批处理脚本读取 `diagnostics_summary.json`，检查：
  - `portfolio_return_pct`（在当前独立账户口径下是否赚钱？）
  - `portfolio_gross_return_pct`（参考价择时毛利是否充足？）
  - `friction_consumption_ratio_pct`（毛净差额是否失控？）
  - `win_rate_pct`、`mean_annual_sr` 与路径回撤（稳健性是否达到研究阈值？）

### 阶段 2：模型推理与决策来源 (Model Decision Attribution)
- **目标**：区分真实策略推理、动作持续和规则风控，并检查模型输出与最终请求动作的关系。
- **动作**：读取 Prediction Record 与 Trade Record：
  - 统计真实推理覆盖率、short/flat/long Action Value 分布及最优—次优 margin；
  - 按宏观 Slot、合约和 TradingDay 比较模型建议动作、请求动作及实际仓位；
  - 检查动作持续是否引用启动持续性的 Prediction Record；
  - 检查 warm-up、防御平仓、硬止损、冷静期和熔断是否保持空 `source_prediction_id` 并带明确决策原因。
- **限制**：Action Value 与 margin 只描述当前奖励定义下的相对动作价值，不是盈利概率、方向命中率或未来 1—2 个 TradingDay 的校准预测。

### 阶段 3：预测—交易—账户证据链 (Decision, Execution & Account Reconciliation)
- **目标**：把每个决策连接到真实模拟执行和账户后果，锁定毛净差额与风险累积来源。
- **动作**：沿 `Prediction Record → Trade Record → Account Record` 核对：
  - 请求仓位与实际仓位差异、拒绝、部分成交及反手的两个执行 leg；
  - 参考价、成交价、数量、手续费、已实现盈亏和滑点证据；
  - 每个执行 leg 的交易前后账户快照，以及权益、钱包、浮盈亏和保证金变化；
  - 多空平结构、真实持仓回合、费用来源和未解释残差；
  - 风控事件与期间回撤。触发次数仍不能单独证明机制有效，需同环境消融。

### 阶段 4：宏观路由与选模元数据溯源 (Routing & Manifest Trace)
- **目标**：检查高层 VAE 路由与底层 Q-Net 槽位协作是否健康，并用逐事件证据检验失效假设。
- **动作**：
  - 查看 `macro_routing_distribution.csv` 及 Prediction Record：是否有槽位未激活、单一槽位垄断、Default / Reject 过高，或某槽位持续输出低 margin 决策？
  - 查看 `two_dimensional_selection_manifest.json`：各槽位对应训练 Epoch、候选得分和覆盖，确认是否存在弱模型入选；
  - 将槽位、模型动作、实际成交与账户损益连起来。槽位激活与亏损共现只形成假设，不自动构成根因。

### 阶段 5：消融、超参回溯与冻结评价 (Ablation, Tuning & Freeze)
- **目标**：制定可验证的调优修改，并在统一环境中确认增量效果。
- **动作**：
  - 复合目标明确为 `J = 汇总净回报率 × 盈利合约比例`，记录资金分母、无交易合约处理及评分方向。有限搜索只称“本轮已评估候选中的最优”；
  - 对模型推理、路由、动作持续、执行和三级风控分别做同环境消融，不能由汇总数值变化完成因果归因；
  - 对照 `best_result.csv` 与 `high_level_agent_para.txt` 分析指标权衡，旧参数须在新版审计级环境重跑；
  - 冻结代码、模型、特征、参数、执行规则、审计 schema 和评价版本后再运行时间外测试。测试结果一旦用于调参，即转为开发数据。

---

## 8. 一次性批处理读取脚本使用规范 (Batch Diagnostics Loader)

为了彻底消除逐个文件调用读取导致的上下文膨胀、Token 浪费和多轮等待，系统内置了高效聚合脚本：
`FineFT/analysis/diagnostics/batch_diagnostics_loader.py`。

### 8.1 运行指令

在 `finetf` conda 环境下执行：

```bash
# 模式 1: 紧凑人类/LLM可读文本简报（推荐，约 60~80 行，高密度展示全局结论与自动调优建议）
conda run -n finetf python FineFT/analysis/diagnostics/batch_diagnostics_loader.py \
    --analysis_path analysis_result/DiHFT/high_level_heurstic/fu/10min_parallel \
    --format text

# 模式 2: 完整结构化 JSON 格式（便于脚本进一步程序化解析或传参）
conda run -n finetf python FineFT/analysis/diagnostics/batch_diagnostics_loader.py \
    --analysis_path analysis_result/DiHFT/high_level_heurstic/fu/10min_parallel \
    --format json
```

### 8.2 脚本核心功能与输出概览
1. **单次 I/O 汇聚**：一次性读取 `diagnostics_summary.json`、`contract_pnl_friction.csv`、`contract_behavior_risk.csv`、`macro_routing_distribution.csv` 以及 `best_result.csv`；
2. **派生核心诊断指标**：
   - 自动计算净毛比（$\text{Net PnL} / \text{Gross PnL}$）；
   - 自动计算每千步调仓频率（`trades_per_1k_steps`）；
   - 自动提取 Dominant Slot（主导路由槽位及占比）；
   - 自动截取 Top-K 最优与最差合约（直接暴露尾部亏损合约）；
3. **内置规则诊断引擎与警报标志 (Diagnostic Flags)**：
   - `[EXCESSIVE_FRICTION_CONSUMPTION]`：摩擦吞噬率 $> 80\%$；
   - `[ULTRA_SHORT_HOLDING]`：平均持仓周期 $< 5$ 步；
   - `[DIRECTIONAL_ALPHA_WEAK]`：毛利为负或微弱；
   - `[FAT_TAIL_CONTRACT_RISK]`：单一合约亏损拖累超过总本金 $15\%$；
   - `[MACRO_SLOT_MONOPOLY]`：单一槽位垄断 $> 80\%$；
4. **生成即时调优建议**：根据触发的标志自动输出具体的调优参数指令（如 `--action_persistence`、`--stop_loss_return_threshold` 等）。

该批处理工具提供**汇总指标层**的描述性体检，不替代阶段 0 和阶段 2–3 的逐事件证据检查。每日审计记录实现后，数据批处理入口还应汇总：真实推理覆盖率、动作持续引用完整率、规则动作比例、请求/实际仓位差异、拒绝/部分成交/反手 leg 数量、Prediction→Trade 与 Trade→Account 外键完整率、账户对账通过率、TradingDay 覆盖及每日收益/费用/回撤。在此接入完成前，工具输出必须明确标记“逐事件审计未加载”，不能把现有汇总称为全方位审计。

---

## 9. 核心 Python 源码批量提取与压缩工具 (Batch Code Loader)

策略诊断过程中，往往需要核对风控、撮合、动作持续性及门控选模的具体算法逻辑。若逐一读取 9 个相关的核心 Python 源码文件，将造成数万 Token 的极大浪费和严重的上下文漂移。

为此，系统配套开发了基于 AST 的代码批量提取与压缩工具：
`FineFT/analysis/diagnostics/batch_code_loader.py`。

### 9.1 内置代码清单 Profile

工具提供三个内置 profile；显式 `--files` 仍可覆盖 profile：

| Profile | 用途 | 主要覆盖 |
| --- | --- | --- |
| `strategy` | 路由、选模、门控和汇总诊断 | 原九个 DiHFT 策略核心文件 |
| `audit` | 逐事件证据、执行和账户语义 | 交易诊断、数据汇总、高层运行器、账户环境、钱包/撮合工具及 artifact 契约 |
| `all` | 联合审查 | `strategy` 与 `audit` 去重并集 |

`audit` profile 特别纳入 `base_env.py` 与 `futures_util.py`。仅检查高层路由代码无法确认成交数量、账户推进、费用、已实现盈亏和审计快照的真实语义。每日审计生成器及 schema 模块实现后，应继续加入该 profile。

### 9.2 三种提取模式与运行指令

在 `finetf` conda 环境下执行：

#### 模式 1：AST 架构大纲、schema 字段与参数提取 (`outline` 模式，强烈推荐)
- **特点**：通过 Python 原生 AST 遍历，提取模块说明、类继承、类字段（包括 dataclass / Enum / artifact 常量）、方法签名、顶级函数以及 `argparse` CLI 参数。类字段让审计记录契约可以在大纲中直接检查，不必打开全量源码。
- **Token 节约**：用于快速定位接口和 schema；节省比例取决于所选 profile 和源码版本，不把固定 95% 当作保证。
- **运行命令**：
```bash
conda run -n finetf python FineFT/analysis/diagnostics/batch_code_loader.py \
    --profile audit \
    --mode outline
```

#### 模式 2：语义关键字定向抽取 (`targeted` 模式)
- **特点**：递归检查类与函数，优先返回最小匹配方法，避免某个关键词位于 `Base_Env.step` 时输出整个大型环境类。默认关键词覆盖风控、摩擦、路由以及 `audit`、`prediction`、`action_value`、`trade_event`、`execution`、`account_snapshot`、`trading_day`、`reconciliation`、`commission`、`realized_pnl` 和 `slippage`。
- **运行命令**：
```bash
conda run -n finetf python FineFT/analysis/diagnostics/batch_code_loader.py \
    --profile audit \
    --mode targeted \
    --keywords prediction action_value trade_event account_snapshot trading_day reconciliation
```

#### 模式 3：全量代码一键打包 (`full` 模式)
- **特点**：一次性将指定代码文件拼接打包，并生成标准代码块与带行号视图，避免多次发起文件读取请求。
- **运行命令**：
```bash
conda run -n finetf python FineFT/analysis/diagnostics/batch_code_loader.py \
    --profile all \
    --mode full \
    --output analysis_result/code_bundle.md
```

`full` 模式用于已定位问题后的完整复核，不应作为默认诊断入口；`audit` profile 中的环境与撮合文件较大，优先使用 `outline` 或 `targeted`。

---

## 10. 汇总、逐事件证据与代码语义三层协同

完整诊断由三层证据共同构成，不能以固定“两次调用”代替证据资格判断：

1. **汇总指标层**：`batch_diagnostics_loader.py` 汇聚净值、参考价毛利、毛净差额、换手、风险指标、路由分布与尾部合约，回答“结果表现如何”。
2. **逐事件证据层**：`audit_manifest.json` 及按 TradingDay 分区的 Prediction / Trade / Account Records 回答“哪次推理驱动了什么动作、实际如何执行、账户为何变化”。这一层先执行资格校验，再提供阶段 2–3 的归因证据。
3. **代码语义层**：`batch_code_loader.py --profile audit` 检查记录生成、动作来源、成交、账户和对账实现，回答“这些字段在代码中究竟如何产生”。

推荐顺序是：先读 manifest 完成阶段 0，再加载汇总指标定位问题，然后按异常 TradingDay / 合约 / Decision ID 下钻三类记录，最后用 `targeted` 模式抽取对应方法。若逐事件记录或账户对账缺失，只能报告描述性现象和审计缺口，不能用源码设计意图补足缺失事实。
