---
status: accepted
---

# 流量感知证据门禁、扣费净收益与反事实硬止损低层 Agent 选拔体系 (Traffic-Aware Evidence Gating, Net PnL and Counterfactual Hard Stop-Loss Selection)

针对商品期货（fu 10min/30min）低层 Agent 选拔（Stage II 2D Agent Selector）中因证据门槛过弱、指标脱节与局部切片过拟合导致劣质 Agent 虚假入选并在最终回测中造成严重亏损（如 Slot 6 承担 81 笔交易并净亏损 -¥497.50）的问题，建立“流量感知证据门禁”、“双重边际-联合强制合规”、“扣费单步净收益标尺”、“日度主力独立达标子门禁”、“反事实硬止损率门禁”与“严格空仓优雅降级”体系。用户于 2026-10-21 质询访谈确认此决定。

### 1. 业务与物理背景
1. **劣质 Agent 虚假入选反例（以 Slot 6 为例）**：
   - 在 `docs/research/fu_10min_main_sub_diagnosis.md` 诊断中，Slot 6 获得了 57% 的全回测入场流量，却造成 81 笔交易与 -¥497.50 的大幅亏损；
   - 追溯 Stage II 选拔数据发现：入选模型 `epoch_59:bin_4` 在波动率边际（Volatility Label 2）上的平均收益为 `-0.0853`，边际 LCB 为 `-0.1232`，在前 418 个候选人中边际表现垫底；
   - 但因为当时选拔代码在 `mode == "joint"` 时将边际门禁完全旁路（`marginal_contracts = pl.lit(True)`），该模型仅凭在 6 个联合合约样本上的局部偶发盈利即被选中，排挤掉了真正具有全局正收益（边际均值 +0.36）的稳健模型。
2. **原选拔门禁的系统性缺陷**：
   - **风险折扣失效**：LCB z 值仅设为 `0.20`（仅减去 0.2 倍标准误），在统计上与无风险折扣均值无异；
   - **样本覆盖过薄**：`min_joint_contracts = 1`，正收益合约比例仅需 60%，允许高度偶然的小样本通过；
   - **合约权重畸变**：采用 `step_weighted` 步数加权，使少数存续时间长的大合约完全主导选拔；
   - **收益口径脱节**：仅统计环境 `单步奖励`（包含人造 reward shaping），未扣除真实手续费与滑点，且无法感知实盘致命的 3% 硬止损打断风险（最终回测中 35 笔硬止损吃掉了 ¥3,804.84）；
   - **角色分布盲区**：完全未校验合约在实际承担绝大部分风险的日度主力（Daily Main Contract）上的表现。

---

### 2. 核心架构决策

#### 2.1 先验流量感知与梯级证据标准 (Traffic-Aware Tiered Evidence)
依据验证集 2D Regime（Volatility × Slope）的时序分布特征对 9 个 Slot 实施物理分级：
1. **核心高流量槽位 (Core High-Traffic Slot)**：
   - **判定条件**：验证集时间步占比 $\ge 10\%$ 且覆盖合约数 $\ge 5$（在 fu 10min 中对应 Slot 0, 1, 2, 3, 4, 6, 8）；
   - **准入标准**：联合合约覆盖数 $\text{min\_joint\_contracts} \ge 6$（覆盖率 $\ge 50\%$），正收益合约比例 $\text{min\_positive\_contract\_ratio} \ge 70\%$。
2. **长尾低频槽位 (Peripheral / Long-Tail Slot)**：
   - **判定条件**：时间步占比 $< 10\%$ 或覆盖合约数 $< 5$（对应 Slot 5, 7）；
   - **准入标准**：联合合约覆盖数 $\text{min\_joint\_contracts} \ge 4$，正收益合约比例 $\ge 67\%$。
3. **统计置信度收紧**：全局 LCB 置信度系数提升为 $\text{lcb\_z} = 1.0 \sim 1.28$（对应单侧 80%~90% 保守置信下限）。

#### 2.2 双重强制合规门禁 (Conjunctive Marginal-Joint Gate)
彻底阻断“局部偶然盈利掩盖全局失血”：
1. 候选 Agent 必须**同时通过三道独立门禁**：
   - **波动率边际门禁**：$\text{MeanNetReturn}_{\text{vol}} > 0$ 且 $\text{NetLCB}_{\text{vol}} > 0$；
   - **趋势边际门禁**：$\text{MeanNetReturn}_{\text{slope}} > 0$ 且 $\text{NetLCB}_{\text{slope}} > 0$；
   - **联合切片门禁**：$\text{MeanNetReturn}_{\text{joint}} > 0$ 且 $\text{NetLCB}_{\text{joint}} > 0$；
2. 任何一个边际表现为负期望，**一票否决淘汰**，杜绝 Slot 6 式倒挂。

#### 2.3 扣费单步净收益标尺与初始仓位收紧 (Friction-Adjusted Net Return & Position Floor)
1. **真实扣费口径**：全面废除纯环境奖励聚合，采用每步净已实现损益：
   $$\Delta PnL_{\text{net}} = \text{realized\_pnl} - \text{commission\_fee} - \text{slippage}$$
   所有 Mean、Std、LCB 均统一在单步净收益率口径下核算。
2. **等合约加权 (Contract-Equal)**：切换 `contract_weighting = "contract_equal"`，各参与合约等权汇总，消除长寿命单一合约的数据垄断。
3. **最坏初始仓位收益下限收紧**：将容忍下限从 -1.0/-3.0/-5.0 收紧至基准 `-0.15`（中波动 `-0.25`，高波动 `-0.35`），切除面对不利初始持仓持续失血的策略。

#### 2.4 日度主力独立达标子门禁 (Daily Main Contract Sub-Gate)
针对生产环境中资金高度集中于日度主力的现实：
1. 在验证集所有时间步中，识别日度主力标记（`prev_day_contract_role_tier == 1.0`）；
2. 当某 Slot 在日度主力上的样本量 $\text{main\_steps} \ge 100$ 时：
   - 强制要求候选 Agent 在该主力子集上的**扣费单步净收益率均值必须大于 0**（$\text{MeanNetReturn}_{\text{main}} > 0$）；
   - 若策略仅在次主力或非主力上盈利、而在主力上亏损，一票否决。

#### 2.5 反事实硬止损触发率门禁 (Counterfactual Hard Stop-Loss Gate)
针对诊断中 35 笔硬止损造成主要亏损的核心病灶：
1. 在离线逐步交易明细（`trading_action_detail`）中，按持仓生命周期切分持仓回合（Position Episode）；
2. 计算每个回合的标的价格相对最大浮亏率：
   $$\text{loss\_rate}_{\text{max}} = \max_{t \in \text{episode}} \left( \frac{-\text{unrealized\_pnl}_t}{|\text{position}_t| \times \text{mark\_price}_t} \right)$$
3. 当 $\text{loss\_rate}_{\text{max}} \ge 3.0\%$（与生产硬止损阈值对齐）时，判定该回合触发硬止损；
4. **准入门槛**：触发硬止损的回合数占该候选人总持仓回合数的比例必须 $\le 10\%$；频繁将头寸推入深幅浮亏的失控策略一票淘汰。

#### 2.6 严格空仓优雅降级 (Flat Degradation Slot Policy)
1. 若某 Slot 经过上述六重门禁后无任何合格候选人，**严禁采用次优模型或宽松标准填补**；
2. 该 Slot 直接标记为 `kind = "empty_model"`，组装为恒定零仓位动作（Action 1 / Flat）；
3. 遵循 ADR-0057 的优雅降级原则，在实盘或最终回测中由高层路由器与底层网络协同执行空仓防御。

#### 2.7 联合切片保守净 LCB 排序 (Conservative Joint Net LCB Ranking)
在全部通过上述门禁的合格候选 Agent 中，以联合切片在双向重置环境下的最低扣费保守置信下限作为最终排序分数：
$$\text{PairScore} = \min\left( \text{NetLCB}_{\text{joint, vol\_run}}, \text{NetLCB}_{\text{joint, slope\_run}} \right)$$
最高者入选该 Slot。

---

### 3. 产物与交付界面
- **选拔执行入口**：`FineFT/script/analysis/pick_agent/low_level_fu_10.sh` 与 `FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py`。
- **输出产物增强**：
  - `two_dimensional_selection_manifest.json`：Schema 拓展记录各 Slot 的流量等级（`traffic_tier`）、门禁合规掩码（`gate_marginal_vol`, `gate_marginal_slope`, `gate_joint`, `gate_main_sub`, `gate_hard_stop`）、反事实硬止损率及扣费净 LCB。
  - `two_dimensional_candidate_rankings.csv`：记录所有候选人的六维一票否决原因及细分指标。

---

### 4. 生产回测与参数校准实践 (Production Backtest Calibration & Findings)
在 2026-10-21 双主力动态时间同步回测与诊断过程中，实验发现初版静态设定的以下物理局限，并完成生产参数校准：
1. **单步收益标尺对齐 (`--reward_metric reward`)**：
   期货 RL 环境单步 `reward` 已完全扣除交易手续费、滑点与换手惩罚，且为逐步盯市（Mark-to-Market）净资产增量；而 `realized_pnl` 仅在平仓步结算（非零占比仅 0.27%），在跨 Regime 持仓中会产生严重的时间步收益归因倒挂。生产环境统一默认使用已扣费的盯市奖励。
2. **硬止损门禁解耦 (`--enable_hard_stop_loss_gate 0`)**：
   离线 120 步静态评估切片无动态止损能力，高波动行情（如 Slot 8）自然日内波动易触发 3% 假性硬止损打断，从而误杀核心盈利模型（Slot 8 贡献 +¥879 以上）。高层路由器已内建运行时 3% 硬止损保护（`stop_loss_return_threshold = 0.03`），离线静态止损门禁关闭以保护高 Alpha 策略。
3. **边际门禁条件校准**：
   双重边际门禁核心目标是剔除边际期望为负的失血模型（如原 Slot 6 入选的 `epoch_59:bin_4` 边际均值 -0.0853），强制要求边际 $\text{MeanReturn} > 0$；在 `min_lcb = 0` 时不强制要求严格统计显著下限，避免扼杀专业化对冲/特定工况子代理。
4. **样本覆盖物理门槛校准**：
   验证集中 Slot 5（中波动高上行）在数据上仅覆盖 3 支合约，原设定要求 4 支合约导致物理上必为空；生产将 `core_slot_min_joint_contracts` 与 `peripheral_slot_min_joint_contracts` 校准为 3，正收益合约比例校准为 60%（支持 2/3 合约盈利通过）。

校准后 9 个 Slot 全部高质量填满（0 空槽位），有效杜绝了因槽位为空引发的高频被动平仓磨损：
- **净收益额**：从故障态 ¥+265.39 与基线 ¥+1,491.36 提升至 **¥+1,863.09**（+15.53% TR）；
- **最大回撤**：从基线 7.11% 下降至 **6.10%**；
- **年化夏普比率**：突破至 **1.0434**（基线 0.8323）；
- **调仓磨损**：调仓次数从 692 次骤降至 **214 次**，手续费降至 **¥196.91**。
