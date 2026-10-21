---
status: accepted
---

# 0061. 空仓降级槽位平滑退出与禁止新增风险语义架构 (Graceful Hold Exit and No New Risk Semantics for Flat Degradation Slots)

针对商品期货（fu 10min/30min）低层 Agent 选拔中因恐惧“空槽位暴力强平”而不断放水降标、导致劣质模型混入生产组合造成严重失血的问题，建立“宁缺毋滥的严格证据底线（Strict Evidence Floor）”、“空仓降级槽位平滑退出（Graceful Hold Exit）”与“禁止新增风险（No New Risk）”架构。用户于 2026-10-21 质询访谈中确认全部决策。

---

### 1. 业务与物理背景

1. **历史走偏与妥协的沉重教训**：
   - 在 `ADR-0045` 的早期定义中，空模型槽位被归入“防御优先抢占（Defensive Preemption）”，规定只要高层门控切入该槽位，无论当前处于何种持仓状态，立即清零动作持续期（`remaining_persist = 0`）并强行调用 `rule_based_close` 规则平仓；
   - 这种“暴力强平”机制导致系统只要微观上在某个无模型槽位边缘轻微振荡，就会频繁触发非预期的被动平仓，造成巨大的手续费与滑点磨损；
   - 为了回避这一损耗，在 `ADR-0059` 落地实施时走向了另一个极端——**通过不断降低门禁指标强行拼凑满 9 个槽位**（如将置信度 $z$ 降至 0.2、旁路边际检查、削减合约覆盖）；
   - **后果**：强行填满槽位的代价，是把根本没有稳健统计证据的劣质模型（如入场即亏损的 Slot 4、频繁止损的 Slot 6/7）放进了生产组合，导致实盘大幅失血。
2. **核心逻辑反思**：
   - 过去之所以怕“空槽”，是因为旧逻辑把空槽简单理解为“每步强制平仓”；
   - **正确的空槽逻辑是“禁止新增风险（No New Risk）”而非“暴力强平”**：当市场处于无合格模型的 Regime 时，严禁新开仓；但若已有来自前序槽位的存量持仓，应给予合理的持续期缓冲和平滑看护，直至安全退出。

---

### 2. 核心架构决策

#### 2.1 宁缺毋滥的严格证据底线 (Strict Evidence Floor)
彻底废止以凑齐槽位为导向的任何放水妥协，坚守可信统计下限：
1. **统计置信度底线**：
   - 全局与边际 LCB 置信度系数严格锚定 $z \ge 1.0$（对应单侧 84% 以上保守置信下限），绝不退回 $z=0.2$ 的伪折扣；
2. **双重边际-联合强制合规 (Conjunctive Marginal-Joint Gate)**：
   - 候选 Agent 必须同时满足：
     $$\text{MeanReturn}_{\text{vol}} > 0 \land \text{LCB}_{\text{vol}} > 0$$
     $$\text{MeanReturn}_{\text{slope}} > 0 \land \text{LCB}_{\text{slope}} > 0$$
     $$\text{MeanReturn}_{\text{joint}} > 0 \land \text{LCB}_{\text{joint}} > 0$$
   - 任何一个边际表现为负期望或无统计显著性，一票否决淘汰；
3. **流量感知梯级证据标准与长尾留空机制**：
   - **核心高流量槽位 (Core High-Traffic Slot)**：验证集时间步占比 $\ge 10\%$ 且覆盖合约数 $\ge 5$，联合合约覆盖数 $\ge 4$，正收益合约比例 $\ge 67\%$；
   - **长尾低频槽位 (Peripheral / Long-Tail Slot)**：若样本稀疏或候选人无法在独立合约块上取得稳健正收益，**直接标记为 `kind = "empty_model"`（空仓降级槽位），绝不降标硬塞**；
4. **日度主力独立达标子门禁**：
   - 在具备有效样本（$\ge 100$ 步）的日度主力合约时间步子集上，强制要求候选 Agent 扣费单步净收益均值必须大于 0。

#### 2.2 空仓降级槽位平滑退出语义 (Graceful Hold Exit & No New Risk)
重构高层 VAE 路由（`vae_routing_util.py`）中对 `kind == "empty_model"` 的执行逻辑：
1. **空仓状态下：严格禁止开立新仓 (No New Position)**：
   - 当 `current_position == 0.0` 时，强制输出 `zero_position_action`（Flat，Action 1）；
   - 决策原因记录为 `DEFENSIVE_RULE_CLOSE`，宏观动作记录为 `self.slot_count`。
2. **持有存量持仓下：持续期缓冲与平滑退出 (Graceful Hold Exit)**：
   - **免除底层前向推理**：不调用任何底层 Q 网络的推断，避免由于 Regime 概念漂移引发失控预测；
   - **持续期内保持原持仓 (Hold)**：存量持仓继续消耗其开仓时赋予的动作持续期缓冲（`self.remaining_persist`）：
     - 当 `remaining_persist > 0` 时，动作映射为保持当前持仓档位的 `Hold`（通过 `map_position_leverage_to_action` 实现），`remaining_persist` 逐步递减；决策原因标记为 `ACTION_PERSISTENCE`；
   - **持续期耗尽优雅平仓 (Graceful Close)**：
     - 当 `remaining_persist == 0` 且下一时刻市场仍处于空仓降级槽位时，判定该工况缺乏有效模型护航，立即调用 `rule_based_close` 逐步平仓至 0；决策原因标记为 `DEFENSIVE_RULE_CLOSE`；
   - **极端风险最高优先级防御**：
     - 在持续期缓冲期间，若当前持仓浮亏触发 3% 相对硬止损（Tier 1）或单合约断路器（Tier 3），无条件解除锁定并立即强平。

#### 2.3 系统级覆盖率安全底线与 Fail-Fast 机制
1. **有效时间步覆盖率统计**：
   - Stage II 2D Agent Selector 在产出清单时计算通过门禁的合格槽位在验证集上的累计时间步覆盖率：
     $$\text{TrafficCoverageRatio} = \sum_{i \in \text{valid\_slots}} \text{traffic\_ratio}_i$$
2. **Fail-Fast 熔断中断**：
   - 若合格槽位累计覆盖率低于安全底线（`min_ensemble_traffic_coverage = 0.40`）或核心中枢槽位（横盘震荡）全军覆没，选拔脚本默认抛出 `RuntimeError` 阻断执行；
   - 提示研究员 Stage I 训练缺乏足够的表征容量或策略多样性，防止无效实验静默放行；
   - 仅在显式传入 `--allow_sparse_ensemble` 参数时允许豁免并输出稀疏清单。

---

### 3. 与既有 ADR 的演进与取代关系

1. **修正 ADR-0045**：
   - `ADR-0045` 将空模型槽位归为“防御优先抢占（Defensive Preemption）瞬时强平”。本 ADR 明确将空槽位从防御抢占中剥离，修正为遵循“平滑退出与禁止新增风险（Graceful Hold Exit & No New Risk）”；防御抢占仅保留针对 OOD 漂移、3% 持仓浮亏硬止损与单合约断路器。
2. **取代 ADR-0059 中的降标妥协条款**：
   - 废除 `ADR-0059` 在第 4 节历史回测校准中为凑满 9 槽位而引入的临时妥协（如放宽边际检查、低置信度 $z=0.2$ 等），恢复 $z \ge 1.0$ 与双重边际强合规的初始设计。

---

### 4. 产物与交付界面

- **选拔脚本与逻辑**：
  - `FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py`
  - `FineFT/script/analysis/pick_agent/low_level_fu_10.sh` 与 `low_level_fu_30.sh`
- **高层路由执行逻辑**：
  - `FineFT/RL/DiHFT/high_level/vae_routing_util.py`
- **产物元数据**：
  - `two_dimensional_selection_manifest.json` 中记录各 Slot 的 `kind`（`model` 或 `empty_model`）以及整体 `traffic_coverage_ratio`。
