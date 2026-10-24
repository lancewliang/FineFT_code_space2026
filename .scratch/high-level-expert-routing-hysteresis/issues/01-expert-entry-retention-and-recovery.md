# 01: 接通专家进入、保持与规则恢复

**What to build:** 研究人员能够显式选择 expert_hysteresis，通过既有公开路由入口完成首次进入、当前专家保持、失效后的规则接管与恢复。当前专家身份独立于仓位和上一次动作；专家继续适用时，每步依据最新观察与真实账户重新给出动作。本票交付该闭环，专家仍有效时的替换由 02 扩展。

**Blocked by:** None (can start immediately).

**Status:** ready-for-human

依据本功能实施规格与 ADR-0062；复用既有双轴门控工厂及公共决策契约。必要的契约适配随本票完成，不另开纯接口重构票。

- [x] 通过既有公开配置和工厂选择 expert_hysteresis，并由公开 get_action 返回可执行动作；absolute、hierarchical 的既有配置含义和行为通过回归场景保持可复现。
- [x] 当前专家用 Selection Manifest 中既有的二维 Slot 身份表示；首次进入及规则恢复要求同一最高分普通模型 Slot 的两个轴均达到各自 E，连续 K 个合格决策步后接管。确认完成前使用既有规则分支，不附加领先第二名的门槛。
- [x] 当前专家有效性只检查该 Slot 自身的波动率、斜率分数：两轴均大于等于 H 才有效；任一轴小于 H 时当步撤销资格，不能被另一专家的高分保活。尚无已确认替代者时当步进入规则接管模式。
- [x] H_axis = exit_ratio × E_axis；新策略配置满足 0 <= H < E <= 1、0 < exit_ratio < 1、有限且非负的 Δ，以及正整数 K。非法配置明确报错；H/E 的等值边界按规格作包含比较。
- [x] 当前专家输出 Flat 后仍保留专家身份并使用保持阈值。普通专家路径即使 legacy 动作持续计数非零，也每步用最新观察、真实前一动作、账户状态和可用动作掩码重新推断。
- [x] 在公开入口用固定双轴分数、可控账户和逐步改变输出的假专家，证明同一专家的最终动作随新输出变化；仅断言专家 ID 不足以通过验收。
- [x] 初次进入及恢复的候选变化从 1 重计，不合格步骤清除候选证据；完成确认后清除待确认状态。初次预热沿用既有评分协议，预热结束后正常累计 K。
- [x] 输入空向量、NaN/Inf、越界分数、轴维度变化、标签顺序或 Slot 映射变化时明确中止运行，不裁剪、不回退为普通低置信度防御结果。
- [x] 每个独立合约 rollout、独立运行和重配置明确清除当前专家与候选状态；共享只读数据不携带状态。通过公开入口的连续运行场景验证无跨合约、跨运行污染。
- [x] 主测试接缝为公开 get_action，观察选中专家、最终动作及公共归因结果；沿用既有路由与动作持续性 fixtures，不为计数器另建私有测试接缝。现有风险防御与空槽行为继续生效，专项连续性验收见 03。

## Implementation evidence

- Added separately selectable `expert_hysteresis` to the factory and public router CLI. The strategy version is `expert_hysteresis.v1`; shared replacement lead and confirmation parameters are `expert_switch_lead` and `expert_confirm_steps`.
- Public `get_action` tests verify entry/recovery confirmation, own-Slot H invalidation, fresh changing expert actions despite nonzero legacy persistence, Flat ownership, strict invalid input handling, and manifest identity checks.
- Public rollout tests execute warm-up plus K-step entry repeatedly across independent contracts/runs. Reset/reconfiguration clears ownership and candidate state.
- Validation: 86 tests passed across the new routing tests plus gating, legacy hysteresis, action persistence, stop/circuit-breaker, quantile cache, and final-result regression suites.
- Atomic replacement while an incumbent remains valid, interruption continuity under risk, and expanded audit output remain in tickets 02–04 as planned.
