# 03: 贯通风险抢占与空槽平滑退出

**What to build:** 专家保持和候选确认始终服从既有风控及 No New Risk 约束。主导空槽期间不调用过期专家，也不新增风险；存量持仓仅消耗有限缓冲后按既有规则退出。风险或空槽阻断会中断候选连续证据，使恢复后的确认可以被逐步验证。

**Blocked by:** 02 — 完成原子 Slot 切换与候选连续确认。

**Status:** ready-for-human

依据本功能实施规格、ADR-0062 及 ADR-0047/0057/0060/0061。本票需要 02 的完整替换状态，才能验证接近确认完成时的风险与空槽阻断。

- [x] 硬止损和断路器按既有优先级当步处置，不等待 K，也不被当前专家身份、候选计数或非零动作持续计数覆盖。
- [x] 方向冷静期、非主力合约限制与动作可用性继续约束最终动作；覆盖被拦截的开仓、加仓、反手及允许的持有或减仓场景，迟滞不扩大可用风险范围。
- [x] 风险或空槽防御实际阻断的步骤不计入合格候选确认，并中断此前连续证据；阻断结束后不能接续旧计数直接接管。覆盖候选已有 K-1 步时被抢占的场景。
- [x] 按既定并列优先级选出的主导 Slot 若为 empty_model，不作为合格普通专家累计确认，不回填次优模型，也不在空槽期间调用之前的普通专家。
- [x] 空槽且 Flat 时保持禁止新增风险；有存量持仓时沿用既有有限缓冲及动作映射，缓冲逐步耗尽后进入规则退出。测试同时覆盖长、短持仓及不可用的保持动作。
- [x] 空槽缓冲不能因每步重新观察到空槽、路由保持或候选状态更新而无限补满；普通专家每步推断与空槽有限保持是可区分的行为。
- [x] 在空槽缓冲期间触发硬止损、断路器或其他必需的防御时，仍当步按原风控处理；规则退出遵循真实执行约束，不将请求退出等同于保证全部成交。
- [x] 用公开 get_action、可控账户及假专家调用记录验收：断言最终动作、公开原因、未发生的过期专家推断和阻断后的重新确认，复用既有风险、空槽及非主力防御 fixtures。
- [x] 历史 absolute、hierarchical 策略及 ADR-0060 的合约边界、账户与计量约束通过相关回归场景；本票不调整风控数值、费用、Reward Shaping 或 DP Q 表。

## Implementation evidence

- Pre-inference hard stops, circuit breakers and strict non-main restrictions clear pending evidence without querying a stale expert.
- Actual post-inference cooldown, non-main reversal/expansion and unavailable-action interceptions clear confirmation and roll back provisional Kth-step admission. Only the previously confirmed incumbent may retain identity, and only while its own H remains valid.
- Public decisions retain `queried_slot` and `proposed_action` when a genuine query is intercepted; persisted prediction/audit integration remains ticket 04. New model errors propagate explicitly.
- Empty dominant Slots clear ownership/candidates, forbid new risk, never query old experts, and consume the existing finite action buffer exactly once per held step for both long and short positions. Hard risk still preempts the buffer.
- Public tests verify K-1/Kth blocking, initial admission and invalid-incumbent rollback, allowed hold/reduction/opposite-direction actions, finite empty buffers, masks, and liquidity-constrained rule requests without changing account state.
- Validation: 166 tests passed across new routing/switching/risk scenarios and legacy gating, persistence, stops, non-main defense, reversal, cache, final-result and account regressions. Ruff check/format passes for the new strategy/risk tests; strategy pyright reports zero errors.
