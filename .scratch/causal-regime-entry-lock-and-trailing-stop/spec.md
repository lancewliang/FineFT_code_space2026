# Specification: Causal Regime Entry Lock & Trailing Profit Stop Engine

## Problem Statement

As a quantitative futures trader and portfolio researcher running long-horizon reinforcement learning strategies, I face severe profit degradation in core contracts running over 2,800 steps. 

Specifically:
1. In prolonged macro trends (such as a 4-month continuous bull run in `fu2503`), the offline dynamic programming (DP) expert table—possessing full omniscient knowledge of all future prices—shorts during minor 2-hour intraday pullbacks and quickly covers at the exact bottom. The low-level RL agent, restricted to a 16-hour micro feature horizon, memorizes these counter-trend demonstrations and aggressively attempts to top-pick in live trading, incurring severe losses.
2. In major cyclical round-trips (such as `fu2505` surging from 2,956 to 3,600 and crashing back to 2,771), the strategy accumulates substantial floating profits (e.g. +697.56 RMB, +11.63% return) but holds long through the decline, surrendering over 51% of its peak profits. This occurs because the DP table exits with 0% retracement at the exact peak tick, leaving the experience buffer devoid of trailing retracement exit demonstrations.
3. In Stage I training, diverse exploration across multiple epochs washes out pretraining-only modifications unless backed by execution-layer and environment-aligned structural guardrails.

## Solution

A dual-track, mathematically consistent risk-control and expert-regularization architecture:

1. **Directional Entry Lock**: Enforce an asymmetric physical action mask in both the offline DP Bellman backward induction and the online high-level execution router. In bull regimes, new short positions and flips to short are strictly forbidden, while long-to-flat exits for capital preservation remain fully permitted. In bear regimes, new long positions and flips to long are strictly forbidden, while short-to-flat exits remain permitted.
2. **Tier 4 Trailing Profit Stop Engine**: Implement a deterministic execution-layer risk control engine operating independently of low-level neural network predictions. Once unrealized position return crosses an activation hurdle (+8%), the engine tracks peak floating return. If floating return retraces by 25% from the peak, the engine forcibly closes the position to flat, guaranteeing capital preservation.
3. **Dual Spatiotemporal Re-entry Guard**: Following a trailing stop exit, enforce a combined time and price hurdle. Re-entry in the same direction is prohibited until both a 24-step cooldown timer elapses and market price breaks out above the previous peak price (or the macro regime shifts), completely eliminating premature re-entries during persistent declines.
4. **Comprehensive Decision Attribution**: Integrate events into the standardized audit triad with dedicated decision reason enumeration, CSV risk summaries, and performance attribution artifacts.

## User Stories

1. As a portfolio researcher, I want the offline dynamic programming solver to reject counter-trend entry actions in strong macro regimes, so that the pretraining Q-table is not contaminated by unrepeatable omniscient scalping.
2. As a trading system operator, I want the DP solver to permit closing existing positions to cash in any regime, so that de-risking and capital preservation trajectories remain available to the policy.
3. As an execution engine, I want to mirror the exact DP regime action mask during online routing inference, so that the low-level agent cannot open counter-trend positions during verified macro trends.
4. As a quantitative risk manager, I want the execution layer to monitor single-holding unrealized return, so that high-watermark profit peaks can be tracked in real time.
5. As an investor, I want the trailing profit stop engine to remain dormant until a minimum activation profit threshold (+8%) is achieved, so that ordinary intraday noise does not trigger premature exits.
6. As an investor, I want the trailing profit stop engine to trigger an immediate, unconditional flat closure when profit drops by 25% from its peak, so that the majority of accumulated gains are locked in.
7. As a portfolio manager, I want an absolute breakeven profit floor (+0.3%) enforced during trailing stop execution, so that an activated profit lock never exits at a net loss.
8. As a trading algorithm, I want an active trailing stop event to clear action persistence counters, so that multi-step persistence locks do not delay immediate flat closure.
9. As a risk controller, I want a directional cooldown timer (24 steps / 4 hours) initiated immediately after a trailing stop, so that the strategy is temporarily prevented from re-entering in the same direction.
10. As a quantitative researcher, I want a price-level re-breakout requirement enforced alongside the cooldown timer, so that the strategy cannot re-enter in the stopped direction unless price exceeds the prior peak.
11. As a trader, I want the peak price hurdle to reset automatically whenever the high-level regime shifts, so that new macro market cycles are not blocked by outdated peak memories.
12. As a trader, I want opposite-direction trades permitted immediately after a trailing stop, so that legitimate trend reversals can be captured without artificial delay.
13. As an ML engineer, I want the low-level Q-net to remain decoupled from raw regime indicators, so that out-of-distribution generalization is not compromised.
14. As an ML engineer, I want existing model checkpoints to benefit from the Tier 4 trailing stop engine without requiring retraining, so that historical models immediately achieve profit-preservation gains.
15. As a backtest analyst, I want trend lock interceptions and trailing stop events recorded in decision reason history arrays, so that execution behavior is fully transparent across time steps.
16. As a compliance auditor, I want contract-level risk CSV reports to include trailing stop counts and preserved profit values, so that policy performance attribution is verifiable.
17. As an MLOps engineer, I want unified CLI flags and defaults for all trend lock and trailing stop parameters, so that experiments and pipelines are fully reproducible.

## Implementation Decisions

### Modular Architecture & Integration Points
- **Offline DP Solver Module**: Augmented to accept macro regime grid arrays and an entry lock toggle. During Bellman backward induction, transitions that would open or flip into counter-trend positions in trending regimes are assigned extreme negative penalties. Neutral exits to zero position remain permitted.
- **High-Level VAE Routing & Execution Module**: Augmented to house the Tier 4 Trailing Profit Stop Engine and Directional Trend Action Mask. Operates after low-level network evaluation and before environment step submission.
- **Common Artifacts & Enumeration Contracts**: Extended with canonical decision reasons:
```python
# Prototype Schema from Architectural Decisions
class ActionDecisionReasons(IntEnum):
    POLICY_INFERENCE = 0
    ACTION_PERSISTENCE = 1
    DEFENSIVE_PREEMPTION = 2
    DEFENSIVE_RULE_CLOSE = 3
    ACTION_UNAVAILABLE_BREAK = 4
    HARD_STOP_LOSS = 5
    STOP_LOSS_COOLDOWN = 6
    CIRCUIT_BREAKER_SUSPENSION = 7
    TREND_ENTRY_LOCK = 8
    TRAILING_PROFIT_STOP = 9
    TRAILING_STOP_COOLDOWN = 10
```
- **CLI & Parameter Registry**: Standardized configuration arguments:
  - `--enable_trend_entry_lock`: default `True`
  - `--enable_trailing_stop`: default `True`
  - `--trailing_stop_activation_threshold`: default `0.08`
  - `--trailing_stop_retracement_ratio`: default `0.25`
  - `--trailing_stop_profit_floor`: default `0.003`
  - `--trailing_stop_cooldown_steps`: default `24`
  - `--trailing_stop_require_peak_breakout`: default `True`

### State Machine for Trailing Profit Stop Engine
The execution router maintains contract-isolated state tracking:
1. `Dormant`: Position is flat or unrealized return has not crossed the activation threshold (+8%).
2. `Tracking`: Unrealized return exceeds the activation threshold. Peak return and peak mark price are updated on each step.
3. `Triggered`: Current return drops from peak return by more than the retracement ratio (25%) while remaining above the profit floor. Action is overridden to flat, action persistence is reset to 0, and state transitions to Cooldown & Hurdle.
4. `Cooldown & Hurdle`: Directional lockout active. Steps count down to zero. Current mark price must exceed peak mark price for same-direction re-entry. Any macro regime change resets the hurdle and cooldown.

## Testing Decisions

### What Makes a Good Test
Tests must verify observable external behavior and contracts rather than internal variable naming:
- Verify that under a simulated bull regime, the DP Q-table gives minimal transition values to short actions when starting from flat, while giving valid values to flat actions when starting from long.
- Verify that the execution router forcibly outputs a flat action when a simulated position experiences a 25% retracement from peak profit, emitting the exact trailing stop reason.
- Verify that within the cooldown window or below the peak price, same-direction actions are blocked and tagged as cooldown, while opposite-direction actions are allowed.
- Verify that changing the regime resets the peak price hurdle.

### Modules Tested
- Dynamic programming Q-table solver and configuration builders.
- High-level VAE risk-aware router execution, persistence, and state transitions.
- Trading diagnostics summary generation and risk metric serialization.

### Prior Art
- Existing DP turnover penalty tests verifying Bellman matrix values under asymmetric regimes.
- Existing stop-loss and circuit-breaker tests verifying action overrides, cooldown counters, and decision reason arrays.

## Out of Scope

- Modifying the feature dimensions or input shapes of the low-level Q-network (preserves existing 148-feature contracts).
- Retraining low-level checkpoints as part of this specification (all changes designed to be forward and backward compatible with existing weights).
- Dynamic Optuna hyperparameter optimization of risk control thresholds (treated as fixed infrastructure guardrails).

## Further Notes

- Fully documented in ADR 0051 (`docs/adr/0051-causal-regime-entry-lock-and-trailing-stop-engine.md`).
- Domain concepts synchronized in `CONTEXT.md` (`单向开仓硬锁`, `因果体制动作掩码`, `移动追踪止盈引擎`, `移动止盈时空双重防再入护栏`, `前高破位再入`).
