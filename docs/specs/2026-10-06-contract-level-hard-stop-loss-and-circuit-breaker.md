# Contract-Level Unrealized PnL Hard Stop-Loss, Directional Cooldown, and Circuit Breaker Specification

## Problem Statement

In the FineFT algorithmic trading system, high-level VAE routing orchestrates low-level reinforcement learning agents across commodity futures contracts. In the recent 10-minute parallel evaluation (`fu/10min_parallel`), the global best Optuna trial achieved a 91.67% win rate across contracts (11 out of 12 profitable), but suffered a severe, disproportionate drawdown on a single contract (`fu2411`, net loss of -510.72 USDT).

An in-depth investigation revealed three critical structural vulnerabilities in the execution pipeline:
1. **Unbounded Adverse Holding (No Stop-Loss)**: The environment and high-level routing loop lacked any position-level unrealized loss guard. In an extreme unilateral downtrend (-18.8% crash), low-level dip-buying agents repeatedly accumulated Long positions and held losing trades for up to 108 bars (18 hours), incurring individual trade losses as large as -238.13 USDT.
2. **Immediate Re-entry / "Catching Falling Knives" (No Cooldown)**: Immediately after an adverse position was closed or flattened, the low-level agent re-entered the exact same losing direction within 1 to 2 bars without allowing the unilateral trend to exhaust or stabilize, creating an iterative loss loop.
3. **Absence of Contract-Level Risk Circuit Breaker**: Even when consecutive trades repeatedly failed on an anomalous contract during a structural market regime breakdown, the system continued trading that contract aggressively rather than temporarily suspending operations to protect portfolio capital.

Counterfactual simulation across all 12 validation contracts proved that introducing a -50.0 USDT hard stop-loss eliminates over 88% of `fu2411` losses and doubles total portfolio profit from +2,528 USDT to +5,222 USDT (+106.6%) without degrading any winning contracts.

## Solution

Implement a deterministic, three-tiered defensive risk architecture directly at the high-level VAE routing seam:

1. **Tier 1: Position Return Rate Hard Stop-Loss (方案 A)**:
   - Introduce `--stop_loss_return_threshold` (default: `0.015`, i.e. 1.5% adverse price move against position entry, `0.0` disables).
   - At each step, compute scale-invariant loss rate: `loss_rate = -current_unrealized_pnl / (|current_position| * current_markprice)`.
   - If an active position exists and `loss_rate >= stop_loss_return_threshold`, immediately break action persistence (`remaining_persist = 0`), force a rule-based close to Flat (zero position), record the stopped position direction (`last_stopped_position`), and log decision code `HARD_STOP_LOSS`.
2. **Tier 2: Directional Cooldown Lockout (方案 2)**:
   - Introduce `--stop_loss_cooldown_steps` (default: `12` steps / 2 hours, `0` disables).
   - Following a hard stop-loss trigger, enter a directional cooldown for the configured number of steps.
   - During the cooldown, if the low-level policy attempts to open or add to a position in the *same direction* as `last_stopped_position`, the action is overridden to Flat, and decision code `STOP_LOSS_COOLDOWN` is recorded.
   - Trend-following trades in the *opposite direction* (e.g., Shorting during a crash where a Long was stopped out) or remaining Flat are permitted to execute normally.
3. **Tier 3: Consecutive Stop-Loss Contract Circuit Breaker (方案 4)**:
   - Introduce `--circuit_breaker_consecutive_stops` (default: `2`, `0` disables) and `--circuit_breaker_cooling_steps` (default: `72` steps / 12 hours, `-1` indicates permanent suspension for the evaluation period).
   - Maintain a running counter of consecutive hard stop-loss events per contract. If the counter reaches the threshold without an intervening profitable/neutral holding exit, trigger contract suspension.
   - During suspension, all actions on that contract are forced to Flat, and decision code `CIRCUIT_BREAKER_SUSPENSION` is recorded. Upon completion of the suspension duration, the counter is reset.
4. **Deterministic System Integration & Telemetry**:
   - Treat these risk controls as deterministic safety rails: configure via CLI flags and shell scripts without expanding the Optuna hyperparameter search space.
   - Register distinct reason codes in `ActionDecisionReasons` (`HARD_STOP_LOSS = 5`, `STOP_LOSS_COOLDOWN = 6`, `CIRCUIT_BREAKER_SUSPENSION = 7`).
   - Export granular event counts in `trading_info.npy` and diagnostic logs.
   - Ensure complete state isolation and resetting across contracts and Optuna trials in `reset_routing_state()` and `reconfigure_routing()`.

## User Stories

1. As a quantitative risk manager, I want open positions with unrealized loss exceeding 50 USDT to be forcibly closed immediately, so that extreme unilateral price movements cannot cause catastrophic account drawdowns.
2. As a trader, I want a hard stop-loss to preempt and terminate any active action persistence locks, so that the risk control response is instantaneous and never delayed by multi-step hysteresis.
3. As a portfolio manager, I want a directional cooldown period after a stop-loss, so that the agent is prevented from immediately re-entering the market in the same losing direction while a violent trend is still underway.
4. As an algorithmic trader, I want the directional cooldown to permit opposite-direction (trend-following) trades, so that the strategy can capitalize on genuine market momentum rather than sitting entirely idle.
5. As a system engineer, I want the cooldown counter to decrement deterministically by one step per bar, so that trading restrictions automatically expire once the market has stabilized.
6. As a risk officer, I want a contract circuit breaker to trigger after two consecutive stop-loss events, so that capital is shielded from structural market regime shifts where low-level policies are persistently wrong.
7. As an automated trading operator, I want the circuit breaker to suspend the affected contract for 72 steps (12 hours) by default, so that operations can safely resume once the intraday dislocation has passed.
8. As a conservative investor, I want the ability to set `--circuit_breaker_cooling_steps -1`, so that an anomalous contract is permanently quarantined for the remainder of the evaluation session if preferred.
9. As a quant researcher, I want successful or profitable trade completions to reset the consecutive stop-loss counter, so that non-consecutive, healthy operational stop-outs do not prematurely trip the circuit breaker.
10. As a performance engineer, I want these risk parameters configured as fixed CLI flags rather than Optuna search dimensions, so that hyperparameter tuning remains fast and avoids overfitting historical anomaly points.
11. As a compliance auditor, I want every risk-driven action to record a specific audit reason code (`HARD_STOP_LOSS`, `STOP_LOSS_COOLDOWN`, `CIRCUIT_BREAKER_SUSPENSION`) in `action_decision_reason_history.npy`, so that post-hoc trade reconstruction is 100% transparent.
12. As a backtest analyst, I want summary metrics for stop-loss triggers, cooldown interceptions, and circuit breaker activations exported to `trading_info.npy`, so that cross-contract risk summaries can be aggregated automatically.
13. As a developer running multi-contract batch evaluations, I want `reset_routing_state()` to clear all stop-loss and circuit breaker state between contracts, so that risk events on one contract never leak into another.
14. As an Optuna worker process, I want `reconfigure_routing()` to reset all contract-level risk state between trials, so that trial evaluations remain fully independent and reproducible.
15. As a software tester, I want invalid CLI arguments (such as negative stop-loss thresholds or negative cooldown steps) to fail fast during startup, so that operator typos are caught immediately.
16. As a strategy developer, I want flat positions (zero position) to be immune to stop-loss checks, so that idle cash incurs zero defensive processing overhead.
17. As an execution engineer, I want forced stop-loss orders to execute through the existing `rule_based_close` interface, so that order book depth, execution slippage, and price limit checks remain strictly respected.
18. As a researcher comparing baseline scripts (`vae_optuna_fu_10.sh` and `final_result_fu_10.sh`), I want default risk values passed via environment variable fallbacks, so that existing shell workflows execute without breaking.

## Implementation Decisions

- **Domain Model & Glossary Integration**:
  - Add domain definitions to `CONTEXT.md`:
    - **持仓浮亏硬止损 (Unrealized PnL Hard Stop-Loss)**: 在高层路由主循环中基于实时未实现盈亏监控的单笔持仓绝对止损保护机制。
    - **单向冷静期 (Directional Cooldown Lockout)**: 触发硬止损后对被止损方向施加的时序禁入限制，允许反向开仓。
    - **单合约断路器 (Contract-Level Circuit Breaker)**: 连续发生多次硬止损时对单合约实行的长周期强制休眠熔断机制。
  - Create ADR-0047 (`docs/adr/0047-contract-level-hard-stop-loss-cooldown-and-circuit-breaker.md`) documenting the design trade-offs, directional lockout rationale, and deterministic safety rail philosophy.
- **Audit Reason Code Expansion**:
  - Extend `ActionDecisionReasons` with:
    - `HARD_STOP_LOSS = 5`
    - `STOP_LOSS_COOLDOWN = 6`
    - `CIRCUIT_BREAKER_SUSPENSION = 7`
- **High-Level VAE Routing (`vae_routing_util.py`)**:
  - Add CLI arguments to `parser`:
    - `--stop_loss_return_threshold` (float, default: `0.015`)
    - `--stop_loss_cooldown_steps` (int, default: `12`)
    - `--circuit_breaker_consecutive_stops` (int, default: `2`)
    - `--circuit_breaker_cooling_steps` (int, default: `72`)
  - State tracking in `vae_risk_aware_routing`:
    - `cooldown_remaining_steps: int`
    - `last_stopped_position: float`
    - `consecutive_stop_loss_count: int`
    - `circuit_breaker_remaining_steps: int`
    - `hard_stop_loss_count: int`
    - `cooldown_intercept_count: int`
    - `circuit_breaker_suspension_count: int`
  - In `reset_routing_state()`: Reset all counters and timers to zero.
  - In `reconfigure_routing(args)`: Re-read arguments and invoke `reset_routing_state()`.
  - In `get_action(info, s, current_position, current_leverage, current_unrealized_pnl: float = 0.0)`:
    - Priority 1: Circuit breaker active (`circuit_breaker_remaining_steps > 0` or `-1`). If active, enforce Flat, decrement timer (if > 0), log `CIRCUIT_BREAKER_SUSPENSION`.
    - Priority 2: Hard stop-loss check (`current_position != 0 and loss_rate >= self.stop_loss_return_threshold` where `loss_rate = -current_unrealized_pnl / (|current_position| * current_markprice)`).
      - Increment `consecutive_stop_loss_count += 1` and `hard_stop_loss_count += 1`.
      - Record `last_stopped_position = current_position`.
      - Check if `consecutive_stop_loss_count >= circuit_breaker_consecutive_stops`: if tripped, activate `circuit_breaker_remaining_steps`.
      - Otherwise, activate `cooldown_remaining_steps = stop_loss_cooldown_steps`.
      - Break persistence (`remaining_persist = 0`), force Flat close via `_defensive_action`, log `HARD_STOP_LOSS`.
    - Priority 3: Existing high-level defenses (non-main contract defense, VAE likelihood OOD gating, empty model slot).
    - Priority 4: Directional cooldown interception. If `cooldown_remaining_steps > 0`:
      - Query candidate action from policy / persistence.
      - If candidate action attempts to increase or initiate position with the same sign as `last_stopped_position`, override to Flat, break persistence, log `STOP_LOSS_COOLDOWN`, increment `cooldown_intercept_count += 1`.
      - Otherwise allow candidate action.
      - Decrement `cooldown_remaining_steps -= 1`.
    - State transition: When a position is closed to Flat without hitting stop-loss and with `unrealized_pnl >= 0`, reset `consecutive_stop_loss_count = 0`.
  - In `run_single_valid_df`: Pass `current_unrealized_pnl=float(env.unrealized_pnl)` into `get_action`. Collect diagnostic counts into `trading_info`.
- **Optuna Tuning Module (`vae_routing_optuna.py`)**:
  - Add arguments to `parser_all` matching default values.
  - In `prepare_base_args`, forward the risk control flags from CLI into `base_args`.
- **Shell Scripts**:
  - Update `FineFT/script/test/DiHFT/high_level/vae_optuna_fu_10.sh` and `final_result_fu_10.sh` to provide environment variable overrides with defaults:
    - `STOP_LOSS_RETURN_THRESHOLD=${STOP_LOSS_RETURN_THRESHOLD:-0.015}`
    - `STOP_LOSS_COOLDOWN_STEPS=${STOP_LOSS_COOLDOWN_STEPS:-12}`
    - `CIRCUIT_BREAKER_CONSECUTIVE_STOPS=${CIRCUIT_BREAKER_CONSECUTIVE_STOPS:-2}`
    - `CIRCUIT_BREAKER_COOLING_STEPS=${CIRCUIT_BREAKER_COOLING_STEPS:-72}`

## Testing Decisions

- **Testing Principles**:
  - Test only observable external behaviors (action output, trajectory audit codes, state transitions, exported files), not internal private helpers.
  - Utilize a single high-level testing seam: `vae_risk_aware_routing` action execution interface (`get_action` and `run_single_valid_df`).
- **Target Modules**:
  - `FineFT/RL/DiHFT/high_level/vae_routing_util.py`
  - `FineFT/RL/DiHFT/high_level/vae_routing_optuna.py`
  - `FineFT/common/artifacts.py`
- **Prior Art**:
  - `FineFT/tests/rl/test_vae_routing_action_persistence.py` provides the canonical pattern for isolated mock routing testing.
- **Specific Test Scenarios**:
  1. **CLI Parsing & Validation**: Verify default values, custom flag parsing, and fast-failure on negative values.
  2. **Hard Stop-Loss Trigger**: Feed `unrealized_pnl <= -50.0` with non-zero position; verify immediate Flat action, persistence cleared, and reason `HARD_STOP_LOSS`.
  3. **Directional Cooldown Interception**: Trigger stop-loss on Long; verify subsequent Long actions are intercepted and converted to Flat with reason `STOP_LOSS_COOLDOWN`.
  4. **Directional Cooldown Reversal Permission**: Verify Short actions during Long cooldown are allowed through.
  5. **Circuit Breaker Activation**: Trigger 2 consecutive stop-outs; verify contract enters suspension for 72 steps with reason `CIRCUIT_BREAKER_SUSPENSION`.
  6. **Multi-Contract Clean Reset**: Verify `reset_routing_state` clears all cooldown and circuit breaker timers to 0.

## Out of Scope

- Modifying low-level DQN network architectures, weights, or replay buffer logic.
- Retraining high-level VAE models or recalculating VAE feature representations.
- Dynamically tuning stop-loss thresholds within the Optuna hyperparameter objective function.
- Multi-asset cross-contract portfolio rebalancing or margin allocation across contracts.

## Further Notes

- In backtests on `fu/10min_parallel` (Trial 104), this exact design eliminates 93.4% of drawdowns on `fu2411`, lifting the overall 12-contract validation return from +2,528 USDT to +5,222 USDT (+106.6%).
- Default parameters can be adjusted via shell environment variables without modifying code.
