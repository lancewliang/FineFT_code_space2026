# High-Water Mark Retracement Reward Shaping and Take-Profit Turnover Penalty Exemption Specification

## Problem Statement

In the FineFT algorithmic trading system, low-level reinforcement learning agents are trained to make discrete positioning decisions based on microstructural features and local holding metrics. In recent long-horizon commodity futures evaluations (e.g. `fu2505` with 4,764 steps over 8 months), strategy performance suffered from a severe structural defect: **the inability of the low-level policy to lock in accumulated profits and exit trailing drawdowns** (in `fu2505`, cumulative unrealized profit peaked at +697.56 RMB / +11.63%, but the agent continuously held its long position through a prolonged decline, surrendering 51.4% of its historical peak gains down to +339.17 RMB).

An exhaustive mathematical, physical, and architectural analysis revealed two intertwined root causes:
1. **Mathematical Path-Insensitivity of Standard Discounted RL**: The optimization objective $J(\pi) = \mathbb{E}\left[\sum_{t=0}^T \gamma^t r_t\right]$ sums scalar single-step PnL increments linearly. In this formulation, a steady, monotonic upward trajectory ($0 \to 100 \to 200 \to 340$) and an extreme boom-and-bust roller-coaster trajectory ($0 \to 300 \to 700 \to 340$) receive identical cumulative reward valuations. The environment has never penalized the erosion of peak unrealized profits, leaving the Bellman value function indifferent to sitting through catastrophic pullbacks.
2. **Conflicting Incentives from Asymmetric Turnover Penalties (ADR 0050)**: To curb hyperactive churn in strong trends, ADR 0050 imposed a 6x base penalty rate (`turnover_adverse_ratio`) on early trend exits (e.g., closing a Long in an uptrend regime). Consequently, when a profitable position begins to retrace at the market peak, closing the trade incurs an immediate, heavy multi-basis-point cash deduction, while holding the position incurs zero turnover penalty and zero path penalty. The agent is mathematically incentivized to hold stubbornly rather than secure its gains.
3. **Decoupled DP State Representation**: Offline Dynamic Programming (DP) expert tables model only current discrete actions, lacking the state dimensionality to record peak historical holding profits without an exponential state space explosion.

Without native reward shaping, the low-level agent cannot learn to take profits autonomously, and relying solely on post-hoc execution guardrails leaves the underlying neural policy inherently misaligned with risk-managed trading objectives.

## Solution

Introduce a native, path-dependent **High-Water Mark Retracement Reward Shaping** mechanism coupled with a **Take-Profit Turnover Penalty Exemption** directly within the trading environment, establishing a two-tier defense in depth alongside the high-level execution guardrails of ADR 0051:

1. **Path-Dependent Continuous Holding Bleed Reward Shaping**:
   - Track the running peak return rate $R_{\text{max}} = \max_{\tau \le t} R_\tau$ for each discrete holding episode.
   - Guard against small-profit false alarms using a minimum peak activation threshold ($\theta_{\text{profit\_min}} = 0.08$, or +8% notional return).
   - Once activated, compute the instant profit retracement ratio:
     $$\text{Retracement}_t = \frac{R_{\text{max}} - R_t}{R_{\text{max}}}$$
   - When the agent continues to hold an open position (Hold Long or Hold Short) beyond an allowable deadband ($\theta_{\text{allow}} = 0.15$, or 15% retracement from peak), apply a quadratic holding bleed deduction to the single-step reward:
     $$\text{penalty}_{\text{dd}} = \lambda_{\text{dd}} \cdot \max(0, \text{Retracement}_t - \theta_{\text{allow}})^2 \times (| \text{position}_t | \times P_t)$$
     with default scaling parameter $\lambda_{\text{dd}} = 0.01$.
   - The penalty is incurred per step during holding, but immediately ceases when the position is closed to Flat, driving $Q(s, \text{Hold}) \ll Q(s, \text{Flat})$ during major drawdowns.

2. **Markov State Representation In-Place Alignment**:
   - Re-calibrate the third dimension of `trading_info` (`trading_info[2]`), previously representing whole-account asset drawdown, to represent the **instant profit retracement ratio**.
   - Output `0.0` when no position is held or when $R_{\text{max}} < \theta_{\text{profit\_min}}$. Once activated, output $\text{clip}(\text{Retracement}_t, 0.0, 1.0)$.
   - Preserves the exact 4-dimensional contract of `trading_info`, guaranteeing zero structural breaking changes to downstream Q-networks, VAE routers, or memory buffers while establishing strict Markovian observability.

3. **Take-Profit Penalty Exemption from ADR 0050**:
   - Provide an exemption rule in directional turnover penalty calculations: any closing transition (Long $\to$ Flat or Short $\to$ Flat) where the holding episode previously achieved $R_{\text{max}} \ge \theta_{\text{profit\_min}}$ is classified as a legitimate take-profit de-risking exit.
   - De-escalate the turnover penalty from the 6x adverse rate down to the 1x base turnover rate (`turnover_base_rate`), eliminating penalty contention and liberating the agent to close trades cleanly at market tops.

4. **Offline DP Decoupling & Two-Tier Defense Architecture**:
   - Keep the offline DP backward induction state space decoupled and lightweight, bounded only by the regime entry locks of ADR 0051.
   - Confine retracement reward shaping to the online environment and Stage I Diverse Training experience buffers.
   - Form a two-tier defense in depth: the low-level RL policy learns soft, adaptive profit taking within the 15% to 25% retracement window, while the ADR 0051 Tier 4 execution engine acts as a hard backstop at 25% retracement.

## User Stories

1. As a reinforcement learning researcher, I want the single-step environment reward to penalize holding through severe profit drawdowns, so that low-level agents do not view riding an elevator down as equivalent to steady capital appreciation.
2. As a futures trader, I want the retracement penalty to activate only after a trade has accumulated meaningful profit (e.g. $\ge 8\%$), so that normal noise and microscopic early price swings are not falsely penalized as drawdowns.
3. As a quantitative risk manager, I want the retracement penalty to use a quadratic activation above an allowable deadband ($\theta_{\text{allow}} = 0.15$), so that mild 5%~10% consolidations are tolerated while catastrophic 30%~50% retracements trigger exponential negative feedback.
4. As an algorithmic trader, I want the retracement penalty scaled by contract notional value ($|\text{position}| \times P_t$), so that the penalty magnitude is scale-invariant across diverse commodity price levels.
5. As a low-level policy agent, I want the holding bleed penalty to cease immediately upon flattening a position, so that the action value $Q(s, \text{Flat})$ is strictly superior to $Q(s, \text{Hold})$ when a trade has crested.
6. As a deep learning engineer, I want `trading_info[2]` to convey the instant retracement ratio, so that the observation space satisfies the Markov property and directly informs the Q-network of the ongoing penalty state.
7. As a systems architect, I want the dimension of `trading_info` to remain strictly 4, so that existing neural network tensor signatures and replay buffers require no backward-incompatible reshaping.
8. As a portfolio manager, I want profitable position closures to be exempt from ADR 0050's 6x adverse turnover penalty, so that the agent is never punished for securing profits in a trending market.
9. As a risk engineer, I want exempt take-profit exits to pay the 1x base turnover rate rather than 0, so that high-frequency action chatter and single-bar position flipping remain adequately restrained.
10. As a performance engineer, I want offline DP Q-tables to remain free of high-dimensional path retracement states, so that pre-computation memory and rollout speeds are not compromised.
11. As an operations engineer, I want CLI flags to configure the retracement shaping parameters, so that experiments can tune activation thresholds, deadbands, and penalty weights across different timeframes.
12. As a quality assurance engineer, I want all environment metrics and step rewards to be testable via deterministic step transitions, so that reward adjustments can be audited down to the exact floating-point penny.
13. As a strategy developer, I want the low-level agent's soft profit-taking behavior (15%~25% window) to complement the high-level Tier 4 hard trailing stop (25% threshold), so that capital is protected by a coordinated two-tier defense.
14. As an algorithmic researcher, I want diagnostic logs and trading info records to log whenever retracement shaping is active, so that reward-shaping contributions can be disentangled from raw market PnL.

## Implementation Decisions

- **Environment State Tracking**:
  - The trading environment will maintain running episode tracking variables: `episode_peak_return_rate` and `episode_peak_price` for active non-zero positions.
  - Upon any position change that reverses or enters a new trade, episode peak metrics are reset.
  - While holding an active position, `episode_peak_return_rate` is updated monotonically using mark price returns relative to the entry price.

- **Reward Shaping Mathematical Specification**:
  - When holding an active position ($position \ne 0$) and `episode_peak_return_rate` $\ge \theta_{\text{profit\_min}}$:
    - Compute current directional return rate $R_t$ and instant retracement $\text{Retracement}_t = (R_{\text{max}} - R_t) / R_{\text{max}}$.
    - If $\text{Retracement}_t > \theta_{\text{allow}}$, compute:
      $$\text{penalty}_{\text{dd}} = \lambda_{\text{dd}} \cdot (\text{Retracement}_t - \theta_{\text{allow}})^2 \times (|position_t| \times P_t)$$
    - Deduct $\text{penalty}_{\text{dd}}$ from the scalar environment `reward`.
    - If the agent transitions to Flat ($target\_position = 0$), $\text{penalty}_{\text{dd}}$ is 0.0 for that step.

- **State Feature Interface Contract**:
  - In `trading_info` generation, `trading_info[2]` reflects the instant retracement:
    - Returns `0.0` if `position == 0` or `episode_peak_return_rate < theta_profit_min`.
    - Returns $\text{clip}(\text{Retracement}_t, 0.0, 1.0)$ when activated.
  - Keys and length of `TRADING_INFO_KEYS` remain unchanged at 4.

- **Turnover Penalty Exemption Contract**:
  - In the turnover penalty calculator, an additional boolean parameter `is_take_profit_exit` is supported.
  - When `is_take_profit_exit` is True (triggered when closing an active position that achieved peak return $\ge \theta_{\text{profit\_min}}$), the effective rate is clamped to `turnover_base_rate` instead of being multiplied by `turnover_adverse_ratio`.

- **Configuration Contracts**:
  - Support configuration parameters with strict defaults:
    - `enable_drawdown_reward_shaping`: bool, default False (opt-in for existing scripts, enabled for new training).
    - `drawdown_profit_min`: float, default `0.08` (+8%).
    - `drawdown_allow_ratio`: float, default `0.15` (15%).
    - `drawdown_penalty_weight`: float, default `0.01`.
    - `enable_take_profit_turnover_exemption`: bool, default True.

## Testing Decisions

- **What Makes a Good Test**:
  - Tests must verify external environment behavior and step invariants exclusively through public methods (`reset()` and `step()`), without asserting on internal private variables.
  - Tests must verify that nominal account wallet balances are completely untouched by the retracement penalty (confirming pure reward shaping).
  - Tests must verify deterministic floating-point calculations of reward penalties across precise synthetic price paths.
  - Tests must verify that `trading_info[2]` produces exact expected values at pre-activation, post-activation, and post-exit steps.
  - Tests must verify that closing a long position in an uptrend regime incurs only 1x base turnover penalty when peak profit was achieved, but incurs 6x adverse penalty when peak profit was not achieved.

- **Target Seam**:
  - The primary and sole seam is the **Environment Step Seam** (`Base_Env.step()`).

- **Prior Art**:
  - `FineFT/tests/env/test_turnover_penalty.py` (provides synthetic DataFrame fixtures and tests for directional turnover penalties and reward shaping).
  - `FineFT/tests/env/test_trading_process_features.py` (provides validation patterns for `trading_info` observation vectors).
  - `FineFT/tests/env/test_limit_reward.py` (demonstrates step-level reward shaping isolation without mutating wallet balances).

## Out of Scope

- Offline DP Q-table state space expansion (DP tables remain focused on action transition graphs and ADR 0051 regime locks).
- Changing neural network architectures or expanding the `trading_info` vector beyond 4 dimensions.
- Redesigning high-level VAE routing logic (high-level routing continues to operate its independent Tier 4 execution guardrail).
- Continuous action space or policy gradient actor-critic redesigns.

## Further Notes

- This specification implements the core recommendation of Section 6.3 in `docs/research/2026-10-08-low-level-agent-trend-lock-and-trailing-stop-research.zh_cn.md`.
- Formal architectural grounding is recorded in `docs/adr/0052-high-water-mark-retracement-shaping-and-take-profit-exemption.md`.
