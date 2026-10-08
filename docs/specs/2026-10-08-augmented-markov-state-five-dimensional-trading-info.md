# Augmented Markov State Five-Dimensional Trading Process Feature Specification

## Problem Statement

In the FineFT algorithmic trading system, low-level reinforcement learning agents utilize a memoryless feedforward multi-layer perceptron (MLP) architecture. Because feedforward networks lack internal recurrent states or causal memory caches, their policy $\pi(a | s)$ and value function $Q(s, a)$ depend strictly on the instantaneous observation vector $s$.

In long-horizon trend trading, this memoryless architecture creates a severe Partially Observable Markov Decision Process (POMDP) defect:
1. **Inability to Distinguish Trajectory Contexts**: The policy network cannot distinguish between an early breakout trajectory (e.g., entering at 100 and rising to 110, where the agent should firmly hold or add to its position) and a severe post-peak retracement trajectory (e.g., entering at 100, peaking at 150, and collapsing back to 110, where the agent must urgently exit to secure accumulated profits). Both scenarios present an identical instantaneous unrealized return ($+10\%$) and identical local orderbook microstructures.
2. **Bellman Credit Assignment Breakdown Under Reward Shaping**: When ADR 0052 introduced path-dependent drawdown penalties (holding bleed deductions for severe retracements from peak gains), the lack of explicit peak history in the observation space caused conflicting gradients. The Bellman update pushed the Q-value of holding upwards for early breakouts while dragging it downwards for post-peak retracements on identical state inputs, causing policy instability, premature panic exits, or failure to respect drawdown penalties.
3. **Information Loss in Four-Dimensional In-Place Reuse**: Prior attempts to maintain a 4-dimensional vector by replacing total account drawdown with an instant retracement ratio failed to provide the explicit peak return magnitude $R_{\text{max}}$. Without $R_{\text{max}}$, the network cannot differentiate between a negligible 1% profit retracing by 25% (normal market micro-noise) and a 30% windfall profit retracing by 25% (major trend reversal requiring decisive profit-taking).

To achieve optimal, autonomous trade management without relying on external heuristic overrides, the reinforcement learning state space must be augmented into a strictly first-order Markovian representation.

## Solution

Upgrade the trading process feature contract (`trading_info`) into a fully decoupled, five-dimensional **Augmented Markov State** vector that provides complete, first-order observability of the trade lifecycle:

1. **Five-Dimensional Trading Process Feature Contract**:
   Explicitly expand `trading_info` from 4 to 5 dimensions:
   - Index 0: `position_exposure` (normalized position allocation in $[-1.0, 1.0]$)
   - Index 1: `single_holding_return_rate` (instantaneous underlying mark price percentage return $R_t$)
   - Index 2: `peak_return_rate` (historical peak mark price return achieved during the trade $R_{\text{max}}$)
   - Index 3: `instant_profit_retracement` (instant retracement ratio from peak $\text{Retracement}_t$)
   - Index 4: `current_holding_duration_norm` (normalized duration $\tau_t \in [0.0, 1.0]$)

2. **Unified Scale-Invariant Price Return Metric**:
   Align the physical measurement of both $R_t$ and $R_{\text{max}}$ to the underlying mark price percentage return:
   $$R_t = \frac{P_t - P_{\text{entry}}}{P_{\text{entry}}} \times \text{sign}(\text{position})$$
   $$R_{\text{max}} = \text{clip}\left(\max\left(0.0, \max_{\tau \le t} R_\tau\right), 0.0, 1.0\right)$$
   This eliminates non-stationary scaling artifacts caused by disparate margin ratios across commodity futures contracts, providing scale-invariance and numeric stability within $[0.0, 1.0]$.

3. **Dead-Zone Gated and Hard-Saturated Retracement Computation**:
   Suppress micro-structural noise by applying a dead-band threshold ($\theta_{\text{profit\_min}} = 0.08$):
   $$\text{Retracement}_t = \begin{cases} 0.0, & \text{if } R_{\text{max}} < \theta_{\text{profit\_min}} \\ \text{clip}\left(\frac{R_{\text{max}} - R_t}{\max(R_{\text{max}}, 10^{-12})}, 0.0, 1.0\right), & \text{if } R_{\text{max}} \ge \theta_{\text{profit\_min}} \end{cases}$$
   When a trade crosses into negative territory ($R_t < 0$), the retracement metric saturates cleanly at $1.0$, while the severity of the capital loss is independently conveyed by the negative value of $R_t$.

4. **Complete Markovian Alignment with Drawdown Reward Shaping**:
   The augmented state features directly observe the exact quantities used by ADR 0052 to compute holding bleed penalties. This transforms the path-dependent reward formulation into an exact first-order MDP, ensuring consistent TD target updates and enabling the network to learn $Q(s, \text{Flat}) \gg Q(s, \text{Hold})$ whenever severe retracements are observed.

5. **Strict Hierarchical Decoupling**:
   In strict adherence to the Triple-Stream Feature Decoupling Architecture (ADR 0037/0041), the low-level observation space is strictly restricted to internal trading process state features. No high-level VAE latent vectors or discrete regime identifiers are permitted into the low-level Q-network.

## User Stories

1. As a reinforcement learning researcher, I want the agent's observation space to contain the running peak return rate $R_{\text{max}}$, so that the policy can distinguish between high-conviction winning trades and marginal breakeven positions.
2. As a reinforcement learning researcher, I want the observation space to contain both instantaneous return and retracement depth as separate linear features, so that the network's first linear layer can easily compute profit-at-risk representations.
3. As a quantitative portfolio manager, I want the trade return metrics to be evaluated against the underlying mark price rather than leveraged margin, so that the policy behaves consistently across contracts with differing margin requirements.
4. As an algorithmic trader, I want peak return rate values to be physically clamped to $[0.0, 1.0]$, so that black swan moves do not produce out-of-distribution activation spikes in the neural network.
5. As an execution engineer, I want the instant profit retracement ratio to output zero when peak return is below 8%, so that the agent is not distracted by normal intraday price oscillations during trade incubation.
6. As a risk manager, I want the instant profit retracement to clamp at 1.0 when a trade turns negative, so that the retracement metric remains strictly bounded while the negative return metric captures the loss.
7. As a deep learning engineer, I want `trading_info` to zero out completely when the position is flat, so that idle states present a clean, deterministic baseline to the network.
8. As a deep learning engineer, I want reversing a position in a single step to immediately reset peak return, retracement, and return metrics to zero, so that historical performance from an opposing position does not contaminate the new trade.
9. As a reinforcement learning engineer, I want the environment's drawdown reward penalty to share identical mathematical definitions with the augmented state features, so that policy gradient signals map directly to observable inputs.
10. As a system architect, I want `TRADING_INFO_KEYS` to act as the single source of truth for the dimensionality and ordering of trading features, so that magic numbers are eliminated across the codebase.
11. As a performance engineer, I want the GPU-resident replay buffer to preallocate contiguous tensor memory using `len(TRADING_INFO_KEYS)`, so that transitions are stored and sampled without dynamic reallocation overhead.
12. As a policy developer, I want the low-level Q-network's `fc_trading` layer to accept a 5-dimensional input, so that the full augmented state is encoded into the policy's hidden representations.
13. As a systems auditor, I want any legacy 4-dimensional tensor passed to a 5-dimensional Q-network to fail immediately with a runtime dimension error, adhering to the project's fail-fast guidelines.
14. As a machine learning researcher, I want low-level Q-networks to remain completely decoupled from high-level VAE latent vectors, preventing macro-regime estimation errors from degrading micro-execution decisions.
15. As a backtesting analyst, I want diagnostic logs and step outputs to retain visibility into peak return and instant retracement, allowing granular verification of policy exit behaviors.
16. As a quantitative developer, I want the holding duration index in `trading_info` to be resolved dynamically via named constants, preventing buffer unpacking logic from mistaking retracement for holding duration.
17. As an ML ops engineer, I want pretraining and diverse exploration replay buffers to record the 5-dimensional trading state uniformly, ensuring consistency between offline warmups and online policy updates.
18. As a compliance engineer, I want the state augmentation to introduce zero non-causal lookahead bias, computing all accumulators strictly from past and current price history within the episode.
19. As a test engineer, I want to verify all state augmentation behaviors through public environment step returns, avoiding reliance on private implementation details.
20. As a strategy developer, I want the low-level agent to autonomously converge on taking profits when retracement exceeds 15%, reducing dependence on external rule-based execution overrides.

## Implementation Decisions

### Module Modifications and Responsibilities

- **Trading Environment Core**:
  - The trading environment module will maintain internal state accumulators for single-holding mark price return, running peak return, and instant profit retracement.
  - The trading process feature constant will be defined as an ordered tuple of 5 fields:
    ```python
    TRADING_INFO_KEYS = (
        "position_exposure",
        "single_holding_return_rate",
        "peak_return_rate",
        "instant_profit_retracement",
        "current_holding_duration_norm",
    )
    ```
  - The dimension constant `TRADING_INFO_DIM` will be derived directly from `len(TRADING_INFO_KEYS)`.
  - Holding duration extraction logic will refer to the index of `current_holding_duration_norm` rather than a hardcoded index.
  - Step transitions that result in position closure or position reversal will immediately reset all holding-specific accumulators.

- **Low-Level Policy Architectures**:
  - The low-level Q-network and ensemble Q-network modules will configure their trading feature linear encoder (`fc_trading`) with input dimension `TRADING_INFO_DIM` (5).
  - Calling the forward method with mismatched tensor shapes will fail immediately.

- **Experience Replay Buffers**:
  - Stratified, prioritized, and GPU-resident replay buffers will allocate tensor storage for `trading_info` and `next_trading_info` with a second dimension of 5.
  - Duration-based sampling or weighting calculations will access the duration field at index 4.

- **Diagnostics and Pipeline Utilities**:
  - Shared data managers, diagnostics manifests, and evaluation harnesses will instantiate networks and synthetic buffers using the 5-dimensional contract.
  - Any utility functions that validate or mock trading process features will generate 5-element arrays.

### Architectural Boundaries

- **Strict Triple-Stream Decoupling**:
  Low-level Q-networks receive only market state features, previous action, time encoding, and the 5-dimensional `trading_info`. High-level VAE latent vectors $z_t$ and discrete regime grid IDs remain strictly outside the network graph.

- **No Backward-Compatibility Baggage**:
  In accordance with project guidelines, no backward-compatibility layers, adaptive dimension detection, or legacy fallback shims will be introduced. Existing 4-dimensional checkpoints are deemed incompatible and must be retrained.

## Testing Decisions

### What Makes a Good Test

Tests must verify external behavioral contracts and observable state transitions, never private methods or ephemeral variables. The primary test seam is the environment's public interface (`env.reset()` and `env.step()`), testing how real sequences of actions and price movements shape the returned `info["trading_info"]` array and single-step reward.

### Modules and Seams Under Test

1. **Primary High Seam: Trading Environment Public Boundary**:
   - Verify `env.reset()` outputs a 5-dimensional zero vector.
   - Verify entering a Long position reflects positive position exposure, initial zero return, initial zero peak return, initial zero retracement, and initial non-zero duration.
   - Verify price appreciation monotonically increases both return and peak return, while retracement remains zero.
   - Verify a pullback from peak while $R_{\text{max}} < 8\%$ preserves a zero retracement observation (dead-zone verification).
   - Verify a pullback from peak when $R_{\text{max}} \ge 8\%$ outputs a continuous retracement ratio in $[0.0, 1.0]$.
   - Verify a catastrophic drop below entry price saturates retracement at 1.0 while return turns negative.
   - Verify closing the position zeroes out all fields in the subsequent step.
   - Verify reversing from Long to Short in a single step resets peak and retracement for the new position.

2. **Secondary Subordinate Seam: Network and Buffer Integration**:
   - Verify low-level Q-network constructs `fc_trading` with input dimension 5 and successfully computes forward passes given shape `(batch, 5)`.
   - Verify passing shape `(batch, 4)` raises a shape mismatch error.
   - Verify replay buffer stores, samples, and un-normalizes duration correctly using index 4.

### Prior Art in the Codebase

- `FineFT/tests/env/test_trading_process_features.py`: Baseline tests for trading process feature shapes, normalization, and position changes.
- `FineFT/tests/env/test_drawdown_reward_shaping.py`: Tests for high-water mark tracking and drawdown penalty reward deductions.
- `FineFT/tests/rl/test_qnet_trading_info.py`: Tests for Q-network integration with trading process feature inputs.
- `FineFT/tests/rl/test_replay_buffer_trading_info.py`: Tests for buffer storage and sampling of trading features.

## Out of Scope

1. **Recurrent or Attention-Based Policy Architectures**:
   Implementing DRQN, LSTM/GRU cells, or GTrXL Transformers for low-level trading is out of scope. The solution strictly resolves the POMDP via state augmentation within feedforward MLPs.
2. **Offline Dynamic Programming State Expansion**:
   Expanding the offline DP table's Bellman recursion state space to include peak return or retracement dimensions is out of scope. DP expert generation remains 1D/2D price-bounded per ADR 0051/0052.
3. **High-Level VAE Routing Modifications**:
   The VAE architecture, macro-regime clustering, and Tier 4 trailing stop execution rules defined in ADR 0051 remain untouched.
4. **Retrofitting Legacy 4-Dimensional Checkpoints**:
   Adapting or fine-tuning existing 4-dimensional model weights without full retraining is out of scope.

## Further Notes

- **Retraining Requirement**: Because `TRADING_INFO_DIM` changes from 4 to 5, the first linear layer `fc_trading` changes from `Linear(4, hidden)` to `Linear(5, hidden)`. All low-level checkpoints must be retrained using Stage I pretraining and diverse exploration.
- **Relationship to Existing ADRs**:
  - Complements **ADR 0051** (Directional Entry Locks and Tier 4 Trailing Stop Execution Guard).
  - Supercharges **ADR 0052** (High-Water Mark Retracement Reward Shaping) by providing the neural network with exact state observability of the reward signals.
  - Implements **ADR 0053** (Augmented Markov State Five-Dimensional Trading Info).
