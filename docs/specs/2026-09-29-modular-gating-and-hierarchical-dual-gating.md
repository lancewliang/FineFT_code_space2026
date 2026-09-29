# Spec: Modular Gating Architecture and Hierarchical Dual Gating Strategy

## Problem Statement

In the FineFT hierarchical reinforcement learning framework for commodity futures trading, the high-level VAE meta-router uses dual-axis (trend slope and volatility) Variational Autoencoder reconstruction likelihood quantiles to gate trading decisions. If market confidence is deemed insufficient, the router triggers a defensive flat-position action to protect capital.

However, in real-world evaluations across out-of-sample forward contracts, the current gating mechanism suffers from severe structural flaws:
1. **Severe Return Dilution from 98%+ Cash Paralysis**: On forward commodity contracts (such as 10-minute fuel oil `fu2509`), the strategy remains in cash defense over 98% to 99.6% of the backtest. Over an entire year of market data, active trading occurs for only 20 to 200 bars. Although traded win rates are high, the strategy total return is heavily diluted down to 0.15% ~ 0.50%, rendering the RL policy ineffective in practice.
2. **False OOD Alarms from Inter-Year Distribution Shift**: The VAE likelihood baseline is fixed to historical training contracts from 1 to 2 years prior. Natural macroeconomic drift, basis shifts, and volatility regime evolutions cause test sample reconstruction likelihoods to systematically shift downward. The model mistakes healthy market trends with minor temporal drift for catastrophic tail-risk anomalies.
3. **Coupled Risk Objectives in a Single Hard Threshold**: The legacy system uses a single absolute quantile cutoff (e.g., 0.20 to 0.48) against unnormalized weights. This conflates two completely distinct risk concepts:
   - *Epistemic Uncertainty (Black Swan OOD)*: The market is in an unprecedented state where all models fail.
   - *Aleatoric Uncertainty (Directionless Chop)*: The market is in a known regime, but signals are noisy and directionless.
4. **Lack of Strategy Modularity**: The gating logic is hardcoded as an inline `if` statement inside `vae_routing_util.py`. Researchers cannot benchmark or switch between gating strategies without manually modifying core runtime files.

## Solution

Design and implement a clean, decoupled **Modular Gating Architecture** with two fully isolated, interchangeable gating strategies:

1. **`AbsoluteThresholdGating` (`absolute`)**:
   - The preserved legacy baseline strategy, ensuring 100% bitwise reproducibility and backward compatibility for all existing scripts, benchmarks, and papers.
   - Compares raw rolling VAE likelihood quantiles directly against absolute threshold parameters (`slope_rule_base_threshold`, `volatility_rule_base_threshold`).

2. **`HierarchicalDualGating` (`hierarchical`)**:
   - A new hierarchical defense strategy that cleanly decouples tail-risk circuit breaking from chop-noise filtering:
     - **Gate 1 (Tail-Risk Circuit Breaker)**: An extreme lower-bound safeguard (`ood_threshold`, e.g., 0.005) that triggers defensive cash posture only when the market suffers genuine catastrophic OOD collapse where all models fail.
     - **Normalization**: Computes relative mode probability distributions via parameter-free linear normalization ($p_i = w_i / \sum w_j$).
     - **Gate 2 (Relative Margin Gating)**: Evaluates mode clarity by checking the separation between the dominant regime and the runner-up ($\Delta p = p_{(1)} - p_{(2)}$). If $\Delta p < 	ext{margin\_threshold}$, the strategy stays flat to avoid whipsaw losses during directionless consolidation.
     - **Execution**: When both gates pass, routes directly to the specialized sub-agent for the dominant regime.

3. **Pluggable Factory and Seamless Switching**:
   - A new package `FineFT/RL/DiHFT/high_level/gating/` providing a uniform `BaseGatingStrategy` interface and `GatingDecision` data contract.
   - Dynamic strategy selection via `--gating_strategy absolute|hierarchical` across CLI, Optuna hyperparameter optimization, and final result evaluation.
   - Self-describing parameter serialization in `high_level_agent_para.txt` with automatic format detection.

## User Stories

1. As a quant researcher, I want the system to provide two independent gating strategies (`absolute` and `hierarchical`), so that I can systematically compare legacy quantile gating against dual-layer hierarchical gating on the exact same market data.
2. As a strategy developer, I want the old gating strategy to remain fully intact and selectable via `--gating_strategy absolute`, so that existing research benchmarks and historical experiment results can be reproduced with zero regression.
3. As an algorithmic trader, I want the new `HierarchicalDualGating` strategy to prevent cash-lock paralysis by distinguishing normal temporal distribution drift from genuine market crises, so that the RL agent can actively trade clear market trends and achieve realistic, competitive annual returns.
4. As a risk manager, I want Gate 1 of `HierarchicalDualGating` to enforce an absolute tail-risk circuit breaker (`ood_threshold`), so that the system immediately retreats to cash when an unprecedented black swan event renders all VAE models untrustworthy.
5. As a portfolio manager, I want Gate 2 of `HierarchicalDualGating` to enforce a relative mode clarity threshold (`margin_threshold`), so that the agent avoids opening positions in choppy, sideways markets where no clear trend or volatility regime is established.
6. As a reinforcement learning engineer, I want the strategy implementations to reside in isolated, dedicated Python modules (`absolute_gating.py` and `hierarchical_gating.py`), so that modifying or extending one strategy cannot inadvertently introduce regressions into the other.
7. As a systems architect, I want both strategies to adhere to a common `BaseGatingStrategy` abstract interface and return a standardized `GatingDecision` object, so that the main routing engine (`vae_risk_aware_routing`) remains completely agnostic to specific gating rules.
8. As a backtest analyst, I want `GatingDecision` to record explicit, structured reject reasons (`none`, `absolute_threshold`, `ood_circuit_breaker`, `margin_ambiguity`), so that I can run diagnostic attribution on backtest logs to determine exactly why each flat position was taken.
9. As a quant researcher, I want the high-level macro action history (`macro_action_history.npy`) to retain its existing numerical contract (`slot_count` for defensive action), so that downstream metrics calculation, plotting scripts, and artifact analyzers continue to function without modifications.
10. As a strategy researcher, I want Optuna hyperparameter tuning (`vae_routing_optuna.py`) to automatically adapt its search space based on `--gating_strategy`, tuning absolute thresholds for `absolute` and tuning `ood_threshold` / dual-axis `margin_threshold` for `hierarchical`.
11. As an MLOps engineer, I want `resolve_routing_parameters` to automatically detect the gating strategy from parameter string formats in `high_level_agent_para.txt`, so that final evaluation scripts load the correct strategy without requiring manual flag coordination.
12. As a developer, I want default CLI parameters to fall back to `absolute` when `--gating_strategy` is omitted, so that legacy scripts without the flag continue to execute their expected baseline behavior.
13. As a test engineer, I want comprehensive unit tests covering both gating strategies in isolation with deterministic mock weights, so that edge cases (such as exact boundary equality, zero sum weights, and extreme OOD inputs) are verified independently of neural network weights.
14. As a pipeline developer, I want end-to-end integration tests verifying that `vae_risk_aware_routing` correctly instantiates, switches, and reconfigures both gating strategies during multi-trial runs, so that runtime reconfigurability is guaranteed.
15. As a futures trader, I want the strategy to support independent margin thresholds for slope and volatility axes (`slope_margin_threshold` and `volatility_margin_threshold`), so that I can configure stricter directional trend requirements while allowing more flexible volatility tolerance.

## Implementation Decisions

- **Dedicated Gating Submodule**:
  Create `FineFT/RL/DiHFT/high_level/gating/` containing:
  - `base.py`: Defines `BaseGatingStrategy` and `GatingDecision`.
  - `absolute_gating.py`: Encapsulates `AbsoluteThresholdGating`.
  - `hierarchical_gating.py`: Encapsulates `HierarchicalDualGating`.
  - `factory.py`: Implements `create_gating_strategy(strategy_type, **kwargs)`.
  - `__init__.py`: Exports the public API.

- **Data Contract (`GatingDecision`)**:
  A dataclass with the following signature:
  ```python
  @dataclass
  class GatingDecision:
      is_defensive: bool
      volatility_index: int  # 0 to num_labels - 1, or -1 if defensive
      slope_index: int       # 0 to num_labels - 1, or -1 if defensive
      reject_reason: str     # "none", "absolute_threshold", "ood_circuit_breaker", "margin_ambiguity"
      metrics: dict          # Diagnostic data: normalized probabilities, margins, raw max weights
  ```

- **Separation of Responsibilities**:
  - `vae_risk_aware_routing` retains ownership of: feature tensor preparation, VAE model execution, rolling deque buffers, EMA weight calculations, and environment order placement.
  - The gating strategy acts as a pure decision engine: it receives computed axis weights (`volatility_weights`, `slope_weights`) and returns a `GatingDecision`.

- **Linear Normalization for Hierarchical Gating**:
  - Gate 1 checks raw max rolling weights against `ood_threshold` (default 0.005).
  - Relative probabilities are computed via linear normalization:
    $$p_i = \frac{w_i}{\sum_j w_j + 10^{-12}}$$
  - Gate 2 checks top-1 vs top-2 margin:
    $$\Delta p_\text{vol} = p_{\text{vol},(1)} - p_{\text{vol},(2)}, \quad \Delta p_\text{slope} = p_{\text{slope},(1)} - p_{\text{slope},(2)}$$
    If $\Delta p_\text{vol} < \text{volatility\_margin\_threshold}$ or $\Delta p_\text{slope} < \text{slope\_margin\_threshold}$, triggers defensive rejection (`margin_ambiguity`).

- **CLI & Parameter Registry**:
  Add `--gating_strategy` (`choices=["absolute", "hierarchical"]`, `default="absolute"`).
  Add `--ood_threshold` (`default=0.005`).
  Add `--slope_margin_threshold` (`default=0.12`).
  Add `--volatility_margin_threshold` (`default=0.12`).
  Update centralized parameter column constants in `common/routing_params.py`.

- **Parameter Serialization Format**:
  - Legacy: `trial_0_ws_74_wv_98_gs_0.9368_gv_0.9273_ts_0.2195_tv_0.3030`
  - Hierarchical: `strat_hierarchical_trial_0_ws_74_wv_98_gs_0.9368_gv_0.9273_ood_0.0050_ms_0.1200_mv_0.1200`
  - `resolve_routing_parameters` inspects the parameter line: if `strat_hierarchical` or `ood_` is present, it auto-configures `gating_strategy="hierarchical"`; otherwise defaults to `"absolute"`.

## Testing Decisions

- **Testing Philosophy**:
  Test observable external behavior rather than internal private attributes. Verify that given identical inputs, the system produces the exact expected actions, routing slots, and attribution reasons across both strategies.

- **Primary Seam 1 (Routing Engine Level)**:
  `vae_risk_aware_routing.get_action(info, s, current_position, current_leverage)`
  - Verify that `--gating_strategy absolute` reproduces legacy behavior.
  - Verify that `--gating_strategy hierarchical` correctly branches to `_defensive_action` under Gate 1 (OOD) and Gate 2 (Margin), and routes to the correct `slot_id` when both gates pass.

- **Primary Seam 2 (Isolated Strategy Level)**:
  `BaseGatingStrategy.decide(volatility_weights, slope_weights)`
  - Unit test `AbsoluteThresholdGating` on all boundary conditions.
  - Unit test `HierarchicalDualGating` on Gate 1 triggering, Gate 2 triggering, and clean passage.
  - Unit test handling of all-zero weights and degenerate distributions.

- **Primary Seam 3 (Parameter & Factory Level)**:
  - Test `create_gating_strategy` factory validation and unknown strategy rejection.
  - Test `resolve_routing_parameters` bi-directional auto-detection from text files and CLI args.
  - Test Optuna parameter suggestion isolation between strategies.

- **Prior Art**:
  Follow patterns in `FineFT/tests/rl/test_vae_routing_non_main_defense.py` and `FineFT/tests/rl/test_vae_routing_final_result.py` using mock routers and synthetic payloads.

## Out of Scope

- Modifying the underlying VAE architecture or retraining VAE neural network weights.
- Modifying the low-level DQN agent architectures or Q-network weights.
- Altering the non-main contract defense (`enable_non_main_contract_defense`) pre-filter.
- Re-running the full 50-trial Optuna optimization run within this task (only the infrastructure, strategies, and test suites are built and verified).

## Further Notes

- The default strategy remains `absolute`, guaranteeing zero breaking changes to existing production bash scripts.
- To execute the new strategy on fuel oil 10-minute data, callers simply pass `--gating_strategy hierarchical --ood_threshold 0.005 --slope_margin_threshold 0.12 --volatility_margin_threshold 0.12`.
