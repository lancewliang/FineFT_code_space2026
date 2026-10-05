# High-Level VAE Routing Action Persistence with Defensive Preemption Specification

## Problem Statement

In the FineFT algorithmic trading system, low-level reinforcement learning agents (`parallel_diverse_train.py` and `test_agent_index.py`) are trained and evaluated with action persistence (`--action_persistence 3`). This mechanism forces non-flat trading positions (long or short) to be physically held for a minimum of 3 consecutive 10-minute bars, avoiding micro-churning transaction costs and skipping redundant CPU/GPU forward neural network evaluations.

However, the high-level VAE risk-aware routing pipeline (`vae_optuna_fu_10.sh`, `vae_routing_optuna.py`, and `vae_routing_util.py`) currently lacks support for `--action_persistence`. This causes two critical problems:
1. **Train/Tuning/Evaluation Mismatch**: Hyperparameter optimization via Optuna explores high-level gating parameters under the false premise that low-level agents re-evaluate and churn actions at every single 10-minute step (`action_persistence = 1`), diverging from actual low-level execution semantics and degrading out-of-sample real performance.
2. **Post-Hoc Opacity**: Even when persistence is applied, historical evaluation artifacts only save discrete action values (`micro_action_history.npy`). Quantitative researchers and engineers cannot audit after the fact whether an action was held due to autonomous policy inference agreement, action persistence locking, defensive preemption, or market execution constraints.

## Solution

1. Equip the high-level VAE routing infrastructure (`vae_routing_util.py`, `vae_routing_optuna.py`, and `vae_routing_final_result_macro_action.py`) and corresponding execution shell scripts (`vae_optuna_fu_10.sh` and `final_result_fu_10.sh`) with `--action_persistence` configuration defaulting to `3`.
2. Implement **Defensive Preemption over Action Persistence**:
   - When a non-flat action is initiated, low-level policy inference is locked for `action_persistence` steps as long as the action remains valid in `info["avaliable_action"]` and high-level routing remains in non-defensive regimes (including transitions between valid agent slots).
   - High-level risk controls (such as out-of-distribution drift detection, non-main contract defensive gating, or empty model slots) take absolute precedence: triggering defensive action immediately breaks persistence (`remaining_persist = 0`) and executes rule-based close to zero position.
   - Flat position actions (zero position) do not trigger persistence, allowing continuous re-evaluation.
3. Implement a dual-level diagnostic and audit tracking system:
   - Persist a per-step trajectory artifact `action_decision_reason_history.npy` alongside `micro_action_history.npy` using discrete integer codes (`0: policy_inference`, `1: action_persistence`, `2: defensive_preemption`, `3: defensive_rule_close`, `4: action_unavailable_break`).
   - Log and store aggregated diagnostic metrics in `trading_info.npy` (`total_steps`, `inference_steps`, `persistence_held_steps`, `skip_inference_ratio`, `defensive_preemptions`, `action_unavailable_breaks`).

## User Stories

1. As a quantitative researcher running Optuna tuning (`vae_optuna_fu_10.sh`), I want the optimization search to evaluate high-level routing under `--action_persistence 3`, so that optimal hyperparameters reflect actual low-level execution constraints.
2. As a trader evaluating commodity futures, I want non-flat positions to persist across consecutive timesteps during non-defensive market states, so that portfolio turnover and bid-ask slippage fees are minimized.
3. As a risk manager, I want high-level OOD gating to immediately preempt and terminate active position persistence, so that capital is protected the moment the market drifts into abnormal regimes.
4. As a risk manager, I want non-main contract defensive rules to preempt active position persistence, so that liquidity risks on secondary contracts are instantly mitigated by forced flattening.
5. As a high-frequency trading engineer, I want low-level neural network forward inference to be bypassed during persistent steps, so that high-level backtesting and Optuna trials complete significantly faster.
6. As a researcher analyzing strategy logs, I want a per-step `action_decision_reason_history.npy` artifact, so that I can conclusively distinguish between model inference choices and persistence-forced holds for any historical bar.
7. As a system operator, I want summary metrics including `skip_inference_ratio` and `persistence_held_steps` reported in `trading_info.npy` and console logs, so that I can monitor execution efficiency across contracts at a glance.
8. As a developer running final out-of-sample tests (`final_result_fu_10.sh`), I want `--action_persistence` to be inherited and configurable via environment variables, so that final results align with Optuna tuning conditions.
9. As a data analyst inspecting macro allocation, I want `macro_action_history.npy` to log the active high-level routing slot at each step, so that regime classification timelines remain unbroken.
10. As a software engineer running automated test suites, I want invalid arguments (e.g. `--action_persistence 0`) to fail fast with descriptive validation errors, so that misconfigurations are detected before long runs.
11. As a trading environment wrapper, I want persistence locks to reset immediately when an action becomes blocked by margin or price-limit boundaries, so that the policy can re-evaluate legal alternative actions.
12. As a researcher testing multi-contract datasets, I want `reconfigure_routing` and `reset_routing_state` to cleanly reset all persistence counters between contracts and Optuna trials, so that subsequent trials never suffer from state leakage.

## Implementation Decisions

- **Architecture & Domain Terminology**:
  - Add canonical domain terms `动作持续性 (Action Persistence)` and `防御抢占 (Defensive Preemption)` to `CONTEXT.md`.
  - Record ADR `docs/adr/0045-high-level-vae-routing-action-persistence-with-defensive-preemption.md` documenting the deliberate architectural choice of high-level defensive priority over low-level action hysteresis.
- **Artifact Contract**:
  - Add `HistoryArtifactNames.ACTION_DECISION_REASON_HISTORY_NPY = "action_decision_reason_history.npy"` to `FineFT/common/artifacts.py`.
- **High-Level VAE Routing (`vae_routing_util.py`)**:
  - Expose `--action_persistence` (integer, default: 3) in `parser`.
  - In `vae_risk_aware_routing`:
    - Validate `action_persistence > 0`.
    - Maintain state variables: `remaining_persist`, `current_action`, `persisting_slot_id`, and `action_decision_reason_history`.
    - In `reset_routing_state()`: Reset `remaining_persist = 0`, `current_action = self.zero_position_action`, and clear `action_decision_reason_history`.
    - In `reconfigure_routing(args)`: Update `self.action_persistence = int(args.action_persistence)` and invoke `reset_routing_state()`.
    - In `get_action(info, s, current_position, current_leverage)`:
      - Check defensive conditions (non-main defense, OOD gating, empty model). If defensive: reset `remaining_persist = 0`, log code `2: defensive_preemption` (if previously persisting) or `3: defensive_rule_close`, execute `_defensive_action`.
      - If non-defensive: check if `remaining_persist > 0` and `info["avaliable_action"][current_action]`.
        - If active and available: decrement `remaining_persist -= 1`, reuse `current_action`, log code `1: action_persistence`, skip `agent_act`.
        - If active but unavailable: reset `remaining_persist = 0`, log code `4: action_unavailable_break`, invoke `agent_act`.
        - If inactive (`remaining_persist == 0`): invoke `agent_act`. If action is non-flat and `action_persistence > 1`, set `remaining_persist = action_persistence - 1`. Log code `0: policy_inference`.
    - In `run_single_valid_df`: Save `action_decision_reason_history.npy` alongside `micro_action_history.npy`. Populate `trading_info` with `total_steps`, `inference_steps`, `persistence_held_steps`, `skip_inference_ratio`, `defensive_preemptions`, and `action_unavailable_breaks`.
- **Optuna Tuning (`vae_routing_optuna.py`)**:
  - Expose `--action_persistence` in `parser_all` with default 3.
  - Forward `base_args.action_persistence = args_2.action_persistence` in `prepare_base_args`.
- **Shell Scripts**:
  - In `FineFT/script/test/DiHFT/high_level/vae_optuna_fu_10.sh`: Add `ACTION_PERSISTENCE=${ACTION_PERSISTENCE:-3}` and `--action_persistence "${ACTION_PERSISTENCE}"`.
  - In `FineFT/script/test/DiHFT/high_level/final_result_fu_10.sh`: Add `ACTION_PERSISTENCE=${ACTION_PERSISTENCE:-3}` and `--action_persistence "${ACTION_PERSISTENCE}"`.

## Testing Decisions

- **Testing Principles**:
  - Test external behavior through public execution contracts rather than private helper variables.
  - Rely on a single high-level seam: `vae_risk_aware_routing.test()` / `vae_risk_aware_routing.run_single_valid_df()` and CLI parsing.
- **Target Seam**:
  - Seam: `vae_risk_aware_routing` contract evaluation interface (`FineFT/RL/DiHFT/high_level/vae_routing_util.py`).
  - Inputs: Controlled `pd.DataFrame` slices, manifest stubs, and mock environments.
  - Outputs Verified: Exported `micro_action_history.npy`, `action_decision_reason_history.npy`, `macro_action_history.npy`, and `trading_info.npy`.
- **Test Scenarios**:
  1. Default and custom CLI argument parsing validation, ensuring non-positive values raise `ValueError`.
  2. Action persistence holding non-flat action across consecutive steps and skipping neural network calls.
  3. Action persistence not locking flat action (zero position action), verifying every step invokes inference.
  4. High-level defensive preemption (OOD gating or non-main defense) breaking active persistence immediately.
  5. Unavailable action mask breaking active persistence and triggering immediate re-evaluation.
  6. Optuna multi-trial reconfigutation resetting all persistence counters cleanly.
- **Prior Art**:
  - `FineFT/tests/rl/test_test_agent_index.py` (lines 870-950) for persistence loop verification patterns.
  - `FineFT/tests/rl/test_vae_routing_allow_reverse_position.py` and `test_vae_routing_final_result.py` for high-level routing test harness patterns.

## Out of Scope

- Modifying the underlying Q-network architecture or reward shaping.
- Altering the mathematical formulas for VAE Gaussian log-likelihood or quantile estimations.
- Modifying low-level training routines (`parallel_weight_advantage_pretrain.py`), which already implement their own action persistence.

## Further Notes

- The decision reason integer codes are backward-compatible and self-contained. Downstream tools that only inspect `micro_action_history.npy` remain completely unaffected.
