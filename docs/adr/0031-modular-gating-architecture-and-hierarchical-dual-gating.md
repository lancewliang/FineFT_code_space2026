---
status: accepted
---

# Modular Gating Architecture and Hierarchical Dual Gating Strategy

We decouple the high-level VAE risk-aware gating logic into an isolated, pluggable strategy module, preserving the legacy absolute threshold gating as `AbsoluteThresholdGating` (`absolute`) and introducing `HierarchicalDualGating` (`hierarchical`) to eliminate cash-lock paralysis through dual-layer risk control.

## Context

In the FineFT three-stage framework, the high-level meta-router deploys dual-axis VAE models (trend slope and volatility) to identify market regimes and trigger defensive flat positions during uncertain market states.

Empirical evaluation on out-of-sample forward contracts (e.g., fuel oil `fu2509` at 10-minute frequency) revealed severe structural issues:
1. **Severe Return Dilution from 98%+ Cash Paralysis**: The strategy remained in cash defense over 98% to 99.6% of the backtest. Across a full year of market data, active trading occurred for only 26 to 246 bars. While win rate was high (100% on some contracts), annual total return collapsed to 0.15% ~ 0.50%.
2. **False OOD Alarms from Macro Distribution Shift**: Due to temporal drift across 1-2 years between training and forward test contracts, 90-dimensional feature reconstruction errors increased systematically, driving test log-likelihoods down from +9.96 (training median) to negative values (-8 to -30). The legacy router compared unnormalized likelihood quantiles against hardcoded absolute cutoffs (0.20 to 0.48), misidentifying normal macroeconomic drift as catastrophic black swan events.
3. **Monolithic Architecture**: Gating was hardcoded as an inline condition inside `vae_routing_util.py`, preventing modular benchmarking or isolated algorithm enhancements.

## Decision

We establish the following architectural and algorithmic policies:

### 1. Dedicated Gating Submodule & Interface
Create `FineFT/RL/DiHFT/high_level/gating/`:
- `base.py`: Defines `BaseGatingStrategy` and `GatingDecision(is_defensive, volatility_index, slope_index, reject_reason, metrics)`.
- `absolute_gating.py`: Implements `AbsoluteThresholdGating` (`absolute`), maintaining exact legacy behavior.
- `hierarchical_gating.py`: Implements `HierarchicalDualGating` (`hierarchical`).
- `factory.py`: Implements `create_gating_strategy(strategy_type, **kwargs)`.

### 2. Hierarchical Dual-Gate Algorithm
`HierarchicalDualGating` separates epistemic risk (black swan OOD) from aleatoric risk (choppy noise):
- **Gate 1 (Tail-Risk Circuit Breaker)**: Rejects only when raw max rolling weights drop below `ood_threshold` (default `0.005`). Protects against genuine catastrophic distribution failure where all VAE models fail.
- **Linear Probability Normalization**: Normalizes axis weights into relative probability distributions ($p_i = w_i / \sum w_j$).
- **Gate 2 (Relative Mode Clarity / Margin Gating)**: Computes the margin between top-1 and top-2 regimes ($\Delta p = p_{(1)} - p_{(2)}$). If $\Delta p < \text{margin\_threshold}$, market direction is ambiguous/choppy, triggering defensive flat position (`margin_ambiguity`).
- **Execution**: When both gates pass, dispatches the dominant regime sub-agent.

### 3. Full Backward Compatibility & Switching
- CLI parameter `--gating_strategy` defaults to `absolute`, guaranteeing zero regression for existing scripts and benchmarks.
- Optuna hyperparameter tuning dynamically adapts parameter suggestions based on `--gating_strategy`.
- `resolve_routing_parameters` automatically infers strategy and parameters from `high_level_agent_para.txt`.

## Consequences

### Positive
- Eliminates 98% cash-lock paralysis, increasing in-market active trading rate from ~2% to ~32% on clear market trends.
- Preserves 100% bitwise reproducibility of all legacy research baselines via `--gating_strategy absolute`.
- Decouples black swan tail risk from choppy noise filtering with explicit attribution logging.
- Gating strategies are cleanly isolated into separate files with independent unit test suites.

### Negative
- Introduces additional configuration arguments (`--gating_strategy`, `--ood_threshold`, `--slope_margin_threshold`, `--volatility_margin_threshold`).
