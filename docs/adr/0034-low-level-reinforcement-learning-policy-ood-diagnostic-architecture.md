---
status: accepted
---

# Low-Level Reinforcement Learning Policy-Level OOD Diagnostic Architecture

We establish a dedicated out-of-distribution (OOD) diagnostic architecture for low-level reinforcement learning agents (`ensemble_Qnet`), evaluating epistemic uncertainty, discrete action disagreement, and latent manifold Mahalanobis distance against stratified replay buffer baselines.

## Context

In the FineFT hierarchical trading system, high-level regime routing employs dual-axis VAEs (slope and volatility) to detect macro distribution shifts via Gaussian Negative Log-Likelihood (NLL). However, empirical analysis in `docs/research/feature_optimization_and_dual_stream_ood_architecture_research.md` established that VAE-based density estimation cannot serve as a diagnostic tool for low-level RL execution policies:

1. **Fundamental Model Paradigm Disconnect**:
   - VAE is an unsupervised generative density estimator $p(s)$, flagging states that lie in low-density feature space.
   - Low-level execution agents (`ensemble_Qnet`) are value-based reinforcement learning function approximators $Q(s, a) \approx \mathbb{E}[R_t | s, a]$. They do not model $p(s)$; they map complex states to expected cumulative returns.
2. **Distinct Failure Modes in Reinforcement Learning**:
   - A high-density feature region can still suffer catastrophic policy failure if the RL agent experiences value function extrapolation error, Q-value divergence, or chaotic action flipping.
   - Conversely, slight feature novelty may be benign if the policy's value landscape remains smooth and confident.
   - Therefore, policy OOD must measure **epistemic uncertainty** and **decision consensus**, rather than input density alone.
3. **Existing Native Model Capabilities**:
   - The low-level agent natively deploys `ensemble_Qnet` (comprising $M$ independent Q-networks initialized with different seeds/bootstraps, `FineFT/model/low_level.py`).
   - The training pipeline generates stratified replay buffer snapshots (`buffer_diverse.pkl` in `FineFT/RL/DiHFT/low_level/parallel_diverse_train.py`), covering 9 market regime grids with tens of thousands of transitions.
   - These assets provide a rigorous, empirical foundation for policy-level OOD diagnostics.

## Decision

We establish the following architectural decisions for low-level agent OOD evaluation:

### 1. Triad Mathematical Metric Framework
We formulate three complementary, non-redundant quantitative diagnostic metrics:

1. **Ensemble Q Epistemic Variance ($U_{\text{greedy}}(s)$)**:
   Measures epistemic variance of the ensemble around the consensus greedy action $a^* = \arg\max_a \bar{Q}(s, a)$:
   $$U_{\text{greedy}}(s) = \frac{1}{M} \sum_{m=1}^M \left( Q_m(s, a^*) - \bar{Q}(s, a^*) \right)^2$$
   Points exceeding the 95th percentile of the training replay buffer distribution are flagged as high epistemic uncertainty OOD.

2. **Optimal Action Decision Disagreement Rate ($\text{Disagreement}(s)$)**:
   Quantifies discrete execution conflict across the $M$ independent sub-networks:
   $$\text{Disagreement}(s) = 1 - \frac{\max_{a} \sum_{m=1}^M \mathbb{I}(a_m^*(s) = a)}{M}, \quad a_m^*(s) = \arg\max_a Q_m(s, a)$$
   Ranges strictly from $0.0$ (unanimous consensus) to $1 - 1/M$ (maximum voting dispersion).

3. **Policy Latent Manifold Mahalanobis Distance ($D_M(s)$)**:
   Measures distance in the agent's internal multimodal decision manifold. Features are extracted post-`fc2` (64-dimensional fused representation incorporating market state, countdown clocks, previous action, and trading account metrics) and averaged across ensemble sub-networks:
   $$\bar{h}(s) = \frac{1}{M} \sum_{m=1}^M h_m(s)$$
   $$D_M(s) = \sqrt{(\bar{h}(s) - \mu_h)^T \Sigma_h^{-1} (\bar{h}(s) - \mu_h)}$$
   Covariance matrix $\Sigma_h$ is fitted on the training reference buffer using Ledoit-Wolf shrinkage to ensure well-conditioned, positive-definite inversion.

### 2. Dual Evaluation Execution Paradigms
The diagnostic tool supports two operational modes via `--eval_mode`:
- **`env_rollout` (Default, High Fidelity)**: Steps through real sequential market data inside `initiate_base_env`, generating authentic trajectories with dynamic position holding, execution masks, and margin accounting. Evaluates contracts continuously from flat position (action 0).
- **`static_scan` (Lightweight Exploration)**: Processes Feather datasets directly with neutral dynamic variables (`previous_action=0`, `avaliable_action=all_ones`, `trading_info=zeros`) for fast offline feature screening.

### 3. Stratified In-Distribution Calibration
To construct an unbiased baseline:
- Sample $K=2,000$ transitions uniformly from each of the 9 regime grids in `buffer_diverse.pkl` ($N \approx 18,000$ total transitions).
- Compute baseline distributions for $U_{\text{greedy}}$, $\text{Disagreement}$, and $D_M$, extracting reference quantiles ($Q_{90}, Q_{95}, Q_{99}$).
- If `buffer_diverse.pkl` is not available, the tool supports auto-synthesizing reference transitions from the training split.

### 4. Implementation Placement and Artifacts
- The tool is implemented at `/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/feature/low_level_agent_ood_analysis.py`, mirroring `vae_feature_ood_analysis.py`.
- Results are saved to `analysis_result/DiHFT/agent_ood/{symbol}/{target_freq}/`, producing:
  - `agent_ood_summary.csv`: Tabular statistics and OOD breach ratios per contract.
  - `agent_epistemic_uncertainty_timeseries.png`: Dual-axis time series with price, agent actions, Q variance, and 95% threshold line.
  - `action_agreement_distribution.png`: Histogram and CDF of ensemble action voting agreement.
  - `latent_mahalanobis_kde.png`: Kernel density estimate comparing In-Distribution vs Out-of-Distribution latent distances.

## Consequences

- The system achieves full-stack OOD observability: high-level VAE monitors macro regime density collapse, while low-level diagnostic tools monitor micro execution uncertainty and decision conflict.
- The pipeline establishes an objective, empirical criterion for detecting RL policy degradation before live capital deployment.
- High-fidelity rollout execution accurately reflects live trading dynamics without introducing artificial static state artifacts.
