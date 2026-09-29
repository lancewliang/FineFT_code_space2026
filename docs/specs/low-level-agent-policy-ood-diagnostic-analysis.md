# Spec: Low-Level Agent Policy-Level OOD Diagnostic Analysis Tool

- **Triage Label**: `ready-for-agent`
- **Related Research**: `docs/research/feature_optimization_and_dual_stream_ood_architecture_research.md` (Section 5)
- **Related ADR**: `docs/adr/0034-low-level-reinforcement-learning-policy-ood-diagnostic-architecture.md`
- **Target Subsystem**: `FineFT/analysis/feature/low_level_agent_ood_analysis.py`

---

## Problem Statement

Existing OOD analysis scripts in the FineFT framework (`vae_feature_ood_analysis.py`) focus solely on generative feature density estimation via VAE Gaussian Negative Log-Likelihood (NLL). However, low-level execution policies (`ensemble_Qnet`) are value-based discriminative networks $Q(s, a)$. Their failure modes manifest as epistemic uncertainty explosion, value function divergence, and erratic action flipping.

Currently, there is no quantitative tool to audit whether an evaluated reinforcement learning policy is operating in familiar territory or blindly extrapolating in uncharted state space during validation or testing.

## Solution

Build a specialized CLI analysis tool `FineFT/analysis/feature/low_level_agent_ood_analysis.py`:
1. Extract and quantify **three policy OOD metrics**:
   - Ensemble Q Epistemic Variance $U_{\text{greedy}}(s)$ on the consensus action.
   - Action Disagreement Rate $\text{Disagreement}(s)$ across ensemble sub-networks.
   - Latent Policy Manifold Mahalanobis Distance $D_M(s)$ from post-`fc2` embeddings with Ledoit-Wolf shrinkage.
2. Establish an **In-Distribution Baseline** using 9-grid stratified sampling from `buffer_diverse.pkl` to compute $Q_{90}, Q_{95}, Q_{99}$ warning thresholds.
3. Support **dual evaluation execution**:
   - `env_rollout`: Full continuous trajectory inside `initiate_base_env` with realistic position and mask dynamics.
   - `static_scan`: Rapid offline Feather file scanning with neutral dynamic defaults.
4. Output structured tabular analytics (`agent_ood_summary.csv`) and graphical diagnostics (`agent_epistemic_uncertainty_timeseries.png`, `action_agreement_distribution.png`, `latent_mahalanobis_kde.png`).

## User Stories

1. As a quantitative researcher, I want to evaluate `ensemble_Qnet` on out-of-sample test splits and see if the agent's actions are supported by ensemble consensus or driven by high-variance extrapolation.
2. As a risk manager, I want to know what percentage of evaluation steps exceed the 95th percentile training variance threshold ($U_{\text{greedy}} > Q_{95}$), so that high-risk contracts or market regimes can be flagged.
3. As a trading strategy engineer, I want time-series plots aligning asset price, position changes, and epistemic uncertainty spikes, allowing visual verification of policy behavior during flash crashes or sudden trend reversals.
4. As an automated test engineer, I want clean, modular functions for calculating Q-variance, disagreement rate, and Mahalanobis distances that can be unit-tested without requiring full GPU resources or multi-gigabyte models.

## Implementation Details

### 1. Mathematical Metric Calculations
```python
def compute_ensemble_metrics(
    q_values: torch.Tensor,  # shape: (B, M, N_ACTIONS)
    hidden_states: torch.Tensor | None = None,  # shape: (B, D)
    baseline_stats: BaselineStats | None = None,
) -> dict[str, np.ndarray]:
    ...
```
- Greedy consensus action: $a^*(s) = \arg\max_a \frac{1}{M}\sum_{m=1}^M Q_m(s, a)$
- Epistemic variance: $U_{\text{greedy}}(s) = \frac{1}{M}\sum_{m=1}^M (Q_m(s, a^*) - \bar{Q}(s, a^*))^2$
- Optimal action per sub-net: $a_m^*(s) = \arg\max_a Q_m(s, a)$
- Disagreement rate: $1.0 - (\max_a \sum_{m=1}^M \mathbb{I}(a_m^*(s) == a)) / M$
- Latent Mahalanobis distance: $\sqrt{(h - \mu_h)^T \Sigma_h^{-1} (h - \mu_h)}$

### 2. Intermediate Feature Extraction
In `ensemble_Qnet`:
- Sub-network $m$ computes `information_hidden = self.fc2(torch.cat([state_hidden, previous_action_hidden, time, trading_hidden], dim=1))`.
- Extract `information_hidden` (dim=64) from each sub-network and average across the $M$ sub-networks to yield $\bar{h}(s) \in \mathbb{R}^{64}$.

### 3. Baseline Calibration
```python
@dataclass
class BaselineStats:
    mu_latent: np.ndarray  # (64,)
    precision_matrix: np.ndarray  # (64, 64) - inverse covariance
    q_var_quantiles: dict[int, float]  # {90: ..., 95: ..., 99: ...}
    disagree_quantiles: dict[int, float]
    mahalanobis_quantiles: dict[int, float]
```
- Load `buffer_diverse.pkl`.
- For each of 9 grids, sample up to $K=2,000$ transitions.
- Feed through model to extract latent embeddings and Q values.
- Fit `sklearn.covariance.LedoitWolf().fit(latent_embeddings)` to obtain precision matrix $\Sigma_h^{-1}$.
- Compute empirical quantiles.

### 4. CLI Interface
```bash
python FineFT/analysis/feature/low_level_agent_ood_analysis.py \
    --model_path result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/epoch_75/trained_model.pkl \
    --buffer_path result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/buffer_diverse.pkl \
    --data_dir PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/10min \
    --feature_path PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/state_features.npy \
    --split test \
    --symbol fu \
    --target_freq 10min \
    --output_dir analysis_result/DiHFT/agent_ood/fu/10min \
    --eval_mode env_rollout \
    --device cpu
```

## Testing Decisions

1. **Unit Tests**:
   - `test_compute_ensemble_metrics_unanimous`: Verify when all $M$ nets produce identical Q-values, $U_{\text{greedy}} = 0.0$ and $\text{Disagreement} = 0.0$.
   - `test_compute_ensemble_metrics_complete_disagreement`: Verify when all $M$ nets vote for different actions, $\text{Disagreement} = 1 - 1/M$.
   - `test_latent_mahalanobis_distance`: Verify Mahalanobis distance equals zero at distribution mean and scales quadratically.
2. **Integration Test**:
   - Run end-to-end analysis on dummy synthetic data in a temporary directory verifying generation of `agent_ood_summary.csv` and png visualizations.
