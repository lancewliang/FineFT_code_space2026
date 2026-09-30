---
status: accepted
---

# Dual-Stream Feature Decoupling Architecture for VAE Regime Identification and Low-Level RL Agent

We decouple the feature selection and state provision architecture into two specialized, orthogonal feature streams: a compact, hyper-stationary VAE Regime Feature Stream (`vae_state_features.npy`, $12 \sim 18$ dimensions) for high-level generative density estimation and zero-OOD regime routing, and an Alpha-rich RL Decision Feature Stream (`rl_state_features.npy`, $50 \sim 65$ dimensions) for low-level reinforcement learning trading policy execution (`ensemble_Qnet`). We unify data persistence via Dual-Stream Union State Features (`state_features.npy`) scaled once into a single wide dataset, avoiding compute and disk storage multiplication.

## Context

Following the Phase 1 & 2 OOD remediation (ADR-0035, ADR-0036), empirical testing on the 10-minute commodity futures pipeline (`fu`) confirmed that the high-level dual-axis VAE OOD collapse was completely eliminated:
- Test-set cumulative $\Delta\text{NLL}$ recovered from $-30.0$ down to $+1.11$.
- Contract variance ratio recovered from $2.34$ back to $0.9885$.
- OOD false rejection rate dropped to near zero.

However, resolving the VAE OOD collapse via a single monolithic state space created a severe secondary problem: **the Low-Level Reinforcement Learning Agent (`ensemble_Qnet`) lost its profitability and trading edge**. Detailed architectural and statistical auditing revealed the root cause:

1. **Fundamental Conflict of Mathematical Objectives**:
   - **High-Level VAE (Unsupervised Generative Model)**: Optimizes Evidence Lower Bound ($\text{ELBO} \approx \mathbb{E}[\log p(x|z)] - D_{\text{KL}}$) under an isotropic standard Gaussian prior. It requires strictly stationary, bounded, low-dimensional ($10 \sim 18$ dims) features with high cross-contract homogeneity ($\text{PSI} \le 0.10$). Heavy-tailed distributions or microstructural transient spikes cause quadratic NLL penalties and destroy density estimation.
   - **Low-Level Agent (Markovian Decision Process)**: Optimizes expected cumulative discounted return ($\max_\theta \mathbb{E}[\sum \gamma^t R(s_t, a_t)]$). It requires high-capacity, microstructural price asymmetries, orderbook imbalances (Order Flow Imbalance, depth depletion), and short-horizon momentum signals ($50 \sim 65$ dims) to discover profitable entry/exit opportunities.
2. **Signal Castration via Single-Stream Filtering**:
   - When a single pipeline filtered features for both models, strict VAE stationarity filters ($\text{PSI} \le 0.10$, correlation $|r| \le 0.70$, aggressive persistence dropping of fast-decaying signals) wiped out over 80% of execution Alpha signals. Crucially, 5-level Order Flow Imbalance (`level5_ofi_weighted_norm`), orderbook depth depletion ratios, and short returns (`log_return_1/2/6`) were completely discarded.
   - The low-level agent was forced to trade on overly smoothed macro indicators, losing its ability to exploit microstructural mispricings.
3. **Coupled Infrastructure**:
   - Legacy downstream data creation scripts and VAE network initializers assumed a single state feature array with hardcoded input dimensions, preventing independent state optimization.

## Decision

We establish the **Dual-Stream Feature Decoupling Architecture (双流特征解耦架构)** spanning feature selection, data scaling, and downstream model consumption:

### 1. Feature Stream Configuration Profile Contract (`StreamFilterProfile`)
We introduce a strongly typed, frozen dataclass contract `StreamFilterProfile` to govern stream filtering and scoring parameters:
- `DEFAULT_VAE_PROFILE`:
  - `name`: `"vae_regime"`
  - Distribution Drift: $\text{PSI} \le 0.10$, Max Pair $\text{PSI} \le 0.20$
  - Predictive: $\text{RankIC} \ge 0.015$, Sign Consistency $\ge 0.70$, Stability $\text{IR} \ge 0.35$
  - Orthogonal Deduplication: Spearman correlation $|r| \le 0.65$, dynamic clusters $K \in [12, 18]$
  - Composite Priority Weights: $\text{PSI}: 0.50$, $\text{RankIC}: 0.30$, $\text{CatBoost}: 0.20$
  - High-frequency persistence noise filter enabled
  - Mandatory features filtered to intraday time topology (`r"^(base_time_|time_|trading_minute_)"`)
- `DEFAULT_RL_PROFILE`:
  - `name`: `"rl_decision"`
  - Distribution Drift: $\text{PSI} \le 0.25$, Max Pair $\text{PSI} \le 0.35$
  - Predictive: $\text{RankIC} \ge 0.020$, Sign Consistency $\ge 0.65$, Stability $\text{IR} \ge 0.30$
  - Orthogonal Deduplication: Spearman correlation $|r| \le 0.80$, dynamic clusters $K \in [50, 65]$
  - Composite Priority Weights: $\text{PSI}: 0.15$, $\text{RankIC}: 0.50$, $\text{CatBoost}: 0.35$
  - High-frequency persistence noise filter disabled (unblocking OFI and short returns)
  - All 17 mandatory features fully preserved

### 2. Single-Pass Dual-Branch Pipeline Seam
We preserve computational efficiency by computing shared statistics and fitting machine learning models exactly once:
- **Shared Stage 1 & 2 Processing**: Data hygiene, distribution drift, predictive audit metrics, and target-horizon ($w=6$) CatBoost regressor importance are evaluated once on the relaxed candidate pool.
- **Dual-Branch Post-Scoring**:
  - The pipeline branches the surviving features through `DEFAULT_VAE_PROFILE` to generate `vae_state_features.npy` ($12 \sim 18$ dims).
  - The pipeline branches through `DEFAULT_RL_PROFILE` to generate `rl_state_features.npy` ($50 \sim 65$ dims).
  - The union set $\mathcal{S}_{\text{union}} = \mathcal{S}_{\text{vae}} \cup \mathcal{S}_{\text{rl}}$ is saved as `state_features.npy`.

### 3. Unified Scaling and Zero-Copy View Slicing
- `muti_contract_scale_save.py` consumes `state_features.npy` (the Union), applying adaptive rolling Z-score and hyperbolic tangent soft-saturation $\tanh(z/M)$ to scale all required columns once into a single unified `df.feather`.
- Downstream models slice their respective columns in memory (`df[vae_features]` or `df[rl_features]`). Disk storage usage remains strictly $1\times$, avoiding duplicate datasets.

### 4. Downstream VAE Dynamic Dimension Adaptation
- `FineFT/common/artifacts.py` registers `VAE_STATE_FEATURES_NPY = "vae_state_features.npy"` and `RL_STATE_FEATURES_NPY = "rl_state_features.npy"`.
- `FineFT/datahandler/vae_data_creation.py` prioritizes loading `vae_state_features.npy` when present (falling back to `state_features.npy` for backwards compatibility).
- VAE network architectures and dataset loaders resolve input dimension dynamically as `INPUT_DIM = len(vae_state_features)`.

### 5. Dual-Stream Manifest Audit
- `feature_selection_manifest.json` includes `stream_mode: "dual"` and adds dedicated `vae_stream` and `rl_stream` audit blocks recording survivor counts, dropped reasons, and profile parameters.

## Consequences

### Positive
- **Restores Agent Alpha and Profitability**: Low-level RL agent regains access to 5-level Order Flow Imbalance, orderbook depth depletion, and short returns, restoring execution edge and PnL.
- **Protects VAE from OOD Drift**: High-level VAE operates on a hyper-stationary, low-dimensional manifold ($12 \sim 18$ dims), preventing OOD log-likelihood collapse and false defensive shutdowns.
- **Zero Computational Waste**: CatBoost model fitting and feature metrics are computed once on the shared pool rather than repeated per stream.
- **Zero Disk Storage Multiplication**: Scaling the union dataset into a single wide feather file prevents doubling storage requirements across contracts.
- **Full Backward Compatibility**: Schedulers and consumers looking for `state_features.npy` continue to find a valid feature array containing all active features.

### Negative & Mitigations
- **Slightly More Complex Feature Manifest**: Manifest schema now contains dual-stream sub-records.
  - *Mitigation*: The manifest dataclass serializes cleanly to JSON with standardized keys, verified by automated unit tests.
- **Potential Discrepancy if Downstream Loader Uses Wrong File**:
  - *Mitigation*: Registered canonical artifact constants in `FineFT/common/artifacts.py` and enforce fail-fast validation in `vae_data_creation.py` and the RL environment reader.

## Compliance and Coding Standards

- Complies with repository `CLAUDE.md`: All configurations use frozen dataclasses with explicit custom class type annotations. No defensive `getattr`/`hasattr`/`isinstance` probing or silent exception swallowing.
- Fail-fast enforcement: Missing columns or empty surviving streams immediately raise `ValueError` at the earliest point of detection.
