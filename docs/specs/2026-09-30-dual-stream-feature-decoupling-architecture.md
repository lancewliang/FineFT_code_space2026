# Spec: Dual-Stream Feature Decoupling Architecture for VAE Regime Identification and Low-Level RL Agent

- **Triage Label**: `ready-for-agent`
- **Related Research**: `docs/research/feature_optimization_and_dual_stream_ood_architecture_research.md`, `docs/research/multi_contract_feature_engineering_and_selection_ood_remediation_report.md`
- **Related ADR**: `docs/adr/0037-dual-stream-feature-decoupling-architecture-for-vae-and-rl-agent.md`
- **Target Subsystems**: Multi-Contract Feature Selection Pipeline (`operator_futures.feature_selection.muti_contract`), Scaling Pipeline (`muti_contract_scale_save.py`), Downstream VAE Ingestion (`FineFT.datahandler.vae_data_creation`), Pipeline Manifest Infrastructure

---

## Problem Statement

Following the Phase 1 & 2 remediation, the multi-contract feature engineering and selection pipeline eliminated catastrophic Out-of-Distribution (OOD) likelihood collapse for the high-level dual-axis Variational Autoencoder (VAE) (recovering the test log-likelihood sum from $-30$ to $+1.11$, and restoring variance ratios from $2.34$ back to $0.9885$).

However, solving the VAE OOD collapse through a single monolithic state feature pipeline inadvertently destroyed the profitability and decision capacity of the Low-Level Reinforcement Learning Agent (`ensemble_Qnet`):

1. **Fundamental Objective Divergence Between VAE and RL**:
   - The high-level VAE is an unsupervised generative density estimator that requires strictly stationary, Gaussian-like, low-dimensional ($12 \sim 18$ dims), and cross-contract invariant features. Any heavy-tailed microstructural spike or non-stationary variance burst triggers quadratic reconstruction penalties and log-likelihood collapse.
   - The low-level RL agent is a Markovian decision policy optimizing cumulative trading reward. It thrives on non-linear, high-dimensional ($50 \sim 65$ dims), microstructural price asymmetries, orderbook imbalances (Order Flow Imbalance, depth depletion), and short-horizon momentum signals.
2. **Microstructural Signal Destruction via Single-Stream Filtering**:
   - Applying strict VAE stationarity gates ($\text{PSI} \le 0.10$, tight pairwise correlation $r \le 0.70$, aggressive persistence filtering) across a unified single stream discarded over 80% of critical execution Alpha features. Specifically, Level-5 Order Flow Imbalance (`level5_ofi_weighted_norm`), orderbook depth depletion ratios, and short-horizon log returns (`log_return_1`, `log_return_2`, `log_return_6`) were completely wiped out.
3. **Impaired Agent Profitability**:
   - Deprived of microstructural Alpha signals, the low-level agent's state representation collapsed into coarse macro-smoothed indicators, rendering it incapable of detecting transient price imbalances or timing entries and exits, resulting in the complete loss of trading edge.
4. **Coupled Dimension Constraints**:
   - Legacy downstream loaders hardcoded state dimensions or assumed identical state vectors for both the VAE and the policy network, creating an architectural barrier that prevented selecting distinct feature representations tailored to each consumer.

---

## Solution

We design and implement the **Dual-Stream Feature Decoupling Architecture (双流特征解耦架构)** within the multi-contract feature selection and downstream training pipeline:

1. **Decoupled Dual Feature Streams**:
   - **VAE Regime Feature Stream (`vae_state_features.npy`)**: A compact, hyper-stationary representation ($12 \sim 18$ dimensions) governed by strict distribution stability ($\text{PSI} \le 0.10$), high correlation deduplication ($r \le 0.65$), persistence filtering of high-frequency noise, and time topology features. This stream guarantees zero VAE OOD false rejections and robust regime clustering.
   - **RL Decision Feature Stream (`rl_state_features.npy`)**: A high-capacity, Alpha-rich representation ($50 \sim 65$ dimensions) governed by relaxed distribution drift ($\text{PSI} \le 0.25$), flexible correlation capacity ($r \le 0.80$), RankIC and non-linear CatBoost importance prioritization, and full unblocking of 5-level Order Flow Imbalance, depth depletion, and short returns.
2. **Single-Pass Deep Module Selection Pipeline**:
   - Execute Stage 1 data hygiene and calculate shared statistical metrics (IC, RankIC, Sign Consistency, IR, PSI) once across the wide candidate pool.
   - Fit the target-horizon ($w=6$) CatBoost regressor with a purged embargo gap once on the union of surviving candidates.
   - Branch into two symmetric `StreamFilterProfile` evaluations (`DEFAULT_VAE_PROFILE` and `DEFAULT_RL_PROFILE`) to independently rank, filter, cluster, and cap the VAE and RL feature sets without redundant I/O or repeated model training.
3. **Unified Scaling via Dual-Stream Union State Features (`state_features.npy`)**:
   - Construct the Dual-Stream Union State Features as the union set: $\mathcal{S}_{\text{union}} = \mathcal{S}_{\text{vae}} \cup \mathcal{S}_{\text{rl}}$, persisted to `state_features.npy`.
   - The scaling module (`muti_contract_scale_save.py`) scales $\mathcal{S}_{\text{union}}$ once using adaptive rolling Z-score and hyperbolic tangent soft-saturation $\tanh(z / M)$ into a single wide dataset.
   - Downstream consumers perform zero-copy column slicing on demand (`df[vae_features]` or `df[rl_features]`), eliminating data duplication and storage bloat.
4. **Physical Separation of Mandatory Features**:
   - The VAE stream isolates and retains only intraday time topology indicators (`base_time_*`, `time_*`, `trading_minute_*`), stripping cross-month velocity, market share, and contract roles.
   - The RL stream retains the complete set of 17 domain mandatory features, ensuring full market microstructure context for policy execution.
5. **Dynamic Input Dimension Adaptation**:
   - Downstream VAE preparation (`vae_data_creation.py`) and model initializers dynamically resolve `INPUT_DIM = len(vae_state_features)` from `vae_state_features.npy`, removing all hardcoded dimensional constraints.
6. **Dual-Stream Manifest Audit Contract**:
   - `feature_selection_manifest.json` adopts a structured schema recording global union features alongside dedicated `vae_stream` and `rl_stream` audit breakdowns detailing surviving counts, dropped reasons, and composite priority distributions.

---

## User Stories

1. As a quantitative researcher, I want the feature selection pipeline to unblock 5-level Order Flow Imbalance (`level5_ofi_weighted_norm`) for the RL decision stream, so that the low-level agent can exploit microstructural orderbook imbalance Alpha.
2. As a reinforcement learning engineer, I want the RL decision stream to retain short-horizon returns (`log_return_1`, `log_return_2`, `log_return_6`), so that the policy network can detect immediate price momentum.
3. As a high-level VAE researcher, I want the VAE regime stream to strictly enforce $\text{PSI} \le 0.10$ and aggressive persistence noise filtering, so that test-set out-of-distribution log-likelihood collapse is permanently prevented.
4. As a quantitative researcher, I want the VAE regime stream restricted to $12 \sim 18$ dimensions, so that the isotropic Gaussian density estimator does not suffer from high-dimensional sphere-shell concentration artifacts.
5. As a reinforcement learning practitioner, I want the RL decision stream to support $50 \sim 65$ dimensions, so that the policy network has sufficient state representation capacity for complex Markovian decision making.
6. As a system architect, I want the selection pipeline to calculate metrics and fit the CatBoost regressor only once across the shared candidate pool, so that dual-stream selection introduces zero redundant computation overhead.
7. As a machine learning engineer, I want feature stream filtering behaviors governed by strongly typed configuration profiles (`StreamFilterProfile`), so that filtering parameters are explicit, validated, and free of magical runtime constants.
8. As a data engineer, I want the pipeline to output a unified `state_features.npy` containing the union of both streams, so that existing scaling scripts and downstream data pipelines continue to operate without breaking changes.
9. As a data engineer, I want `muti_contract_scale_save.py` to scale the union dataset once into `df.feather`, so that disk storage footprint does not double.
10. As a high-level VAE consumer, I want `FineFT/datahandler/vae_data_creation.py` to prioritize loading `vae_state_features.npy` when present, so that VAE datasets are generated strictly from the hyper-stationary regime stream.
11. As a low-level agent trainer, I want the environment state reader to ingest `rl_state_features.npy`, so that the agent's observation space is populated exclusively with Alpha-rich decision features.
12. As a risk controller, I want the VAE regime stream to strip contract roles and cross-month velocity, retaining only intraday time topology, so that contract roll transitions do not induce false regime alarms.
13. As an RL policy designer, I want the RL decision stream to preserve all 17 mandatory state features, so that the execution agent maintains full awareness of spread, basis, open interest, and contract state.
14. As an MLOps engineer, I want `feature_selection_manifest.json` to record separate audit sections for `vae_stream` and `rl_stream`, so that automated deployment pipelines can verify the health of both models independently.
15. As a developer, I want the VAE network architecture to dynamically adapt to `len(vae_state_features)`, so that changing the VAE feature count does not require editing hardcoded model layer dimensions.
16. As a quantitative researcher, I want the RL stream composite scoring to allocate 50% weight to RankIC and 35% weight to CatBoost importance, so that predictive power dominates feature prioritization over distribution stability.
17. As a high-level VAE researcher, I want the VAE stream composite scoring to allocate 50% weight to PSI drift penalty and 30% weight to RankIC, so that distribution invariance dominates regime feature ranking.
18. As a developer, I want the pipeline to raise a fail-fast `ValueError` if either the VAE or RL stream yields fewer than its configured minimum cluster count, so that silent degradation is immediately flagged during training runs.
19. As a quantitative analyst, I want hierarchical clustering in the RL stream to permit a pairwise correlation threshold up to $r = 0.80$, so that complementary multi-scale orderbook features are not prematurely pruned.
20. As a high-level VAE analyst, I want hierarchical clustering in the VAE stream to enforce a strict correlation threshold of $r \le 0.65$, so that latent regime dimensions remain strictly orthogonal and well-conditioned.
21. As a data engineer, I want filtered contract feather files in the feature selection directory to persist the union of selected features, so that intermediate contract slices are self-contained.
22. As a researcher, I want the feature selection pipeline to support single-stream legacy mode via a configuration switch (`dual_stream: bool = True`), so that historical ablation studies remain reproducible.
23. As a developer, I want all configuration objects to be immutable frozen dataclasses with explicit custom class type hints, complying strictly with repository coding guidelines and prohibiting dynamic `getattr` / `hasattr` checks.
24. As a test engineer, I want the dual-stream feature selection pipeline to be fully testable end-to-end using in-memory synthetic DataFrames, so that CI/CD runs execute in seconds without disk dependencies.
25. As a portfolio manager, I want the combined system to restore positive low-level agent trading PnL while maintaining high-level VAE test log-likelihood $\ge 0.0$, so that the trading strategy is both profitable and resilient against regime shifts.

---

## Implementation Decisions

### Decision 1: Structured Feature Stream Configuration Profile Contract
We introduce the `StreamFilterProfile` dataclass to encapsulate all filtering, scoring, and dimensionality thresholds for an individual feature stream.

```python
@dataclass(frozen=True)
class StreamFilterProfile:
    name: str
    max_mean_psi: float
    max_pair_psi: float
    min_abs_ic: float
    min_sign_consistency: float
    min_rank_ic_ir: float
    max_correlation: float
    min_clusters: int
    max_clusters: int
    psi_weight: float
    rank_ic_weight: float
    catboost_weight: float
    filter_micro_persistence: bool
    mandatory_feature_pattern: str | None = None
```

Two canonical pre-configured profile instances are defined:
- `DEFAULT_VAE_PROFILE`:
  - `name`: `"vae_regime"`
  - `max_mean_psi`: `0.10`, `max_pair_psi`: `0.20`
  - `min_abs_ic`: `0.015`, `min_sign_consistency`: `0.70`, `min_rank_ic_ir`: `0.35`
  - `max_correlation`: `0.65`
  - `min_clusters`: `12`, `max_clusters`: `18`
  - `psi_weight`: `0.50`, `rank_ic_weight`: `0.30`, `catboost_weight`: `0.20`
  - `filter_micro_persistence`: `True`
  - `mandatory_feature_pattern`: `r"^(base_time_|time_|trading_minute_)"`
- `DEFAULT_RL_PROFILE`:
  - `name`: `"rl_decision"`
  - `max_mean_psi`: `0.25`, `max_pair_psi`: `0.35`
  - `min_abs_ic`: `0.020`, `min_sign_consistency`: `0.65`, `min_rank_ic_ir`: `0.30`
  - `max_correlation`: `0.80`
  - `min_clusters`: `50`, `max_clusters`: `65`
  - `psi_weight`: `0.15`, `rank_ic_weight`: `0.50`, `catboost_weight`: `0.35`
  - `filter_micro_persistence`: `False`
  - `mandatory_feature_pattern`: `None` (retains all mandatory features)

### Decision 2: Single-Pass Dual-Branch Pipeline Seam
Rather than executing the pipeline twice (which would double I/O and CatBoost training overhead), the pipeline runs data loading, data hygiene, and predictive metric computation once on the wider candidate pool:
1. `execute_data_hygiene` runs once across all candidate features.
2. `audit_distribution_drift` runs once, computing pairwise and mean PSI across contracts.
3. `execute_predictive_audit` runs once across all decision windows, computing vectorized IC, RankIC, Sign Consistency, and Stability IR.
4. `execute_nonlinear_scoring` fits CatBoost once on the target decision window ($w=6$) using the union of candidates surviving the relaxed RL criteria.
5. Dual branching occurs post-scoring:
   - Branch A applies `DEFAULT_VAE_PROFILE` to filter, score, cluster, and select $12 \sim 18$ features for `vae_state_features.npy`.
   - Branch B applies `DEFAULT_RL_PROFILE` to filter, score, cluster, and select $50 \sim 65$ features for `rl_state_features.npy`.
   - The union $\mathcal{S}_{\text{union}} = \mathcal{S}_{\text{vae}} \cup \mathcal{S}_{\text{rl}}$ is saved to `state_features.npy`.

### Decision 3: Storage and Scaling Strategy
- The feature selection step persists three NumPy artifact arrays in the train output directory:
  - `vae_state_features.npy`: VAE Regime Feature Stream.
  - `rl_state_features.npy`: RL Decision Feature Stream.
  - `state_features.npy`: Dual-Stream Union State Features.
- `muti_contract_scale_save.py` consumes `state_features.npy` as its target feature list. It scales each contract DataFrame once using adaptive rolling Z-score and $\tanh(z/M)$ soft-saturation, writing a single unified `df.feather`.
- Downstream models perform column projection on the unified table in memory. No secondary data tables or duplicated feather files are generated.

### Decision 4: Mandatory Features Functional Isolation
- Domain mandatory features (17 indicators including cross-month basis, volume ratios, time progress, and contract roles) are selectively partitioned:
  - For the VAE stream, regex matching against `mandatory_feature_pattern` retains only time-topology features (`base_time_*`, `time_*`, `trading_minute_*`), filtering out contract roles and basis terms that undergo regime shifts during contract rollover.
  - For the RL stream, all 17 mandatory features are passed through into the state vector, ensuring full observation context.

### Decision 5: Dynamic VAE Input Dimension Adaptation
- In `FineFT/common/artifacts.py`, register `VAE_STATE_FEATURES_NPY = "vae_state_features.npy"` and `RL_STATE_FEATURES_NPY = "rl_state_features.npy"`.
- In `FineFT/datahandler/vae_data_creation.py`, inspect the dataset directory: if `vae_state_features.npy` is present, load it; otherwise fall back to `state_features.npy`.
- VAE network architectures and dataset builders instantiate their input layers using `INPUT_DIM = len(vae_state_features)` rather than static constants.

### Decision 6: Dual-Stream Manifest Schema Contract
`feature_selection_manifest.json` is augmented with a top-level `stream_mode: "dual"` property and dedicated audit sub-objects:
```json
{
  "stream_mode": "dual",
  "selected_features": ["..."],
  "selected_feature_count": 68,
  "vae_stream": {
    "profile_name": "vae_regime",
    "selected_features": ["..."],
    "selected_feature_count": 16,
    "filter_results": { ... }
  },
  "rl_stream": {
    "profile_name": "rl_decision",
    "selected_features": ["..."],
    "selected_feature_count": 58,
    "filter_results": { ... }
  }
}
```

---

## Testing Decisions

### What Makes a Good Test
A good test exercises external behavior against public contracts rather than internal implementation details:
- Verify that calling the pipeline entry point with `dual_stream=True` creates `vae_state_features.npy`, `rl_state_features.npy`, and `state_features.npy` on disk.
- Verify that `len(vae_state_features)` is within the VAE profile range ($12 \sim 18$) and contains time topology features without high-noise micro returns.
- Verify that `len(rl_state_features)` is within the RL profile range ($50 \sim 65$) and retains order flow imbalance and short-horizon returns.
- Verify that `state_features.npy` contains the exact set union of `vae_state_features.npy` and `rl_state_features.npy`.
- Verify that `feature_selection_manifest.json` contains `stream_mode: "dual"` and valid audit structures for both streams.
- Verify that `muti_contract_scale_save.py` successfully scales the union dataset without missing column errors.
- Verify that `vae_data_creation.py` loads `vae_state_features.npy` when available and generates valid VAE train/test arrays with the exact dimension matching the VAE stream.

### Highest Testing Seams
1. **Primary Seam (Feature Selection Entry Point)**:
   - Module: `data_preprocess.operator_futures.feature_selection.muti_contract.pipeline.run_feature_selection`
   - Input: In-memory synthetic multi-contract DataFrames with known microstructural Alpha signals and macro drift properties.
   - Assertions: Output file presence, correct feature dimensions, set union property, and manifest schema compliance.
2. **Secondary Seam (Scaling & Downstream Ingestion)**:
   - Modules: `data_preprocess.operator_futures.scale_describe_save.muti_contract_scale_save` and `FineFT.datahandler.vae_data_creation.make_data`
   - Input: The output directory generated by the primary seam.
   - Assertions: Successful scaling into unified `df.feather` and correct VAE array generation with dynamic input dimensions.

### Prior Art
- `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`: Existing synthetic contract generator and CatBoost mock infrastructure (`fake_catboost`, `tmp_path`).
- `data_preprocess/tests/test_commodity_scale_save.py`: Existing test suite validating scaling and NaN inspection.

---

## Out of Scope

1. **Phase 2 Ingestion of VAE Latents into RL Policy**:
   - Feeding VAE latent embedding $z_t$ into `ensemble_Qnet` input state is reserved for Phase 2. In Phase 1, the RL agent consumes pure `rl_state_features` without synchronous VAE inference coupling.
2. **Retraining High-Level Agent Hyperparameters**:
   - Tuning heuristic routing thresholds (`rule_base_threshold`) or retraining VAE weights on the new 15-dimensional dataset is an execution task, not a feature selection architecture change.
3. **Modifying Feature Operator Formulas**:
   - The underlying mathematical calculation of features in `multi_processing_util.py` (e.g. OFI, Bollinger bands, volatility) is already completed and out of scope for this spec.

---

## Further Notes

- **Zero Storage Bloat**: Because both feature streams are sliced as views from a single scaled `df.feather`, disk usage remains $1\times$ rather than $2\times$.
- **Zero Latency Penalty**: Fitting CatBoost once on the shared pool adds zero extra tree training time compared to the single-stream pipeline.
