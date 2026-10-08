# Multi-Horizon Cross-Frequency State Representation and Three-Tier Predictive Funnel Specification

## Problem Statement

In the FineFT commodity futures trading system, both the low-level reinforcement learning agent and the high-level VAE regime router suffer from acute temporal blindness due to a "16-hour micro horizon" bottleneck:

1. **Low-Level Agent Counter-Trend Shorting**:
   The candidate technical indicator pool caps rolling window calculations at 96 bars (equivalent to 16 trading hours or less than 3 trading days), with only isolated features reaching 192 bars. When trading long-horizon trend contracts spanning 3,500 to 4,800 bars (3 to 4 months of sustained directional price action, such as `fu2503` rising +36.7%), the agent has zero visibility into the macro trend. During healthy multiday bull runs, minor 2-to-3 hour technical pullbacks of 30 to 50 points appear inside the agent's 16-hour micro window as high-conviction downward reversals. The agent repeatedly executes counter-trend short trades, suffering severe whipsaws and capital losses (e.g., executing 692 short steps during a 4-month single-sided bull rally).

2. **High-Level VAE Regime Blindness and Micro Noise Contamination**:
   The high-level VAE router is tasked with identifying market regimes (Bull/Bear slope and High/Low volatility) to route trades to specialized expert agents. However, Slope VAE currently caps its input features at a 48-step window (8 hours), while Volatility VAE caps at 96 steps (16 hours). Furthermore, VAE feature sets are currently contaminated with high-frequency microstructural noise (such as 20-minute price trends and 2-hour orderbook flow imbalances). These micro features introduce severe latent space jitter, triggering spurious regime switches and false out-of-distribution (OOD) circuit-breaker halts. VAE completely lacks 20-to-40-day macro trend anchors.

3. **Strict Prohibition of Contextual Conditioning**:
   Injecting high-level VAE latent vectors $z_t$ or discrete regime IDs into the low-level policy network (Contextual RL) is strictly prohibited. Crossing the architectural boundary violates the Triple-Stream Feature Decoupling Architecture (ADR 0037 / ADR 0041), directly contaminating micro-execution with macro clustering estimation noise and inducing out-of-distribution (OOD) policy collapse. The low-level agent and high-level VAE must resolve multi-horizon perception natively within their own decoupled observation spaces.

4. **Short-Term Predictive Evaluation Paradox**:
   In multi-contract feature selection, candidate features are evaluated against short-horizon forward return labels ($k \le 48$ bars, or up to 8 hours). Long-horizon macro factors spanning 720 bars (20 trading days) have negligible instantaneous RankIC on 10-minute next-bar returns. Subjecting macro factors to standard short-term predictive filters guarantees that 100% of macro factors are erroneously eliminated in both RL and VAE feature selection.

5. **Lack of Granular Multi-Tier Manifest Diagnostics**:
   Existing feature selection manifests (`FeatureSelectionManifest` and `StreamAuditRecord`) provide aggregate dropped counts but completely lack scale-tier breakdown. Developers and risk auditors cannot inspect how many candidates entered at micro, meso, and macro levels, how each statistical gate filtered each scale, or whether final feature sets satisfy target scale allocations.

6. **Cold-Start Boundary and Zero-NaN Constraints**:
   The preprocessing pipeline enforces rigid zero-NaN validation, terminating execution on any missing values. Applying 720-step rolling windows to individual contract datasets would produce NaNs across the initial 20% of contract samples unless a principled causal expanding-window fallback is implemented.

## Solution

Construct a fully integrated **Asymmetric Multi-Horizon Triple-Stream Architecture** paired with a **Three-Tier Predictive Funnel**, **Scale-Stratified Clustering Quotas**, and **Multi-Tier Manifest Diagnostics**:

1. **Asymmetric Scale Specialization Across Decoupled Streams**:
   Establish clear physical division of labor between high-level macro routing and low-level trade execution:
   - **High-Level VAE (Strategic Commander)**:
     - **Slope VAE Stream** (12~16 dims): Micro 0% (0 dims, micro orderbook noise strictly blacklisted), Meso ~60% (8~10 dims of 1~5 day swing momentum and support/resistance), Macro ~40% (4~6 dims of 20~40 day moving average deviation, ROC, and trend slope).
     - **Volatility VAE Stream** (10~14 dims): Micro 0~10% (0~1 dims), Meso ~55% (6~8 dims of multiday range and dispersion), Macro ~40% (4~5 dims of 720-step trend-to-noise ratio and term-structure spread z-scores).
   - **Low-Level RL Agent (Tactical Executor)**:
     - **RL Decision Stream** (135~160 dims): Micro ~60% (85~100 dims of orderbook depth, orderflow imbalance, and queue dynamics), Meso ~25% (35~45 dims of intraday swings and calendar session progress), Macro ~15% (10~15 dims of macro trend deviation and cross-month basis anchoring).
   - **Network Topology**: Preserves single-stream flat state tensor inputs for both VAE encoders and RL Q-networks without multi-branch complexity.

2. **Scale-Invariant Continuous Causal Macro Operators**:
   Generate continuous rolling causal features for macro windows $W \in \{720, 1440\}$ steps (20 to 40 trading days) strictly restricted to benchmark mark prices and cross-month spreads:
   - **Directional Trend Operators (for Slope VAE & RL)**:
     - Dimensionless Moving Average / Exponential Moving Average Deviation normalized by Average True Range:
       $$\text{mark\_price\_ema\_deviation}_w = \frac{P_t - \text{EMA}_w(P)_t}{\text{ATR}_w(P)_t + \epsilon}$$
     - Standardized Long-Horizon Rate of Change:
       $$\text{mark\_price\_roc}_w = \frac{P_t - P_{t-w}}{P_{t-w} + \epsilon} \times 1000$$
     - Normalized Linear Trend Slope (Beta):
       $$\text{trend\_beta}_w = \frac{\text{Slope}_w(P) \times w}{P_{t-w} + \epsilon}$$
   - **Undirected Dispersion & Signal Purity Operators (for Volatility VAE & RL)**:
     - Dimensionless Trend-to-Noise Ratio:
       $$\text{trend\_to\_noise}_w = \frac{|P_t - P_{t-w}|}{\sum_{i=0}^{w-1} |P_{t-i} - P_{t-i-1}| + \epsilon} \in [0, 1]$$
     - Multi-Month Rolling Spread Z-Scores:
       $$\text{cm\_main\_sub\_spread\_rolling\_zscore}_w = \frac{S_t - \text{Mean}_w(S)_t}{\text{Std}_w(S)_t + \epsilon}$$

3. **Causal Expanding-Window Cold-Start Guard**:
   Enforce a minimal periods threshold (`min_periods = 48`). When the elapsed contract history is less than the target macro window $w$, compute statistics using expanding causal windows anchored from contract inception. This guarantees zero NaNs across the entire contract lifecycle, strictly satisfies zero-NaN validation, and avoids discarding the initial 20% of contract training samples.

4. **Three-Tier Predictive Funnel**:
   Partition feature evaluation into three distinct physical scales, each audited against forward return targets matched to its native horizon:
   - **Micro Tier**: Features matching orderbook microstructure and short windows ($W \le 24$). Evaluated on forward return targets $k \in \{1, 2, 6, 12\}$, with decision window $k_{\text{dec}} = 6$. Thresholds: $|\text{RankIC}| \ge 0.010$, $\text{SignConsistency} \ge 0.55$, $\text{RankIC-IR} \ge 0.18$.
   - **Meso Tier**: Features matching intraday swings and multiday bars ($48 \le W \le 192$). Evaluated on forward return targets $k \in \{16, 24, 48, 96\}$, with decision window $k_{\text{dec}} = 24$. Thresholds: $|\text{RankIC}| \ge 0.015$, $\text{SignConsistency} \ge 0.58$, $\text{RankIC-IR} \ge 0.15$.
   - **Macro Tier**: Features matching long-horizon trend and cross-month spreads ($W \ge 720$). Evaluated on forward return targets $k \in \{192, 384, 720\}$, with decision window $k_{\text{dec}} = 192$. Thresholds: $|\text{RankIC}| \ge 0.020$, $\text{SignConsistency} \ge 0.60$, $\text{RankIC-IR} \ge 0.12$.
   - Supports directional RankIC for Slope VAE and RL, and absolute return Vol-RankIC for Volatility VAE.

5. **Stratified Orthogonal Dedup Quotas**:
   In the correlation-based clustering deduplication stage, allocate explicit cluster capacity across scales for all three streams:
   - **RL Stream**: Micro 85~100 features, Meso 35~45 features, Macro 10~15 features (total bounded to 135~160 features).
   - **Slope VAE Stream**: Meso 8~10 features, Macro 4~6 features (total bounded to 12~16 features; Micro strictly blacklisted).
   - **Volatility VAE Stream**: Micro 0~1 features, Meso 6~8 features, Macro 4~5 features (total bounded to 10~14 features).

6. **Granular Multi-Tier Manifest Diagnostics**:
   Upgrade `StreamAuditRecord` and `FeatureSelectionManifest` to log complete scale-stratified diagnostic breakdowns for each stream:
   - Candidate counts per tier (Micro, Meso, Macro).
   - Gate drop counts per tier across anti-causality, RankIC, SignConsistency, and Stability-IR.
   - Dedup cluster allocations and final selected feature lists categorized by tier.

7. **Decoupled Offline Dynamic Programming Alignment**:
   Maintain the backward induction dynamic programming expert table on discrete price transitions without macro state discretization. During pretraining warmup, the student policy network directly projects the multi-horizon observation vector onto the expert's trend-locked optimal discrete actions.

## User Stories

1. As a quantitative portfolio manager, I want the low-level trading policy to observe 20-to-40-day macro trend features, so that the agent avoids initiating counter-trend short positions during intraday pullbacks in major bull markets.
2. As a systemic risk researcher, I want the high-level Slope VAE router to observe 20-to-40-day macro moving average deviation and momentum, so that the regime classification does not flip to Bear or Flat during temporary multihour dips in major bull runs.
3. As a high-level router architect, I want microstructural orderbook noise to be completely blacklisted from Slope VAE, so that high-frequency quote fluctuations do not destabilize the latent regime representation.
4. As a volatility modeler, I want Volatility VAE to incorporate 720-step trend-to-noise ratios, so that the router cleanly differentiates between smooth single-sided rallies and volatile range-bound chop.
5. As a reinforcement learning researcher, I want the multi-horizon observation space to remain completely decoupled from high-level VAE latent vectors, so that estimation noise from macro regime classification does not corrupt low-level trade execution.
6. As a deep learning engineer, I want both the VAE encoders and the low-level Q-network to accept flat state vectors, avoiding multi-branch channel overhead while allowing dense layers to learn cross-scale feature interactions.
7. As a quantitative researcher, I want macro trend features to be evaluated on long-horizon forward returns ($k \in [192, 720]$), so that predictive macro factors are not discarded by short-term IC filters.
8. As an algorithmic trader, I want micro orderbook features to continue being audited on short-horizon forward returns ($k \in [1, 12]$), so that execution alpha and queue-imbalance metrics preserve their high-frequency predictive standards.
9. As a quantitative risk analyst, I want the macro evaluation tier to enforce a high directional RankIC threshold ($|\text{RankIC}| \ge 0.020$) and cross-contract sign consistency ($\ge 0.60$), so that noisy non-directional long-term indicators are eliminated.
10. As a data pipeline engineer, I want macro rolling calculations to utilize an expanding window fallback with `min_periods = 48`, so that individual contracts produce zero NaNs from step zero.
11. As a compliance auditor, I want feature scaling validation to reject any dataset containing NaNs, ensuring that invalid numerical inputs fail fast before reaching model training.
12. As a machine learning engineer, I want the feature selection pipeline to automatically categorize candidates into micro, meso, and macro tiers via naming contracts, avoiding manual whitelisting maintenance.
13. As a feature engineering developer, I want macro moving average deviation to be normalized by Average True Range, so that price level variations across different commodity contracts do not introduce non-stationary scale drift.
14. As a quantitative researcher, I want the macro feature pool to include dimensionless trend-to-noise ratios, so that the system can differentiate between persistent clean trends and volatile sideways drift.
15. As a portfolio manager, I want the macro feature set to include multi-month cross-contract spread z-scores, so that structural term-structure shifts are observable directly by the models.
16. As a machine learning architect, I want the correlation clustering deduplication stage to maintain dedicated feature quotas for micro, meso, and macro tiers across all three feature streams, so that macro factors are not wiped out by correlation pruning.
17. As an algorithmic trading researcher, I want the RL decision stream to retain between 10 and 15 orthogonal macro features, providing sufficient dimensionality for trend level, momentum velocity, and signal purity.
18. As a regime modeling researcher, I want Slope VAE to retain 4 to 6 macro directional features alongside 8 to 10 intermediate swing features, achieving a responsive yet stable macro classification.
19. As a systems auditor, I want `FeatureSelectionManifest` to log granular candidate and survivor statistics broken down by scale tier for every stream, providing complete traceability of the selection funnel.
20. As a performance engineer, I want macro features to be generated directly inside the continuous time feature calculation module, avoiding redundant intermediate file I/O and additional script orchestration.
21. As an RL practitioner, I want the offline DP expert algorithm to remain free of macro state discretization, preventing exponential state-space explosion and slow tabular solving.
22. As an execution researcher, I want the student Q-network to map multi-horizon state inputs to regime-locked DP expert actions during behavioral cloning warmup, establishing a strong initial trend-following policy.
23. As a backtesting analyst, I want the trained agent to maintain long exposure throughout multiday bull runs, securing historical peak profits without panic exits during minor pullbacks.
24. As a data scientist, I want mandatory time-of-day and contract role features to count against intermediate and macro quotas, preserving total state dimension bounds between 135 and 160 features for RL.
25. As a DevOps engineer, I want feature selection manifests to record the multi-horizon funnel audit metrics for each tier, ensuring complete mathematical reproducibility across training runs.

## Implementation Decisions

### 1. Decoupled Asymmetric Triple-Stream Architecture
- Maintain physical decoupling across three distinct feature spaces:
  - **Slope VAE Stream** (`vae_slope_state_features.npy`): $12 \sim 16$ dimensions, composed of ~60% Meso ($8 \sim 10$ dims) and ~40% Macro ($4 \sim 6$ dims). Micro orderbook and high-frequency features are strictly blacklisted.
  - **Volatility VAE Stream** (`vae_volatility_state_features.npy`): $10 \sim 14$ dimensions, composed of $0 \sim 1$ Micro (0~10%), ~55% Meso ($6 \sim 8$ dims), and ~40% Macro ($4 \sim 5$ dims).
  - **RL Decision Stream** (`rl_state_features.npy`): $135 \sim 160$ dimensions, composed of ~60% Micro ($85 \sim 100$ dims), ~25% Meso ($35 \sim 45$ dims), and ~15% Macro ($10 \sim 15$ dims).
- Preserve single-stream flat state tensor inputs for both VAE encoders and the low-level Q-network.
- Strictly prohibit contextual conditioning (e.g., injecting VAE latent $z_t$ or regime grid IDs into the low-level Q-network).

#### VAE Scale Allocation Analysis and Justification
Why should high-level VAE features be concentrated in the **Meso (~60%) and Macro (~40%) tiers**, while **strictly blacklisting Micro features** from Slope VAE?
1. **Strategic Router vs. Tactical Executor**:
   - The High-Level VAE functions as the strategic commander, identifying market regimes (Bull/Bear slope and High/Low volatility) to route control to specialized policies.
   - The Low-Level RL agent functions as the tactical executor, managing order execution, queue timing, and fine-grained entry/exit on 10-minute bars.
2. **Micro Noise Toxicity to VAE Latent Spaces**:
   - Microstructural features (orderbook depth imbalance, 10~20 min price fluctuations, orderflow ratios) are dominated by high-frequency white noise, bid-ask bounce, and fast mean-reverting dynamics.
   - Injecting micro features into Slope VAE causes high-frequency latent vector $z_t$ jitter. The VAE rapidly flickers between Bull and Bear regimes within a single session (e.g., 10:00 Bull -> 10:30 Bear -> 11:00 Bull). This destabilizes downstream policy execution and triggers spurious out-of-distribution (OOD) circuit breakers.
   - Therefore, Slope VAE enforces a **strict 0% Micro quota** (0 features). Volatility VAE permits at most 1 dimension of short-term volatility shock (0~10%).
3. **Meso Tier (~60% Slope VAE, ~55% Volatility VAE) as the Dynamic Foundation**:
   - Spanning 1 to 5 trading days ($W \in [48, 192]$ steps), the Meso tier captures multi-day swing momentum, moving average slopes, swing support/resistance, and session progression.
   - For Volatility VAE, multi-day ATR ranges and Parkinson volatility capture volatility clustering (GARCH persistence) without intraday quote noise.
   - This provides the regime router with responsive adaptation to multi-day swing turning points without overreacting to intraday blips.
4. **Macro Tier (~40% Slope VAE, ~40% Volatility VAE) as the Trend Anchor**:
   - Spanning 20 to 40 trading days ($W \in [720, 1440]$ steps), Macro features provide scale-invariant structural anchors:
     - Directional: ATR-normalized EMA deviation (`mark_price_ema_deviation_720`), long Rate of Change (`mark_price_roc_720`), and trend slope (`trend_beta_720`).
     - Volatility & Dispersion: Dimensionless trend-to-noise ratio (`trend_to_noise_720`) and term-structure spread z-scores (`cm_main_sub_spread_rolling_zscore_720`).
   - Macro anchors completely cure VAE's "16-hour blindness". When a 3-month single-sided bull market experiences an intraday 2-hour dip of 30~50 points, the macro anchor anchors the Slope VAE latent vector firmly inside the Bull regime region, preventing disastrous false flips to Bear or Flat.

### 2. Dimensionless Macro Feature Set Specification
- Extend continuous time feature calculation to support long-horizon window lengths $W \in \{720, 1440\}$ steps for 10-minute bars (corresponding to 20 and 40 trading days).
- Restrict macro indicator generation strictly to benchmark mark price and cross-month contract price spreads:
  - `mark_price_ema_deviation_{w}`: $(P_t - \text{EMA}_w(P)_t) / (\text{ATR}_w(P)_t + \epsilon)$
  - `mark_price_roc_{w}`: $(P_t - P_{t-w}) / (P_{t-w} + \epsilon) \times 1000$
  - `trend_to_noise_{w}`: $|P_t - P_{t-w}| / (\sum_{i=0}^{w-1} |P_{t-i} - P_{t-i-1}| + \epsilon)$
  - `trend_beta_{w}`: $(\text{Slope}_w(P) \times w) / (P_{t-w} + \epsilon)$
  - `cm_main_sub_spread_rolling_zscore_{w}`: $(S_t - \mu_w) / (\sigma_w + \epsilon)$
- Prohibit rolling orderbook volume/depth calculations over macro windows to eliminate uninformative noise.

### 3. Causal Expanding-Window Cold-Start Fallback
- For all rolling window operators with $W \ge 48$, set minimum observation periods to 48 steps (`min_periods = 48`).
- When contract index $t < W$, evaluate statistical operators over the expanding history $[0, t]$, ensuring all output arrays contain valid finite values with zero NaNs.
- Maintain seamless convergence to standard rolling statistics once history length reaches $W$.

### 4. Automatic Scale Classification via Naming Contracts
- Classify candidate features into three tiers prior to predictive auditing using strict regular expression patterns:
  - **Macro Pattern**: Matching long windows and calendar horizons:
    `r"(_(720|1440|2160)_|prev_(5|10|15|20|30)_day|prev_(1|2|4|6)_week|cm_.*_(720|1440))"`
  - **Meso Pattern**: Matching intraday and multiday swing windows:
    `r"(_(48|96|192)_|prev_day_|prev_2_day_|session_|trading_minute_)"`
  - **Micro Pattern**: Default fallback for all remaining features (capturing high-frequency orderbook levels, orderflow imbalance, and short rolling windows $\le 24$).

### 5. Three-Tier Predictive Funnel Architecture
- Compute multi-horizon forward return arrays for each contract frame across distinct window sets:
  - Micro Horizons: $k \in \{1, 2, 6, 12\}$
  - Meso Horizons: $k \in \{16, 24, 48, 96\}$
  - Macro Horizons: $k \in \{192, 384, 720\}$
- Execute predictive audit gates per tier:
  - Evaluate mean $|\text{RankIC}|$, $\text{RankIC-IR}$, and $\text{SignConsistency}$ on tier-specific forward returns (directional for Slope VAE and RL; Vol-RankIC for Volatility VAE).
  - Sign consistency decision windows: $k_{\text{dec}} = 6$ for Micro, $k_{\text{dec}} = 24$ for Meso, $k_{\text{dec}} = 192$ for Macro.
  - Filter thresholds:
    - Micro: $|\text{RankIC}| \ge 0.010$, $\text{SignConsistency} \ge 0.55$, $\text{RankIC-IR} \ge 0.18$
    - Meso: $|\text{RankIC}| \ge 0.015$, $\text{SignConsistency} \ge 0.58$, $\text{RankIC-IR} \ge 0.15$
    - Macro: $|\text{RankIC}| \ge 0.020$, $\text{SignConsistency} \ge 0.60$, $\text{RankIC-IR} \ge 0.12$

### 6. Stratified Multi-Scale Clustering and Quota Allocation
- Group surviving candidates from the predictive funnel by scale tier before correlation deduplication.
- Execute agglomerative clustering within each tier using maximum correlation threshold 0.80 for RL, 0.65 for Slope VAE, and 0.60 for Volatility VAE.
- Enforce bounded capacity quotas per tier across all three streams:
  - **RL Stream**: Micro 85~100, Meso 35~45, Macro 10~15 (total $135 \sim 160$).
  - **Slope VAE Stream**: Meso 8~10, Macro 4~6 (total $12 \sim 16$; Micro strictly 0).
  - **Volatility VAE Stream**: Micro 0~1, Meso 6~8, Macro 4~5 (total $10 \sim 14$).
- Persist individual feature lists: `vae_slope_state_features.npy`, `vae_volatility_state_features.npy`, and `rl_state_features.npy`.

### 7. Multi-Tier Manifest Diagnostics Schema and Process Documentation
- The feature selection manifest (`FeatureSelectionManifest`) and per-stream audit records (`StreamAuditRecord`) must provide complete transparency and traceability into the multi-tier selection process across all three streams (`rl_stream`, `vae_slope_stream`, `vae_volatility_stream`).
- Upgrade `StreamAuditRecord` to include a structured `tier_breakdown` dictionary, recording the end-to-end funnel journey for each scale tier:
  ```json
  {
    "tier_breakdown": {
      "micro": {
        "forward_horizons": [1, 2, 6, 12],
        "decision_horizon": 6,
        "thresholds": {
          "min_abs_ic": 0.010,
          "sign_consistency": 0.55,
          "stability_ir": 0.18
        },
        "candidates": 120,
        "anti_causal_dropped": 0,
        "rank_ic_dropped": 15,
        "sign_consistency_dropped": 8,
        "stability_ir_dropped": 5,
        "survivors": 92,
        "selected_quota": [85, 100],
        "selected": 90
      },
      "meso": {
        "forward_horizons": [16, 24, 48, 96],
        "decision_horizon": 24,
        "thresholds": {
          "min_abs_ic": 0.015,
          "sign_consistency": 0.58,
          "stability_ir": 0.15
        },
        "candidates": 80,
        "anti_causal_dropped": 0,
        "rank_ic_dropped": 10,
        "sign_consistency_dropped": 6,
        "stability_ir_dropped": 4,
        "survivors": 60,
        "selected_quota": [35, 45],
        "selected": 40
      },
      "macro": {
        "forward_horizons": [192, 384, 720],
        "decision_horizon": 192,
        "thresholds": {
          "min_abs_ic": 0.020,
          "sign_consistency": 0.60,
          "stability_ir": 0.12
        },
        "candidates": 25,
        "anti_causal_dropped": 0,
        "rank_ic_dropped": 2,
        "sign_consistency_dropped": 1,
        "stability_ir_dropped": 2,
        "survivors": 20,
        "selected_quota": [10, 15],
        "selected": 12
      }
    }
  }
  ```
- **Manifest Process Documentation Field (`process_documentation`)**:
  Include a structured narrative block in the manifest documenting the multi-tier selection process:
  1. **Tier Partitioning Stage**: All input features are automatically mapped to Micro, Meso, or Macro via non-invasive regex naming patterns without modifying underlying extraction logic.
  2. **Predictive Funnel Stage**: Each tier is audited strictly against forward return labels matched to its native horizon. Candidates must sequentially satisfy leakage checks, tier-specific $|\text{RankIC}|$ minimums, cross-contract sign consistency, and temporal stability ($|\text{RankIC}| / \text{Std}(\text{RankIC}) \ge \text{IR}_{\text{min}}$).
  3. **Stratified Clustering Deduplication Stage**: Surviving features are grouped by tier. Agglomerative clustering operates within each tier under correlation caps (0.80 for RL, 0.65 for Slope VAE, 0.60 for Vol VAE) to enforce dedicated tier capacity quotas, ensuring macro features are never crowded out by correlated micro/meso indicators.
  4. **Audit Traceability**: Manifest records dropped feature names under each gate per tier, enabling developers and risk auditors to verify exact pass/fail reasons for any candidate.

### 8. Decoupled DP Expert Behavioral Cloning
- Retain backward induction dynamic programming table solver without state-space augmentation.
- Rely on ADR 0051 causal regime directional entry locks to ensure expert actions are strictly aligned with macro market direction.
- Supervise student policy training on augmented multi-horizon state observations against filtered DP actions during warmup.

## Testing Decisions

### 1. Principles of Effective Testing
- Test external observable behavior and dataset invariants across multi-contract frames rather than internal loop counters.
- Enforce strict deterministic numerical checks for scale-invariance, boundedness, and zero-NaN constraints.
- Verify that pipeline failure modes trigger immediate fast-fail exceptions (e.g., detecting unexpected NaNs or schema mismatches).

### 2. Testing Seams and Modules

#### Primary Seam: Multi-Contract Feature Selection Pipeline Integration
- **Target Module**: Multi-contract feature selection engine (`operator_futures.feature_selection.muti_contract`).
- **Verifications**:
  - Verify that candidates are correctly routed to Micro, Meso, and Macro tiers based on regular expression naming contracts.
  - Verify that Macro candidates are evaluated against long forward return targets ($k \in [192, 720]$) and survive predictive auditing when displaying strong long-term RankIC.
  - Verify that stratified clustering preserves the allocated quotas across all three streams (RL: 85~100 micro, 35~45 meso, 10~15 macro; Slope VAE: 0 micro, 8~10 meso, 4~6 macro; Volatility VAE: 0~1 micro, 6~8 meso, 4~5 macro).
  - Verify that `FeatureSelectionManifest` records the full multi-tier diagnostic breakdown for all three streams.
- **Prior Art**: `data_preprocess/tests/test_commodity_multi_contract_feature_selection.py`.

#### Secondary Seam: Continuous Time Feature Generation and Cold-Start Validation
- **Target Module**: Time feature calculation utility (`operator_futures.time_operator.multi_processing_util`).
- **Verifications**:
  - Verify that $W \in \{720, 1440\}$ operators generate valid scale-invariant features for `mark_price` and cross-month spreads.
  - Verify that datasets shorter than 720 rows produce zero NaNs using expanding window fallbacks with `min_periods = 48`.
  - Verify that zero-NaN validation passes on all processed contract feather outputs.
- **Prior Art**: `data_preprocess/tests/test_time_operator_polars.py`.

#### Tertiary Seam: Reinforcement Learning and VAE Model Ingestion
- **Target Module**: Dataset loaders, VAE routing, and Q-network interface (`FineFT.datahandler.commodity_contract_dataset`, `FineFT.RL.DiHFT.high_level.vae_routing_util`, `FineFT.model.low_level.Qnet`).
- **Verifications**:
  - Verify that Slope VAE and Volatility VAE models ingest the updated Meso/Macro feature lists and perform stable latent inference without dimensionality errors.
  - Verify that dataset loading correctly parses the updated `rl_state_features.npy` containing multi-horizon features.
  - Verify that forward passes through the low-level Q-network execute cleanly across the updated state dimensions without tensor shape mismatches.
- **Prior Art**: `FineFT/tests/datahandler/test_commodity_contract_dataset.py`, `FineFT/tests/rl/test_vae_routing_final_result.py`.

## Out of Scope

- **Contextual Reinforcement Learning**: Any injection of high-level VAE latent vectors $z_t$, discrete regime IDs, or FiLM modulation into the low-level network.
- **Recurrent or Attention Policy Architectures**: Implementation of DRQN, GRU, or Transformer-XL memory backbones for the low-level agent.
- **Tabular DP State Discretization**: Introducing multi-horizon state dimensions directly into the backward induction dynamic programming table solver.
- **High-Frequency Orderbook Macro Windows**: Calculating 720-step or 1440-step rolling statistics on microstructural orderbook quantities (e.g., bid/ask depth or cancel ratios).
- **Execution Engine Refactoring**: Modifying live order management, order routing, or physical exchange gateway connections.

## Further Notes

- **ADR References**: This specification directly implements the architectural decisions established in **ADR 0054** (`docs/adr/0054-multi-horizon-cross-frequency-state-and-three-tier-predictive-funnel.md`), while respecting the boundary constraints defined in **ADR 0037**, **ADR 0041**, **ADR 0051**, and **ADR 0053**.
- **Downstream Migration**: Implementation of this specification requires a clean execution of the data preprocessing pipeline to produce updated feather files, followed by feature selection to regenerate `rl_state_features.npy`, `vae_slope_state_features.npy`, `vae_volatility_state_features.npy`, and scaling artifacts prior to policy retraining.
