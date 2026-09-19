---
status: accepted
---

# Previous-Day Contract Role Tier as Base_Time_feature

The `prev_day_contract_role_tier` feature is integrated into `Base_Time_feature` as a mandatory, unscaled passthrough state feature encoding a 3-tier liquidity status: `1.0` for main contract, `0.5` for sub-main contract, and `0.0` for other contracts or cold-start boundaries. We chose this over a binary indicator and multi-contract cross-month features because it directly quantifies the contract's liquidity standing across three tiers in a bounded $[0, 1]$ scalar without requiring aligned cross-month contract pairs. Furthermore, following ADR-0003, it passes through `Scale Save` without `RobustScaler` scaling to preserve strict monotonic tier semantics (`1.0 / 0.5 / 0.0`), and defaults to `0.0` when no previous trading day exists in the summary.
