---
status: accepted
---

# Require Trading Info Qnet Input

Qnet and ensemble_Qnet calls must explicitly pass `trading_info` after adding Trading Process Feature input. We chose a forced migration instead of defaulting missing `trading_info` to a zero vector because silent compatibility would let training or evaluation paths run without the new risk-aware input, making experiment results unreliable. Old checkpoints without `fc_trading` weights are intentionally incompatible and should be retrained or migrated explicitly.
