---
status: accepted
---

# Cross-Month Term Structure Features Mandatory Inclusion and Scaling

Cross-month term structure features (covering both Main/Sub dynamic pairing and $M_1/M_2/M_3$ delivery month sequence pairing) are stored in an independent `CROSS_MONTH_FEATURE` directory and merged during daily feature pipeline execution. These features bypass Feature Selection filtering by being appended as `mandatory_state_features` into `state_features.npy`. Unlike `Base_Time_feature` which passes through unscaled, all cross-month features participate in `Scale Save` Rolling Robust Scaling. Furthermore, no absolute price level features are permitted (only relative metrics such as log price ratios, volume/OI ratios, and Z-scores), and missing values from contract liquidity gaps after timestamp alignment are filled with `0.0`.
