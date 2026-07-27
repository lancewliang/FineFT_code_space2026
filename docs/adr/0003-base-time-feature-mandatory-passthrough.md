---
status: accepted
---

# Base_Time_feature Mandatory Passthrough

`Base_Time_feature` columns must be appended to `state_features.npy` as mandatory state features, but they do not participate in Feature Selection metric calculation or Scale Save robust scaling. We chose this because these columns encode required trading-time and contract-lifecycle context rather than alpha candidates: filtering them by IC/importance would remove business-required context, while scaling already bounded one-hot, ratio, and sin/cos encodings would make their semantics less transparent.
