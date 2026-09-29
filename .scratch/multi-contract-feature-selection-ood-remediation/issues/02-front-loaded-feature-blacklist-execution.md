# 02: Front-Loaded Feature Blacklist Execution in Multi-Contract Selection Pipeline

**What to build:** Restructure `operator_futures.feature_selection.muti_contract.pipeline.py` to immediately filter `candidate_universe` against `feature_blacklist` and `feature_ablation_patterns` prior to metric computation (CatBoost and ICs) and prior to correlation filtering, eliminating the "borrowed knife" phenomenon and redundant model fitting.

**Blocked by:** None (can start immediately, independent of Ticket 01)

**Status:** closed

- [x] `run_feature_selection` purges blacklisted features from `candidate_universe` immediately after loading split frames and extracting raw state features.
- [x] CatBoost regressors, IC calculations, and Permutation Importance evaluate only unblacklisted candidate features.
- [x] Correlation deduplication (`select_feature`) operates strictly on unblacklisted features, preventing blacklisted features from killing valid stationary candidates.
- [x] `feature_selection_manifest.json` logs `Feature Blacklist Dropped` at the input gate with zero blacklist drops occurring post-correlation.
- [x] Existing multi-contract unit tests in `test_commodity_multi_contract_feature_selection.py` pass with updated manifest structure.
