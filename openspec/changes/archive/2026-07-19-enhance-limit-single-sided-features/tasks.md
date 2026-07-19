## 1. Implementation

- [x] 1.1 Extend commodity orderbook downscale outputs and reward/execution manifest to include `LowerLimitPrice` and `UpperLimitPrice`. <!-- 已实现: 商品 orderbook 下采样与 reward/execution manifest 已包含涨跌停价列 -->
- [x] 1.2 Enhance snapshot feature generation for single-sided books, including `ask_side_empty` and `bid_side_empty`. <!-- 已实现: snapshot 特征已支持单边盘口、空侧 WAP fallback、side-empty 标志和双侧空 fail-fast -->
- [x] 1.3 Update expected-column helpers and documentation for the expanded reward and snapshot feature contracts. <!-- 已实现: expected columns、snapshot/reward/time 文档已同步 84/108/3375 合同并覆盖 snapshot/commodity manifest 顺序 -->
- [x] 1.4 Add and run focused validation for reward columns, single-sided snapshot behavior, time feature input legality, and OpenSpec strict validation. <!-- 已实现: 已增加 time_feature_input 合法单边盘口回归并通过 focused/combined pytest 与 OpenSpec strict validation -->
