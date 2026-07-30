# 06 — 在 Scale Save 中缩放跨月特征

**要验证什么:** 不修改 Scale Save 生产代码。现有 Scale Save 逻辑会缩放所有未列入 `passthrough_features` 的 State Feature；Ticket 05 已将跨月合约结构特征写入 Feature Selection 的 mandatory state feature 列表，而商品期货 Scale Save 脚本只把 Base_Time_feature 传入 `--passthrough_features`。因此本 ticket 只需要用测试确认跨月特征会自然参与 Rolling Robust Scaling 的 fit、transform 和 clip，并且不会被记录为 passthrough feature。

**被阻塞于:** 05 — 让跨月特征通过 Feature Selection 强制保留。

**Status:** skipped

- [x] 不修改 Scale Save 生产代码。
- [x] 现有 Scale Save 机制已满足：未列入 `passthrough_features` 的 State Feature 会参与缩放。
- [x] Ticket 05 已保证跨月特征进入 mandatory state feature，但不会进入 Scale Save passthrough 配置。
