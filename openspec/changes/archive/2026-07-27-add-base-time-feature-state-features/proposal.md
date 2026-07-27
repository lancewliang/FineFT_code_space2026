# add-base-time-feature-state-features

## 背景与目标
在商品期货交易中，绝对日历时间（如裸分钟、绝对月份、绝对剩余天数）容易导致 RL Agent 过拟合。
本变更引入 9 个非绝对 `BASE_TIME_FEATURE` 时间编码特征，作为与 `BASE_FEATURE` 平级的特征产物。这些特征强制作为 State Feature 保留，不参与 Feature Selection 筛选指标计算，也不参与 Scale Save 的 Robust Scaler 缩放。

## 关键决策
- **9 个非绝对特征**：`trading_minute_progress` (Session 内进度), `morning_session`/`afternoon_session`/`night_session` (one-hot), `is_opening_30m`/`is_closing_30m` (0/1), `contract_month_sin`/`cos` (周期编码), `contract_life_remaining_ratio` (剩余比例)。
- **产物与 Merge**：产物路径在 `PREPROCESS_DATASET/commodity-futures/BASE_TIME_FEATURE/...`，daily merge 按 timestamp 强制匹配 join。
- **Feature Selection 强制保留**：通过 `--mandatory_state_features` 保护，不被任何 Filter 或 Blacklist 删去。
- **Scale Save Passthrough**：通过 `--passthrough_features` 保护，不执行 Robust Scaling 或 Clip。

## 验收标准
- 生成 9 个标准 `BASE_TIME_FEATURE` 列，值均为有界非绝对数。
- Daily Merge 在 timestamp 不一致时 fail-fast。
- Feature Selection 生成的 `state_features.npy` 必须包含全部 9 个特征。
- Scale Save 保存的特征数据中 9 个特征维持原始编码值。
