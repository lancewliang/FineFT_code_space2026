# 任务列表：add-base-time-feature-state-features

- [x] 新增 `BASE_TIME_FEATURE` 生成逻辑，生成 9 个时间与生命周期特征列
- [x] 在 daily merge 中实现 `BASE_TIME_FEATURE` 的 timestamp 匹配与 join
- [x] 在 Feature Selection 中实现 `--mandatory_state_features` 保护与黑名单冲突校验
- [x] 在 Scale Save 中实现 `--passthrough_features` 保留原始值跳过 Robust Scaling
- [x] 编写并运行 `test_commodity_base_time_feature.py` 等测试
