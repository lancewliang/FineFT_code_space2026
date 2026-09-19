# Tasks: 增加前一交易日合约角色档位状态特征 (add-previous-day-main-contract-state-feature)

## 1. 核心特征生成与参数对接 (Core Implementation)
- [x] 1.1 在 `data_preprocess/operator_futures/commodity/base_time_feature.py` 中向 `BASE_TIME_FEATURE_COLUMNS` 追加 `prev_day_contract_role_tier`。
- [x] 1.2 在 `generate_base_time_features` 中增加 `main_sub_roles: dict[str, dict[str, str]] | None = None` 参数，实现查表与前一交易日角色判定逻辑（`main` 为 1.0，`sub` 为 0.5，其他及无历史日为 0.0）。
- [x] 1.3 更新 `generate_and_write_base_time_feature` 及命令行解析接口，支持通过 `--summary` 路径加载 `main_contract_summary.json` 并传入 `main_sub_roles`。
- [x] 1.4 更新调用脚本（如 `fu_full_process.sh` 中的 `run_commodity_time_feature` 以及 `time_feature.sh`），确保生成 `BASE_TIME_FEATURE` 时传入 `--summary "$summary_path"`。

## 2. 流水线整合与直通配置 (Pipeline Integration)
- [x] 2.1 检查 `BASE_TIME_FEATURE_COLUMNS` 声明：确认 `fu_full_process.sh` 中既有的 `BASE_TIME_FEATURE_COLUMNS` 数组同步追加 `prev_day_contract_role_tier`。
- [x] 2.2 验证 `muti_contract_scale_save.py`：确认新特征被 `--passthrough_features` 覆盖，输出中数值严格保持 0.0 / 0.5 / 1.0，不被 RobustScaler 缩放。
- [x] 2.3 验证 `feature_selection`：确认新特征被 `--mandatory_state_features` 保护，必然输出在 `state_features.npy` 中。

## 3. 单元测试与验证 (Testing & Verification)
- [x] 3.1 编写 `test_base_time_feature_previous_day_role_tier` 单元测试：
  - 测试正常上一日为主力合约（main）时，输出特征列全为 1.0。
  - 测试正常上一日为次主力合约（sub）时，输出特征列全为 0.5。
  - 测试正常上一日为非主力合约（other）时，输出特征列全为 0.0。
  - 测试历史首日/冷启动（无历史交易日记录）时，输出特征列全为 0.0 且无 NaN。
- [x] 3.2 运行现存预处理测试套件，确保下游 merge、scale_save 及 dataset_split 接口回归测试通过。
