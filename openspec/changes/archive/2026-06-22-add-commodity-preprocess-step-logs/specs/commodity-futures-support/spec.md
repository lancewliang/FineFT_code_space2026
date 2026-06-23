## ADDED Requirements

### Requirement: 商品期货主流程步骤日志
系统 SHALL 为商品期货 preprocess 主流程的主要阶段生成独立步骤日志，并在总日志中记录阶段状态。

#### Scenario: 主流程生成步骤日志
- **WHEN** 用户运行 `data_preprocess/script_preprocess/future_upgraded/commodity/main.sh`，且 `SYMBOL=fu`、`TARGET_FREQ=5min`、`START_DATE=2025-11-03`、`END_DATE=2025-11-08`
- **THEN** 系统 SHALL 为 `stitch_main_contract`、`downscale_continuous_by_trading_day`、`cross_section`、`merge`、`concat`、`time_feature`、`merge_clean`、`ic_correlation` 和 `scale_save` 生成独立日志文件
- **AND** 每个步骤日志文件名 SHALL 包含 symbol、target_freq、start_date、end_date 和步骤名
- **AND** 每个步骤日志 SHALL 捕获该步骤的 stdout 和 stderr

#### Scenario: 总日志记录阶段状态
- **WHEN** 商品 preprocess 主流程执行任一主要步骤
- **THEN** 总日志 SHALL 记录该步骤的开始信息和步骤日志路径
- **AND** 当步骤成功完成时，总日志 SHALL 记录该步骤成功完成
- **AND** 当步骤失败时，总日志 SHALL 记录该步骤失败和对应日志路径

#### Scenario: 失败语义保持 fail-fast
- **WHEN** 任一主要步骤返回非 0 状态
- **THEN** 商品 preprocess 主流程 SHALL 以非 0 状态退出
- **AND** 系统 SHALL 保留失败步骤日志中的错误输出
- **AND** 系统 MUST NOT 因日志包装而继续执行后续主要步骤

#### Scenario: 现有子日志继续保留
- **WHEN** `cross_section` 或 `merge` 阶段继续按日期启动子任务日志
- **THEN** 系统 SHALL 保留现有按日期子日志目录和文件
- **AND** 新增步骤日志 MUST NOT 删除、重命名或替代这些子日志
