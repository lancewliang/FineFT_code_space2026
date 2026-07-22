## ADDED Requirements

### Requirement: 商品期货主力合约 summary 构建 SHALL 使用 dataclass 对象内部状态
系统 SHALL 使用 dataclass 对象表达商品期货主力合约 summary 构建过程中的源文件发现、月度统计、合约入选和交易日裁剪状态，并在 `main_contract.py` 内部通过对象属性和对象方法传递这些状态，而不是通过裸 dict key 传递多层聚合数据。

#### Scenario: 源文件发现和月度统计以对象传递
- **WHEN** `load_contract_files_by_trading_day_for_years()` 扫描 `data/原始下载/{commodity_name}/{YYYY}` 目录
- **THEN** 函数 SHALL 返回 `TradingDayContractSources` 之类的 dataclass 对象集合
- **AND** 每个交易日条目 SHALL 使用对象记录 `trading_day` 以及其下的 contract/source_file 明细
- **AND** `build_main_contract_summary_model_for_date_range()` SHALL 使用对象表达按月成交量、按月高成交量天数、合约交易日历史和已选月份
- **AND** 主构建流程 SHALL NOT 把嵌套 dict 当作函数之间的中间状态边界

#### Scenario: 主力合约 summary 以对象返回
- **WHEN** `build_main_contract_summary_for_date_range()` 完成构建
- **THEN** 函数 SHALL 返回 `MainContractSummary` 对象
- **AND** 对象 SHALL 暴露 `symbol`、`commodity_name`、`start_date`、`end_date`、`selection_rule` 和 `contracts`
- **AND** 下游读取 summary 的调用方 SHALL 继续通过对象属性访问 contract、trading_days 和 source_file

### Requirement: 商品期货主力合约 summary JSON serialization SHALL preserve existing contract
系统 SHALL 只在 JSON 写入边界将商品期货主力合约 summary 对象序列化为 dict，并保持现有 `main_contract_summary.json` 结构兼容。

#### Scenario: JSON 结构保持兼容
- **WHEN** `write_main_contract_summary_for_date_range()` 写出 `main_contract_summary.json`
- **THEN** JSON SHALL 保持当前兼容结构，包含 `symbol`、`commodity_name`、`start_date`、`end_date`、`selection_rule` 和 `contracts`
- **AND** `contracts` 中每项 SHALL 保持 `contract`、`start_trading_day`、`end_trading_day`、`trading_day_count`、`selected_months` 和 `trading_days`
- **AND** 写出的 JSON payload SHALL 等于 `MainContractSummary.to_dict()` 的结果
- **AND** `load_main_contract_summary()` SHALL 继续读取同一 JSON 结构并返回 `MainContractSummary`

#### Scenario: 对象层不改变现有失败语义
- **WHEN** 现有原始 CSV 缺字段、重复 `TradingDay + contract`、无入选合约、空裁剪窗口或其他主力合约生成错误条件出现
- **THEN** 系统 SHALL 继续抛出当前相同类别的异常
- **AND** dataclass 对象层 SHALL NOT 吞掉底层异常或新增独立业务校验

### Requirement: 商品期货主力合约 summary object refactor SHALL be covered by focused tests
系统 SHALL 通过聚焦测试同时验证主力合约 summary 内部对象接口和外部 JSON 兼容性。

#### Scenario: focused tests assert object return types and attributes
- **WHEN** 执行 commodity 主力合约 summary 相关测试
- **THEN** 测试 SHALL 断言 `load_contract_files_by_trading_day_for_years()` 返回 dataclass 对象集合
- **AND** 测试 SHALL 断言 `build_main_contract_summary_for_date_range()` 返回 `MainContractSummary`
- **AND** 测试 SHALL 断言 `write_main_contract_summary_for_date_range()` 写出的内容与返回对象 `to_dict()` 一致
- **AND** 测试中针对返回值的业务断言 SHALL 使用对象属性访问

#### Scenario: focused tests assert JSON payload compatibility
- **WHEN** focused tests 读取 `main_contract_summary.json`
- **THEN** 测试 SHALL 断言 `json.loads(...)` 的结果等于对应 summary 对象 `to_dict()` 的结果
- **AND** 测试 SHALL 保留关键字段、层级、合约顺序和交易日窗口裁剪字段的兼容性断言
- **AND** focused verification SHALL 使用 `conda activate finetf && pytest data_preprocess/tests/test_commodity_main_contract.py`
