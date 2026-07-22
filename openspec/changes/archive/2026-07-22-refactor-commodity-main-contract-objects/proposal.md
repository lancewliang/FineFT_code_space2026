# refactor-commodity-main-contract-objects

## 背景与目标

`data_preprocess/operator_futures/commodity/main_contract.py` 现在已经有
`MainContractSummary*` 这组 dataclass，但主构建流程里仍然用多个中间 `dict` 聚合状态：
按交易日收集文件、按月份累计成交量、统计高频交易日、选择主力合约、最后再拼成 JSON。
这些状态边界是隐式的，后续维护时容易依赖字符串 key 传递数据，阅读成本也高。

本次重构目标是把 commodity 主力合约 summary 的内部拼装过程对象化，让构建流程在函数之间
传递 dataclass，而不是裸 `dict`。对外写出的 `main_contract_summary.json` 结构保持不变，
业务计算逻辑、异常时机和调用入口都不改。

## 用户场景

- 开发者维护主力合约生成逻辑时，可以通过对象字段理解“按月统计、按合约收集、按交易日裁剪”
  这条链路，而不是在多个 `dict.setdefault()` 里猜字段含义。
- 开发者修改 `main_contract_summary.json` 的写出逻辑时，可以在最终边界统一做
  `to_dict()`，避免 JSON 契约散落在流程各处。
- 下游 `downscale_continuous_by_trading_day.py` 和 `feature_selection/contract_feature_union.py`
  继续只消费 `MainContractSummary` 对象，不需要知道中间拼装细节。

## 设计方向

采用轻量 dataclass 方案，围绕 `main_contract.py` 的主链路补齐内部状态对象。

保留现有 `MainContractSummaryTradingDay`、`MainContractSummaryContract`、`MainContractSummary`
作为对外 JSON 模型；新增若干只用于流程内部的 dataclass，承载：
- 按交易日聚合到的原始文件路径
- 按月份累计的成交量
- 按月份统计的高频交易日次数
- 按合约收集的交易日列表
- 按合约记录的已选月份

`build_main_contract_summary_model_for_date_range()` 继续负责扫描原始数据和做主力合约选择，
但它会在内部传递对象而不是 `dict`。`build_main_contract_summary_for_date_range()` 改为返回
`MainContractSummary`，`write_main_contract_summary_for_date_range()` 只在文件边界执行
`summary.to_dict()` 并写出 JSON。

`load_main_contract_summary()` 保持现有对象加载行为不变。

## 关键决策

- 只重构 `data_preprocess/operator_futures/commodity/main_contract.py` 的主链路。
- 使用标准库 `dataclass`，不引入额外 schema 或校验库。
- 保持 `main_contract_summary.json` 的字段名、层级和数值类型兼容。
- 内部流程对象化，但不改变现有错误抛出和失败时机。
- 不把 `feature_selection` 目录里的 manifest JSON 一并纳入本次改造。

## 范围边界

**包含：**
- `main_contract.py` 内部主力合约构建流程的对象化。
- `build_main_contract_summary_for_date_range()` 返回对象而不是 dict。
- `write_main_contract_summary_for_date_range()` 统一在写文件边界调用 `to_dict()`。
- 更新 commodity 相关 focused tests，覆盖对象返回值和 JSON 兼容性。

**不包含（本次）：**
- 不重构 `data_preprocess/operator_futures/feature_selection` 的 JSON manifest。
- 不改变 `main_contract_summary.json` 的外部结构。
- 不新增 JSON 反序列化 API。
- 不修改主力合约选择规则、成交量计算规则或异常策略。

## 验收标准

- [ ] `build_main_contract_summary_for_date_range()` 返回 `MainContractSummary`。
- [ ] `write_main_contract_summary_for_date_range()` 只在 JSON 写出边界调用 `to_dict()`。
- [ ] 主构建流程内部不再用裸 `dict` 传递月份统计、交易日收集和合约选择状态。
- [ ] `main_contract_summary.json` 的内容与现有输出兼容。
- [ ] 新增或更新的 focused tests 能验证对象属性访问和写出 JSON 的一致性。
- [ ] `conda activate finetf && pytest` 相关 commodity focused tests 通过。
