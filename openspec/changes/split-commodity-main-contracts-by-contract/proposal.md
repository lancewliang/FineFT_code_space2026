# split-commodity-main-contracts-by-contract

## 背景与目标

当前商品期货预处理会从多合约原始数据中挑选单一主力合约，并生成连续主力数据文件。这个做法会把不同实际合约拼成一个连续合约序列，不适合后续按真实合约生成和分析因子。

本次变更将商品期货主力合约处理改为 summary 驱动的多合约流程：`main_contract` 先计算所有入选主力合约并写出 summary JSON；后续因子管线按具体合约生成数据文件，避免不同合约混入同一个连续合约文件。

## 用户场景

- 作为数据预处理使用者，我希望按自然月成交量选出主力合约集合，而不是每天拼接成一个连续主力合约。
- 作为因子生成使用者，我希望每个入选合约都生成独立的因子文件，路径能明确区分品种和具体合约。
- 作为批处理使用者，我希望商品 shell 主流程能自动读取 summary 并对所有入选合约跑完整流程，不需要手工枚举合约。

## 设计方向

采用 summary 驱动的多合约商品期货预处理管线。

`main_contract.py` 扫描原始商品期货文件，按自然月统计每个合约的成交量。成交量口径沿用现有逻辑：单日成交量为 `Volume.max - Volume.min`，自然月成交量为该月内单日成交量求和。每个自然月选择成交量最高的前 2 个合约；同时，如果某合约在一个自然月内至少 10 个实际交易日的单日成交量大于品种配置阈值，也进入该月主力合约集合。合约只要任意月份入选，就进入主力合约集合。

`stitch_main_contract.py` 不再写出 `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv` 连续主力日文件，而是在 `PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/{symbol}/main_contract_summary.json` 写出主力合约 summary。summary 每个合约包含 `contract`、`start_trading_day`、`end_trading_day`、`trading_day_count`、`selected_months` 和 `trading_days`。其中 `trading_day_count` 等于 `len(trading_days)`，`trading_days` 记录该合约裁剪后交易窗口内实际存在的交易日、`source_file` 和 `daily_volume`。

`downscale_continuous_by_trading_day.py` 改为读取 summary 中的源文件明细，按合约和交易日生成 downscale 四类基础因子文件。它默认处理 summary 内所有合约，并可选支持 `--contract` 过滤单个合约，便于单合约重跑。

`cross_section`、`merge`、`concat`、`time_feature`、`merge_clean`、`ic_correlation` 和 `scale_save` 增加 `--contract` 参数。传入 contract 时，所有原本 `{symbol}/{target_freq}` 路径扩展为 `{symbol}/{contract}/{target_freq}`；未传 `--contract` 时保持旧路径行为，降低对非商品期货和既有调用的影响。

商品 shell 脚本一起增强或检查兼容性，范围包括：

- `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- `data_preprocess/script_preprocess/future_upgraded/commodity/main_fu.sh`
- `data_preprocess/script_preprocess/future_upgraded/commodity/main_al.sh`
- `data_preprocess/script_preprocess/future_upgraded/commodity/commodity_process.sh`
- `data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh`
- `data_preprocess/script_preprocess/future_upgraded/commodity/flatten_aluminum_raw_csv.sh`

`fu_full_process.sh` 负责先生成 summary，再从 summary 读取合约列表，对每个合约循环执行后续阶段并传递 `--contract`。存在性检查、skip 日志和步骤日志需要包含 contract 维度，避免多合约输出或日志互相覆盖。`validate_features.sh` 需要支持从 summary 读取合约列表并逐合约验证。`flatten_aluminum_raw_csv.sh` 如果只负责铝原始文件展平且没有单合约路径假设，可以保持行为不变，但需要在测试或文档中确认不受影响。

所有合约完成单合约 feature selection / scale save 后，商品主流程需要额外生成品种级训练特征合集。该步骤从 summary 中的合约列表读取每个合约最终 `SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/state_features.npy`，按 summary 合约顺序和各合约特征顺序做稳定去重，写出 `PREPROCESS_DATASET/commodity-futures/FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy` 和 `feature_union_manifest.json`。该步骤只生成统一特征清单，不把各合约行数据合并到一个文件。

现有文件粒度保持不变：前半段仍是一日一个文件，后半段仍是日期范围文件或目录，只新增 contract 目录层级。

## 关键决策

- 主力合约选择按自然月计算，每月选成交量最高的前 2 个合约，并额外纳入满足高成交量天数阈值的合约。
- 月成交量口径为 `sum(daily Volume.max - daily Volume.min)`。
- 高成交量天数规则按自然月统计：同一合约当月至少 10 个实际交易日的单日成交量 `> threshold` 即入选；阈值从品种配置读取，`fu` 为 `15000`。
- 合约只要任意自然月入选，就进入 summary；不要求连续入选。
- summary 使用 JSON，不使用 CSV，以便记录合约级信息和日级源文件明细。
- summary 文件路径为 `PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/{symbol}/main_contract_summary.json`。
- 每个合约的 `start_trading_day` 来自首次入选月份月初之后该合约实际存在的第一个 `TradingDay`。
- 每个合约的 `end_trading_day` 来自该合约原始最后交易日前第 10 个交易日对应的裁剪后最后保留 `TradingDay`。
- 因子生成只处理该合约实际有原始数据的交易日，缺失日期不生成占位文件。
- 所有下游输出路径在传 `--contract` 时统一扩展为 `{symbol}/{contract}/{target_freq}`。
- 未传 `--contract` 的后续 Python 入口保留旧路径行为。
- 多合约训练使用品种级 feature union 产物：`FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy` 是各合约最终 state feature 的并集，供后续同模型训练读取。
- 所有 commodity shell 脚本纳入本次增强或兼容性检查。

## Amendments

### 2026-07-12: Summary model bean construction
- 原因：summary JSON 当前容易退化为多层 dict/list 手工拼装，调用方和测试需要知道过多内部字段细节。
- 摘要：`main_contract.py` 的 summary 构建改为先生成类型化 summary model：顶层 `MainContractSummary`，包含 `list[MainContractSummaryContract]`，合约下包含交易日明细 model；最终由 model 统一转 dict/json。
- 行为影响：外部 `main_contract_summary.json` 字段名、字段值和 schema 不变；这是内部实现设计调整。

### 2026-07-12: Summary model bean deserialization
- 原因：读取 `main_contract_summary.json` 的代码如果继续使用裸 dict/list，会让 summary schema 细节泄漏到 downscale、shell helper 和测试中。
- 摘要：读取侧也改为通过 `MainContractSummary` model 反序列化和校验；消费方使用 summary/contract/trading-day 对象访问字段，再由 model 保持必要的 JSON/dict 兼容出口。
- 行为影响：外部 JSON schema、CLI 参数和输出路径不变；这是内部实现设计调整。

### 2026-07-12: Daily volume in summary trading days
- 原因：`main_contract_summary.json` 的日级交易日明细需要直接暴露该合约当天总成交量，便于后续检查和消费方无需重新读取原始 CSV 计算。
- 摘要：summary 的每个 `contracts[].trading_days[]` 条目新增 `daily_volume` 字段，取值为该合约该 `TradingDay` 源文件的 `Volume.max - Volume.min`。
- 行为影响：外部 `main_contract_summary.json` schema 增加日级数值字段；既有字段名和值保持不变。

### 2026-07-13: Selected contract trading window clipping
- 原因：合约一旦入选后不应再把完整日期范围内的全部交易日写入 summary；有效训练窗口应从该合约首次作为主力的月份开始，并避开临近最后交易日的尾部交易日。
- 摘要：每个入选合约的 summary 交易日窗口改为：开始下限为该合约首次入选月份的月初，结束上限为该合约原始最后交易日前第 10 个交易日；summary 仅写入该窗口内实际存在的合约交易日。
- 行为影响：`start_trading_day`、`end_trading_day`、`trading_day_count` 和 `trading_days` 的取值会变窄；JSON 字段名不变。

### 2026-07-13: High-volume-day main contract rule
- 原因：除月成交量 top 2 外，某些合约如果一个月内有足够多的高成交量交易日，也应被视为主力合约，避免被月总量排名规则漏掉。
- 摘要：新增配置化高成交量天数规则：同一自然月内，某合约至少 10 个实际交易日的 `daily_volume > threshold`，则该合约入选该月主力合约集合；`fu` 阈值配置为 `15000`。
- 行为影响：主力合约集合可能包含多于每月 2 个合约；`selection_rule` 语义扩展为 top2 与高成交量天数规则的并集。

### 2026-07-13: Cross-contract training feature union
- 原因：多合约分开做 feature selection / scale save 后，每个合约最终 state feature 集合可能不同；这些合约用于训练同一个模型时，需要一个品种级统一特征合集。
- 摘要：在所有合约完成单合约因子处理后新增最终步骤，读取 summary 中所有合约的 `state_features.npy`，生成稳定去重后的品种级 union feature list 和 manifest。
- 行为影响：新增 `PREPROCESS_DATASET/commodity-futures/FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/` 输出；不改变单合约因子文件路径和文件粒度。

## 范围边界

**包含：**

- 修改 `main_contract.py` 的主力选择与 summary 构建逻辑。
- 修改 `stitch_main_contract.py`，让 CLI 生成 `main_contract_summary.json`。
- 修改 `downscale_continuous_by_trading_day.py`，让 CLI 读取 summary 并按合约输出。
- 为后续特征脚本增加 `--contract` 路径维度，包括 `cross_section`、`merge`、`concat`、`time_feature`、`merge_clean`、`ic_correlation` 和 `scale_save`。
- 新增所有合约最终 state feature 的品种级合集生成步骤，供后续同模型训练读取。
- 增强或检查所有 commodity shell 脚本，使主流程、品种入口、批处理入口和验证脚本支持 contract 维度。
- 更新相关测试和商品预处理文档。

**不包含（本次）：**

- 不改变商品期货特征公式、盘口深度、reward/state 特征定义。
- 不改变成交量口径以外的主力合约业务规则。
- 不引入回复权、价差调整或跨合约价格连续化。
- 不改变训练环境、模型训练流程或 FineFT 运行时逻辑。
- 不要求后半段区间文件改为一天一个文件。

## 错误处理

- 原始文件缺少必要列时 fail-fast。
- 单个原始文件包含多个 `TradingDay` 时 fail-fast。
- `start_date/end_date` 不是有效左闭右开日期范围时 fail-fast。
- 日期范围内没有可交易合约或没有合约进入月度 top 2 时 fail-fast。
- 同一个 `TradingDay + contract` 命中多个源文件时 fail-fast。
- 月成交量并列时按成交量降序、合约名升序稳定排序，保证 top 2 可复现。
- summary 文件不存在、JSON 结构不合法或 `trading_day_count != len(trading_days)` 时 fail-fast。
- summary 中的 `source_file` 不存在时 fail-fast。
- 单合约重跑时，如果 `--contract` 不在 summary 中，fail-fast。
- 后续流程沿用现有缺上游产物的 skip 逻辑，但日志必须包含 `symbol` 和 `contract`。

## 验收标准

- [ ] `stitch_main_contract.py` 生成 `CONTINUOUS_RAW/{symbol}/main_contract_summary.json`，且不再生成 `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv` 连续主力日文件。
- [ ] summary 按自然月成交量 top 2 规则选出合约集合，成交量口径为日 `Volume.max - Volume.min` 后月内求和。
- [ ] summary 额外纳入每月内至少 10 个实际交易日 `daily_volume > 配置阈值` 的合约，且 `fu` 阈值为 `15000`。
- [ ] summary 每个合约包含 `contract`、`start_trading_day`、`end_trading_day`、`trading_day_count`、`selected_months` 和 `trading_days`，且 `trading_day_count == len(trading_days)`。
- [ ] summary 每个合约的 `trading_days` 只包含从首次入选月份月初到原始最后交易日前第 10 个交易日之间实际存在的合约交易日。
- [ ] summary 每个 `trading_days` 条目包含 `daily_volume`，且等于该合约该 `TradingDay` 源文件的 `Volume.max - Volume.min`。
- [ ] `downscale_continuous_by_trading_day.py` 可从 summary 处理所有合约，并可选用 `--contract` 过滤单合约。
- [ ] downscale 四类输出写到 `{FEATURE_FOLDER}/{symbol}/{contract}/{target_freq}/{date}.feather`。
- [ ] `cross_section`、`merge`、`concat`、`time_feature`、`merge_clean`、`ic_correlation` 和 `scale_save` 传 `--contract` 时读写 `{symbol}/{contract}/{target_freq}` 路径。
- [ ] 后续 Python 入口未传 `--contract` 时保持旧 `{symbol}/{target_freq}` 路径行为。
- [ ] `fu_full_process.sh` 从 summary 读取合约列表，对每个合约运行后续阶段，并在日志、skip 检查和输出路径中包含 contract。
- [ ] `main_fu.sh`、`main_al.sh`、`commodity_process.sh`、`validate_features.sh` 和 `flatten_aluminum_raw_csv.sh` 完成多合约兼容性增强或明确验证无需修改。
- [ ] 所有合约完成 `scale_save` 后，系统生成 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy` 和 `feature_union_manifest.json`。
- [ ] feature union 的 state feature 顺序稳定：先按 summary 合约顺序，再按各合约 `state_features.npy` 内顺序，重复特征只保留首次出现。
- [ ] feature union 生成时，如果 summary 中任一合约缺少最终 `state_features.npy`，系统 fail-fast 并指出缺失合约和路径。
- [ ] 测试覆盖 summary 生成、CLI 行为、contract 路径契约、shell 编排、验证脚本和错误处理。
