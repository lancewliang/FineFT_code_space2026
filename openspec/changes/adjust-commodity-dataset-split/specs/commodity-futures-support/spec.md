## MODIFIED Requirements

### Requirement: 商品期货脚本入口支持日期范围
系统 SHALL 允许商品期货主流程通过 `START_DATE` / `END_DATE` 指定跨年的日期范围，并自动生成该范围所需的主力合约 summary 与后续按合约处理文件。

#### Scenario: 日期范围驱动主流程
- **WHEN** 用户运行商品期货主流程并设置 `START_DATE=2023-01-01`、`END_DATE=2026-03-01`
- **THEN** 系统自动覆盖 2023、2024、2025 和 2026 的原始目录扫描与 summary 生成
- **AND** 系统输出 `CONTINUOUS_RAW/{symbol}/main_contract_summary.json` 供后续下采样使用
- **AND** 系统 MUST NOT 构造或依赖单条跨年连续主力大 CSV
- **AND** 系统 MUST NOT 构造或依赖 `CONTINUOUS_RAW/{symbol}/{YYYY-MM-DD}.csv` 连续主力日文件

#### Scenario: 保持左闭右开语义
- **WHEN** 用户希望覆盖到 2026-02-28 的训练窗口
- **THEN** 系统继续使用左闭右开语义，要求 `END_DATE=2026-03-01`
- **AND** 脚本和日志文件名使用日期范围语义而不是单一年份语义

#### Scenario: YEAR 仅作兼容参数
- **WHEN** 用户继续传入 `YEAR`
- **THEN** 系统可以保留该参数作为兼容输入
- **AND** 主流程不再把单一年份作为唯一运行约束

#### Scenario: full process 传递 summary
- **WHEN** `fu_full_process.sh` 调用主力合约 summary 生成和下采样
- **THEN** stitch 调用 SHALL 传递 `--output_dir PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/{symbol}`
- **AND** downscale 调用 SHALL 传递 `--summary PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/{symbol}/main_contract_summary.json`
- **AND** shell 脚本 MUST NOT 构造 `continuous_file="${symbol}_${start_date}_${end_date}.csv"` 作为 handoff
- **AND** shell 脚本 MUST NOT 把 `CONTINUOUS_RAW/{symbol}` 当作日文件目录传给 downscale

#### Scenario: full process 按 summary 合约循环并在 scale save 后切分数据集
- **WHEN** `main_contract_summary.json` 中包含合约 `fu2601` 和 `fu2605`
- **THEN** `fu_full_process.sh` SHALL 从 summary 读取合约列表
- **AND** `fu_full_process.sh` SHALL 分别为 `fu2601` 和 `fu2605` 调用 `cross_section`、`merge`、`concat`、`time_feature`、`merge_clean` 和 `scale_save`
- **AND** 每次合约级调用 SHALL 传递 `--symbols fu --contract <contract>`
- **AND** `scale_save` SHALL 在同一合约的 `merge_clean` 完成后执行
- **AND** 所有合约 `scale_save` 完成后，`fu_full_process.sh` SHALL 只调用一次 `dataset_split`
- **AND** `fu_full_process.sh` SHALL NOT 调用 `ic_candidate` logged step
- **AND** `fu_full_process.sh` SHALL NOT 调用 `ic_union_finalize` logged step
- **AND** `fu_full_process.sh` SHALL NOT 保留独立后置的旧 `feature_union` 步骤

### Requirement: 商品期货主流程步骤日志
系统 SHALL 为商品期货 preprocess 主流程的主要阶段生成独立步骤日志，并在总日志中记录阶段状态。

#### Scenario: 主流程生成步骤日志
- **WHEN** 用户运行 `data_preprocess/script_preprocess/future_upgraded/commodity/main.sh`，且 `SYMBOL=fu`、`TARGET_FREQ=5min`、`START_DATE=2025-11-03`、`END_DATE=2025-11-08`
- **THEN** 系统 SHALL 为 `stitch_main_contract`、`downscale_continuous_by_trading_day`、`cross_section`、`merge`、`concat`、`time_feature`、`merge_clean`、`scale_save`、`dataset_split` 和 `maintenance_margin_dict` 生成独立日志文件
- **AND** 每个步骤日志文件名 SHALL 包含 symbol、target_freq、start_date、end_date 和步骤名
- **AND** 每个步骤日志 SHALL 捕获该步骤的 stdout 和 stderr

#### Scenario: 总日志记录阶段状态
- **WHEN** 商品 preprocess 主流程执行任一主要步骤
- **THEN** 总日志 SHALL 记录该步骤的开始信息和步骤日志路径
- **AND** 当步骤成功完成时，总日志 SHALL 记录该步骤成功完成
- **AND** 当步骤失败时，总日志 SHALL 记录该步骤失败和对应日志路径

### Requirement: 商品期货跨合约训练特征合集
系统 SHALL 支持在所有入选合约完成单合约特征选择和 scale save 后，生成品种级统一 state feature 合集，供需要统一 state feature 列表的独立流程读取。

#### Scenario: 生成品种级 state feature union
- **WHEN** `main_contract_summary.json` 中包含合约 `fu2601` 和 `fu2605`
- **AND** `SCALE_SAVE/fu/fu2601/5min/2026-01-01-2026-04-01/state_features.npy` 包含 `["alpha", "beta"]`
- **AND** `SCALE_SAVE/fu/fu2605/5min/2026-01-01-2026-04-01/state_features.npy` 包含 `["beta", "gamma"]`
- **THEN** 系统 SHALL 写出 `FEATURE_UNION/fu/5min/2026-01-01-2026-04-01/state_features.npy`
- **AND** 该合集 SHALL 包含 `["alpha", "beta", "gamma"]`
- **AND** 系统 SHALL 写出同目录下的 `feature_union_manifest.json`
- **AND** manifest SHALL 包含 `symbol`、`target_freq`、`start_date`、`end_date`、`contracts`、`state_feature_count`、`state_features` 和每个合约的输入 `state_features.npy` 路径

#### Scenario: feature union 顺序稳定
- **WHEN** 系统生成跨合约 state feature union
- **THEN** 系统 SHALL 按 summary 中 `contracts` 的顺序读取每个合约
- **AND** 系统 SHALL 按每个合约 `state_features.npy` 内的原始顺序追加特征
- **AND** 重复 state feature SHALL 只保留第一次出现的位置
- **AND** 多次运行相同输入 SHALL 生成相同顺序的 union feature list

#### Scenario: feature union 缺失合约产物 fail-fast
- **WHEN** summary 中包含合约 `fu2605`
- **AND** `SCALE_SAVE/fu/fu2605/5min/2026-01-01-2026-04-01/state_features.npy` 不存在
- **THEN** 系统 SHALL 报错并停止 feature union 生成
- **AND** 错误信息 SHALL 包含缺失合约 `fu2605` 和缺失的 `state_features.npy` 路径

#### Scenario: full process 不再生成 feature union
- **WHEN** `fu_full_process.sh` 已对 summary 中所有合约完成 `scale_save`
- **THEN** `fu_full_process.sh` SHALL NOT 调用品种级 feature union 生成步骤
- **AND** `fu_full_process.sh` SHALL NOT 调用 `run_commodity_ic_union_finalize`
- **AND** 后续第 9 阶段 dataset split SHALL NOT 依赖 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy`

#### Scenario: validation 检查 feature union
- **WHEN** `validate_features.sh` 验证商品期货输出
- **THEN** 脚本 SHALL 检查 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy`
- **AND** 脚本 SHALL 检查 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/feature_union_manifest.json`

### Requirement: 商品 FineFT 阶段数据集生成
系统 SHALL 从合约级 `SCALE_SAVE` 输出生成商品阶段数据集，保留合约级 train/valid/test 文件，并额外生成品种级纵向合并的 `train.feather`、`valid.feather` 和 `test.feather`。

#### Scenario: 生成合约级阶段数据文件
- **WHEN** summary 中合约 `fu2601` 在 train、valid、test 集合均命中交易日
- **THEN** 系统 SHALL 读取 `SCALE_SAVE/fu/fu2601/5min/{start_date}-{end_date}/df.feather`
- **AND** 系统 SHALL 按该合约命中的交易日过滤并按时间升序输出 `dataset/5min/fu/train/fu2601.feather`
- **AND** 系统 SHALL 输出 `dataset/5min/fu/valid/fu2601.feather`
- **AND** 系统 SHALL 输出 `dataset/5min/fu/test/fu2601.feather`
- **AND** 输出 SHALL 保留输入 feather 的所有列
- **AND** 输出前 SHALL 重置 DataFrame index

#### Scenario: 生成纵向合并阶段大文件
- **WHEN** 商品 dataset split 已写出一个或多个合约级 train、valid 和 test 阶段文件
- **THEN** 系统 SHALL 分别纵向合并所有合约级 train 文件并写出 `dataset/{target_freq}/{symbol}/train.feather`
- **AND** 系统 SHALL 分别纵向合并所有合约级 valid 文件并写出 `dataset/{target_freq}/{symbol}/valid.feather`
- **AND** 系统 SHALL 分别纵向合并所有合约级 test 文件并写出 `dataset/{target_freq}/{symbol}/test.feather`
- **AND** 纵向合并 SHALL 保留输入 feather 的所有列
- **AND** 纵向合并 SHALL NOT 删除合约级 `train/`、`valid/` 或 `test/` 目录

#### Scenario: dataset split 不依赖 state features
- **WHEN** 第 9 阶段 dataset split 生成阶段数据集
- **THEN** 系统 SHALL NOT 要求 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy` 存在
- **AND** 系统 SHALL NOT 读取 `state_features.npy` 来筛选输出列
- **AND** 系统 SHALL NOT 生成或复制 `dataset/{target_freq}/{symbol}/state_features.npy`

#### Scenario: 缺少必要输入 fail-fast
- **WHEN** 某个非空集合需要合约 `fu2601` 的 `df.feather`，但输入文件不存在
- **THEN** 系统 SHALL 报错并停止
- **AND** 错误信息 SHALL 包含缺失合约和缺失路径

#### Scenario: 计划交易日过滤为空 fail-fast
- **WHEN** summary 显示合约 `fu2601` 在 train 集合存在交易日
- **AND** 系统读取 `SCALE_SAVE/fu/fu2601/5min/{start_date}-{end_date}/df.feather`
- **AND** 按该集合交易日过滤后没有任何行
- **THEN** 系统 SHALL 报错并停止
- **AND** 错误信息 SHALL 包含合约、集合名和输入路径

## ADDED Requirements

### Requirement: 商品期货第 9 阶段 dataset split 入口
系统 SHALL 提供 `future_upgraded/9_dataset_split` 阶段入口，并在商品 full process 中于所有合约 `scale_save` 完成后运行该阶段。

#### Scenario: shell stage 激活 finetf 环境
- **WHEN** `data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh` 运行
- **THEN** 脚本 SHALL 激活 `finetf` conda 环境
- **AND** 脚本 SHALL 调用 `operator_futures.dataset_split.dataset_split`
- **AND** 脚本 SHALL 传递 summary 路径、`SCALE_SAVE` 根目录、输出根目录、`symbol`、`target_freq`、`start_date`、`end_date` 和 split ratio 参数

#### Scenario: full process 只运行一次 dataset split
- **WHEN** `main_contract_summary.json` 包含多个合约
- **AND** `fu_full_process.sh` 已为每个合约完成 `scale_save`
- **THEN** `fu_full_process.sh` SHALL 调用一次 `dataset_split`
- **AND** 该调用 SHALL 使用同一次运行的 `summary_path`、`symbol`、`target_freq`、`start_date` 和 `end_date`
- **AND** 该调用 SHALL NOT 绑定单个 `contract`

### Requirement: 商品期货 dataset split manifest
系统 SHALL 为第 9 阶段商品 dataset split 写出 `dataset_split_manifest.json`，描述 split 边界、合约集合归属、输入输出路径、输出行数和跳过原因。

#### Scenario: 写出 split manifest 边界和集合信息
- **WHEN** `operator_futures.dataset_split.dataset_split` 完成边界计算
- **THEN** 系统 SHALL 写出 `dataset/{target_freq}/{symbol}/dataset_split_manifest.json`
- **AND** manifest SHALL 包含 `symbol`、`target_freq`、`split_ratio`、`boundaries` 和 `sets`
- **AND** `split_ratio` SHALL 记录 `{"train": 5, "valid": 3, "test": 2}`
- **AND** `boundaries` SHALL 记录 `start`、`a`、`b`、`c`

#### Scenario: split manifest 记录合约级输入输出
- **WHEN** 合约 `fu2601` 在 train 集合命中至少一个交易日
- **THEN** manifest SHALL 在 `sets.train.contracts` 中记录 `contract=fu2601`
- **AND** 该记录 SHALL 包含命中的 `trading_days`
- **AND** 该记录 SHALL 包含输入 `SCALE_SAVE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather` 路径
- **AND** 该记录 SHALL 包含阶段输出 `dataset/{target_freq}/{symbol}/train/{contract}.feather` 路径
- **AND** 该记录 SHALL 包含 `output_row_count`

#### Scenario: split manifest 记录集合合并输出
- **WHEN** dataset split 写出 `train.feather`、`valid.feather` 和 `test.feather`
- **THEN** manifest SHALL 为每个集合记录顶层 merged output 路径
- **AND** manifest SHALL 为每个集合记录 `contracts_total_count`
- **AND** `contracts_total_count` SHALL 等于该集合内所有合约 `output_row_count` 之和

#### Scenario: split manifest 记录空命中或跳过原因
- **WHEN** 某合约在 valid 集合没有命中任何交易日
- **THEN** 系统 SHALL NOT 写出空的 `valid/{contract}.feather`
- **AND** manifest SHALL 记录该合约在 valid 集合为空命中或被跳过的原因
