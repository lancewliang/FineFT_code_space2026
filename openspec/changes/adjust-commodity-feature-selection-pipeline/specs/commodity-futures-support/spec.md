## ADDED Requirements

### Requirement: 商品期货 split 后多合约特征选择流水线
系统 SHALL 在商品期货 dataset split 之后执行 train 多合约特征评估与筛选，并执行 valid 多合约评估与报告；后续 scale-save SHALL 只使用 train 产生的最终特征清单。

#### Scenario: train 阶段从 split train 文件生成最终特征清单
- **WHEN** `dataset_split` 已写出 `PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/train/fu2601.feather` 和 `fu2605.feather`
- **THEN** train feature selection SHALL 读取该 split train 目录下的合约级 feather 文件
- **AND** train feature selection SHALL 对所有 state 特征计算每合约 `Permutation Importance`、`CatBoost Importance`、`IC`、`RankIC` 和 `Sharpe`
- **AND** train feature selection SHALL 默认按窗口 `[1, 6, 12]` 计算指标
- **AND** train feature selection SHALL 将每合约明细写入 `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train/per_contract/`
- **AND** 每合约明细 SHALL 包含 `window` 字段
- **AND** train feature selection SHALL 汇总每个指标的 `Mean`、`Std` 和 `Median`
- **AND** train feature selection SHALL 依次执行 `Hard Filter`、`Stability Filter`、`Composite Score` 和 `Correlation Filter`
- **AND** train feature selection SHALL 写出 `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/train/state_features.npy`
- **AND** train feature selection SHALL NOT 写出 `state_features_candidate.npy` 作为下游约定文件
- **AND** train feature selection SHALL 写出 `feature_selection_manifest.json`，记录输入 split 路径、合约、指标明细路径、汇总路径、筛选阶段结果、最终特征数、`windows_list` 和 `composite_drop_ratio`

#### Scenario: valid 阶段只使用 train 特征清单做评估报告
- **WHEN** train feature selection 已写出 `FEATURE_SELECTION/5min/fu/train/state_features.npy`
- **AND** `dataset_split` 已写出 `PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/5min/fu/valid/fu2601.feather`
- **THEN** valid feature selection SHALL 读取 split valid 目录下的合约级 feather 文件
- **AND** valid feature selection SHALL 仅对 train `state_features.npy` 中的 state 特征计算 `Permutation Importance`、`CatBoost Importance`、`IC`、`RankIC` 和 `Sharpe`
- **AND** valid feature selection SHALL 默认按窗口 `[1, 6, 12]` 计算指标
- **AND** valid feature selection SHALL 汇总每个指标的 `Mean`、`Std` 和 `Median`
- **AND** valid feature selection SHALL NOT 执行 `Hard Filter`、`Stability Filter`、`Composite Score` 或 `Correlation Filter`
- **AND** valid feature selection SHALL NOT 写出下游采用的 `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/5min/fu/valid/state_features.npy`
- **AND** valid feature selection SHALL 写出 `feature_selection_manifest.json`，记录 train feature list 路径、输入 split 路径、指标明细路径、汇总路径、评估特征数和 `windows_list`

#### Scenario: 指标目标和多窗口口径
- **WHEN** feature selection 计算窗口 `window` 的任一指标
- **THEN** 系统 SHALL 使用 `mark_price.shift(-window) - mark_price` 作为未来收益 target
- **AND** 系统 SHALL 裁掉最后 `window` 行，使 feature values 与 target 长度一致
- **AND** 系统 SHALL 默认对 `windows_list=[1, 6, 12]` 中每个窗口分别生成每合约指标明细
- **AND** 系统 SHALL 在每合约指标明细中记录对应 `window`
- **AND** 系统 SHALL 按 feature 汇总所有合约和窗口上的指标 `Mean`、`Std` 和 `Median`

#### Scenario: IC 计算沿用原始 ic_correlation 口径
- **WHEN** feature selection 计算 state feature 的 `IC`
- **THEN** 系统 SHALL 对 feature 和 target 执行 pairwise NaN 过滤
- **AND** 当过滤后 feature 或 target 样本数小于 2 时，`IC` SHALL 为 `np.nan`
- **AND** 当过滤后 feature 或 target 标准差为 0 时，`IC` SHALL 为 `np.nan`
- **AND** 其他情况下 `IC` SHALL 为 feature 与 target 的 Pearson correlation

#### Scenario: RankIC 计算沿用原始 rank_ic_correlation 口径
- **WHEN** feature selection 计算 state feature 的 `RankIC`
- **THEN** 系统 SHALL 先检查原始 feature 和 target
- **AND** 当原始 feature 或 target 为空时，`RankIC` SHALL 为 `0.0`
- **AND** 当原始 feature 或 target 的 `np.nanstd` 为 0 时，`RankIC` SHALL 为 `0.0`
- **AND** 其他情况下系统 SHALL 使用 `np.argsort(np.argsort(values))` 生成 ranks
- **AND** 系统 SHALL 计算 feature ranks 与 target ranks 的 Pearson correlation
- **AND** 系统 SHALL 将 NaN、正无穷和负无穷结果转换为 `0.0`

#### Scenario: CatBoost Importance 沿用原始 catbooost 口径
- **WHEN** feature selection 计算 state feature 的 `CatBoost Importance`
- **THEN** 系统 SHALL 使用 `CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, loss_function="MAE", task_type="GPU", random_seed=42)`
- **AND** 系统 SHALL 使用同一窗口下的 feature matrix 和 target 构造 `train_pool` 与 `test_pool`
- **AND** 系统 SHALL 调用 `model.fit(train_pool, eval_set=test_pool, verbose=100)`
- **AND** 系统 SHALL 从 `model.get_feature_importance(train_pool)` 读取 feature importance
- **AND** 系统 SHALL NOT 在 CatBoost 不可用时降级为 `abs(IC)` 或其他替代指标

#### Scenario: Sharpe 使用单特征伪策略收益
- **WHEN** feature selection 计算 state feature `alpha` 的 Sharpe 指标
- **THEN** 系统 SHALL 在当前输入数据内对 `alpha` 执行列内 z-score
- **AND** 系统 SHALL 将 z-score 后的 `alpha` 与未来收益相乘得到伪收益序列
- **AND** 系统 SHALL 根据该伪收益序列计算 Sharpe
- **AND** 系统 SHALL NOT 使用跨 train 和 valid 的联合统计量计算该 Sharpe

#### Scenario: Permutation Importance 使用 IC 损失口径
- **WHEN** feature selection 计算 state feature 的 `Permutation Importance`
- **THEN** 系统 SHALL 以 `abs(IC(feature, target))` 作为 baseline
- **AND** 系统 SHALL 对 feature values 执行确定性 one-step roll 得到 shuffled feature
- **AND** 系统 SHALL 以 `max(baseline - abs(IC(shuffled_feature, target)), 0.0)` 作为 `Permutation Importance`
- **AND** 当任一 IC 结果为 NaN 时，系统 SHALL 在该差值计算中按 `0.0` 处理该 IC 分数

#### Scenario: Composite Score 按优先级删除低分因子
- **WHEN** feature selection 完成 `Hard Filter` 和 `Stability Filter`
- **THEN** `Hard Filter` SHALL 保留 `abs(RankIC_Mean) >= min_abs_ic` 的 features
- **AND** `Hard Filter` SHALL NOT 使用 `abs(IC_Mean)` 作为第一步硬过滤依据
- **AND** `Stability Filter` SHALL 保留 `IC_Std <= max_metric_std` 的 features
- **AND** `Composite Score` SHALL 先按 `abs(RankIC_Mean)` 降序排序
- **AND** 当 `abs(RankIC_Mean)` 相同时，`Composite Score` SHALL 按 `abs(Sharpe_Mean) + Permutation Importance_Mean` 降序排序
- **AND** 当存在 `SHAP Importance_Mean` 时，系统 SHALL 将其加入第二优先级分数
- **AND** 当前两级分数相同时，`Composite Score` SHALL 按 `CatBoost Importance_Mean` 降序排序
- **AND** 系统 SHALL 删除排序后底部 `composite_drop_ratio` 的 features，默认比例为 `0.1`
- **AND** 系统 SHALL 至少保留 1 个 feature
- **AND** 系统 SHALL 在 `feature_selection_manifest.json` 的 `filter_results` 中记录 `Composite Score` 保留列表和 `Composite Score Dropped` 删除列表
- **AND** `Correlation Filter` SHALL 在 Composite Score 删除后执行

#### Scenario: scale-save 使用训练集特征清单处理 split 阶段文件
- **WHEN** train feature selection 已得到最终 `FEATURE_SELECTION/5min/fu/train/state_features.npy`
- **AND** `SPLIT-TRAIN-VALID-TEST/5min/fu/train/fu2601.feather` 存在
- **AND** `SPLIT-TRAIN-VALID-TEST/5min/fu/valid/fu2601.feather` 不存在
- **THEN** scale-save SHALL 使用 train `state_features.npy` 作为 state feature 清单
- **AND** scale-save SHALL 处理存在的 `train/fu2601.feather`
- **AND** scale-save SHALL 跳过缺失的 `valid/fu2601.feather` 并记录 contract 和 stage
- **AND** scale-save SHALL 继续处理同一合约或其他合约的其他存在阶段
- **AND** scale-save SHALL NOT 使用 valid 阶段产生的特征清单

#### Scenario: scale-save 输出只包含训练集选中特征
- **WHEN** scale-save 处理任一存在的 split 阶段合约文件
- **THEN** scale-save SHALL 写出 `SCALE_SAVE/fu/fu2601/5min/{stage}/{start_date}-{end_date}/df.feather`
- **AND** 输出 `df.feather` SHALL 包含商品 reward/execution 列、train `state_features.npy` 中的 state features 和 `symbol`
- **AND** 输出 `state_features.npy` SHALL 与 train feature selection 产生的 `state_features.npy` 一致
- **AND** 系统 SHALL NOT 将未入选 state features 写入 scale-save 输出 `df.feather`

#### Scenario: 特征选择 fail-fast
- **WHEN** train 或 valid split 输入目录不存在、没有合约 feather、train feature universe 为空、train 筛选结果为空或 required feature column 缺失
- **THEN** feature selection SHALL 报错并停止当前阶段
- **AND** 错误信息 SHALL 包含阶段名、缺失或为空的资源路径，以及相关合约或特征名
- **AND** 系统 SHALL NOT 静默跳过该合约
- **AND** 系统 SHALL NOT 写出下游可消费的 train `state_features.npy`

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

#### Scenario: full process 在 dataset split 后执行特征选择和 scale save
- **WHEN** `main_contract_summary.json` 中包含合约 `fu2601` 和 `fu2605`
- **THEN** `fu_full_process.sh` SHALL 从 summary 读取合约列表
- **AND** `fu_full_process.sh` SHALL 分别为 `fu2601` 和 `fu2605` 调用 `cross_section`、`merge`、`concat`、`time_feature` 和 `merge_clean`
- **AND** 每次合约级调用 SHALL 传递 `--symbols fu --contract <contract>`
- **AND** 所有合约 `merge_clean` 完成后，`fu_full_process.sh` SHALL 只调用一次 `dataset_split`
- **AND** `dataset_split` 完成后，`fu_full_process.sh` SHALL 调用 `feature_selection_train`
- **AND** `feature_selection_train` 完成后，`fu_full_process.sh` SHALL 调用 `feature_selection_valid`
- **AND** `feature_selection_valid` 完成后，`fu_full_process.sh` SHALL 对每个合约调用 `scale_save`
- **AND** 每个 `scale_save` SHALL 使用 `FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy`
- **AND** 每个 `scale_save` SHALL 读取 `SPLIT-TRAIN-VALID-TEST/{target_freq}/{symbol}/{train|valid|test}/{contract}.feather` 中存在的 split 阶段合约文件
- **AND** `maintenance_margin_dict` SHALL 在全部合约 `scale_save` 完成后执行
- **AND** `fu_full_process.sh` SHALL NOT 在合约循环内的 `merge_clean` 后立即调用 `scale_save`
- **AND** `fu_full_process.sh` SHALL NOT 使用旧 `IC_RESULT` 作为本次商品特征评估输入源

### Requirement: 商品期货第 9 阶段 dataset split 入口
系统 SHALL 提供 `future_upgraded/9_dataset_split` 阶段入口，并在商品 full process 中于所有合约 `merge_clean` 完成后运行该阶段。

#### Scenario: shell stage 激活 finetf 环境
- **WHEN** `data_preprocess/script_preprocess/future_upgraded/9_dataset_split/dataset_split.sh` 运行
- **THEN** 脚本 SHALL 激活 `finetf` conda 环境
- **AND** 脚本 SHALL 调用 `operator_futures.dataset_split.dataset_split`
- **AND** 脚本 SHALL 传递 summary 路径、`ALL_FEATURE` 根目录、输出根目录、`symbol`、`target_freq`、`start_date`、`end_date` 和 split ratio 参数

#### Scenario: full process 只运行一次 dataset split
- **WHEN** `main_contract_summary.json` 包含多个合约
- **AND** `fu_full_process.sh` 已为每个合约完成 `merge_clean`
- **THEN** `fu_full_process.sh` SHALL 调用一次 `dataset_split`
- **AND** 该调用 SHALL 使用同一次运行的 `summary_path`、`symbol`、`target_freq`、`start_date` 和 `end_date`
- **AND** 该调用 SHALL NOT 绑定单个 `contract`
- **AND** 该调用 SHALL NOT 读取 `SCALE_SAVE` 作为输入根目录

### Requirement: 商品期货主流程步骤日志
系统 SHALL 为商品期货 preprocess 主流程的主要阶段生成独立步骤日志，并在总日志中记录阶段状态。

#### Scenario: 主流程生成步骤日志
- **WHEN** 用户运行 `data_preprocess/script_preprocess/future_upgraded/commodity/main.sh`，且 `SYMBOL=fu`、`TARGET_FREQ=5min`、`START_DATE=2025-11-03`、`END_DATE=2025-11-08`
- **THEN** 系统 SHALL 为 `stitch_main_contract`、`downscale_continuous_by_trading_day`、`cross_section`、`merge`、`concat`、`time_feature`、`merge_clean`、`dataset_split`、`feature_selection_train`、`feature_selection_valid`、`scale_save` 和 `maintenance_margin_dict` 生成独立日志文件
- **AND** 每个步骤日志文件名 SHALL 包含 symbol、target_freq、start_date、end_date 和步骤名
- **AND** 每个步骤日志 SHALL 捕获该步骤的 stdout 和 stderr

#### Scenario: 总日志记录阶段状态
- **WHEN** 商品 preprocess 主流程执行任一主要步骤
- **THEN** 总日志 SHALL 记录该步骤的开始信息和步骤日志路径
- **AND** 当步骤成功完成时，总日志 SHALL 记录该步骤成功完成
- **AND** 当步骤失败时，总日志 SHALL 记录该步骤失败和对应日志路径

### Requirement: 商品期货按合约生成因子文件
系统 SHALL 在商品期货多合约流程中按具体合约生成独立因子文件，并在未传 contract 时保留共享脚本旧路径行为。

#### Scenario: cross-section 按 contract 读写日文件
- **WHEN** `cross_section/create_feature.py` 以 `--symbols fu --contract fu2601 --target_freq 5min --date 2026-01-05` 运行
- **THEN** 系统 SHALL 从 `BASE_FEATURE/fu/fu2601/5min/2026-01-05.feather` 和 `DOWNSCALE_ORDERBOOK_25/fu/fu2601/5min/2026-01-05.feather` 读取输入
- **AND** 系统 SHALL 写出 `CROSS_SECTION/KLINE_FEATURE/fu/fu2601/5min/2026-01-05.feather`
- **AND** 系统 SHALL 写出 `CROSS_SECTION/QUOTES_FEATURE/fu/fu2601/5min/2026-01-05.feather`
- **AND** 系统 SHALL 写出 `CROSS_SECTION/SNAPSHOT_FEATURE/fu/fu2601/5min/2026-01-05.feather`

#### Scenario: merge 按 contract 读写日文件
- **WHEN** `merge_concat/merge.py` 以 `--symbols fu --contract fu2601 --target_freq 5min --date 2026-01-05` 运行
- **THEN** 系统 SHALL 从 downscale 和 cross-section 的 `fu/fu2601/5min` 日文件读取输入
- **AND** 系统 SHALL 写出 `MERGE_CONCAT/MERGED_FEATURE/fu/fu2601/5min/CONCURRENT_FEATURE/2026-01-05.feather`
- **AND** 系统 SHALL 写出 `MERGE_CONCAT/MERGED_FEATURE/fu/fu2601/5min/FUTURE_FEATURE/2026-01-05.feather`

#### Scenario: concat 按 contract 生成日期范围文件
- **WHEN** `merge_concat/concat.py` 以 `--symbols fu --contract fu2601 --target_freq 5min --start_date 2026-01-01 --end_date 2026-04-01` 运行
- **THEN** 系统 SHALL 从 `MERGE_CONCAT/MERGED_FEATURE/fu/fu2601/5min/...` 读取日文件
- **AND** 系统 SHALL 写出 `MERGE_CONCAT/CONCAT_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather`

#### Scenario: time feature 按 contract 生成日期范围文件
- **WHEN** `time_operator/create_feature_multi_processing.py` 以 `--symbols fu --contract fu2601 --target_freq 5min --start_date 2026-01-01 --end_date 2026-04-01` 运行
- **THEN** 系统 SHALL 从 `MERGE_CONCAT/CONCAT_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather` 读取输入
- **AND** 系统 SHALL 写出 `TIME_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather`

#### Scenario: merge clean 按 contract 生成 all feature
- **WHEN** `merge_all/merge_clean.py` 以 `--symbols fu --contract fu2601 --target_freq 5min --start_date 2026-01-01 --end_date 2026-04-01` 运行
- **THEN** 系统 SHALL 从 `MERGE_CONCAT/CONCAT_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather` 和 `TIME_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather` 读取输入
- **AND** 系统 SHALL 写出 `ALL_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather`

#### Scenario: feature selection 和 scale save 按 contract 生成日期范围目录
- **WHEN** 商品 full process 完成 split 后 feature selection
- **THEN** feature selection SHALL 读取 `SPLIT-TRAIN-VALID-TEST/{target_freq}/{symbol}/{train|valid}/{contract}.feather`
- **AND** feature selection SHALL 写出 `FEATURE_SELECTION/{target_freq}/{symbol}/{train|valid}/`
- **AND** train feature selection SHALL 写出 `FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy`
- **AND** valid feature selection SHALL 只写评估明细、汇总统计和 manifest/report
- **AND** scale save SHALL 读取 `SPLIT-TRAIN-VALID-TEST/{target_freq}/{symbol}/{train|valid|test}/{contract}.feather`
- **AND** scale save SHALL 使用 `FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy`
- **AND** scale save SHALL 写出 `SCALE_SAVE/{symbol}/{contract}/{target_freq}/{stage}/{start_date}-{end_date}/`

#### Scenario: 未传 contract 时保留旧路径
- **WHEN** 共享 operator-futures 脚本未传入 `--contract`
- **THEN** 系统 SHALL 继续读写现有 `{symbol}/{target_freq}` 路径
- **AND** 系统 SHALL NOT 要求非商品期货或旧调用方提供 contract 参数

#### Scenario: 多合约日志和 skip 检查包含 contract
- **WHEN** 商品 full process 对多个合约运行后续阶段
- **THEN** 步骤日志文件名、skip 消息和输出存在性检查 SHALL 包含 `symbol` 和 `contract`
- **AND** 一个合约的日志或输出 SHALL NOT 覆盖另一个合约的日志或输出

### Requirement: 商品期货 Polars 预处理兼容性
系统 SHALL 将 `data_preprocess/operator_futures/commodity` 商品期货核心预处理迁移到 Polars，并保持既有商品期货数据契约。

#### Scenario: 主力合约 summary 输出兼容
- **WHEN** 商品期货主力合约 summary 生成读取本地五档 CSV 文件
- **THEN** 系统使用 Polars 处理 CSV 读取、成交量计算、合格合约筛选、月度 top 2 选择和 summary 写入
- **AND** summary SHALL 保留 `TradingDay` 日归属和 `ActionDay + UpdateTime` 事件时间戳语义所需的源文件信息
- **AND** summary SHALL 提供后续 downscale 所需的 contract、date 和 source_file 明细

#### Scenario: 商品 downscale 输出兼容
- **WHEN** 商品期货 summary 源文件运行单日或按合约下采样
- **THEN** 系统使用 Polars 生成 derivative reference、五档 orderbook、base features 和 quote features
- **AND** depth=5 输出不合成第 6 到第 25 档
- **AND** `LastPrice` 回退、funding 兼容列、Volume/Turnover 差分、tick rule 估计方向、右闭窗口聚合和 fail-fast 校验语义保持不变

#### Scenario: 商品 market_type 分支兼容
- **WHEN** `cross_section/create_feature.py`、`scale_describe_save/scale_save.py` 或 split 后 multi-contract feature selection 以 `market_type=commodity_futures` 运行
- **THEN** 商品 reward/execution manifest、depth-aware feature generation、funding 关闭特征处理和 feature selection target 语义保持不变
- **AND** 输出列集合和列顺序继续满足商品期货现有 tests 和 downstream readers
