## ADDED Requirements

### Requirement: 商品期货多合约 feature selection union
系统 SHALL 将商品期货多合约 feature selection 拆分为 candidate 和 union finalize 两个阶段，确保所有合约使用同一份 union state feature 列表，并为每个合约生成按 union 过滤后的标准 `IC_RESULT` 数据文件。

#### Scenario: 单合约 candidate 阶段不写最终数据文件
- **WHEN** 商品期货合约 `fu2601` 运行 IC candidate 阶段，输入为 `PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/fu/fu2601/5min/2026-01-01-2026-04-01.feather`
- **THEN** 系统 SHALL 写出 `PREPROCESS_DATASET/commodity-futures/IC_RESULT/fu/fu2601/5min/2026-01-01-2026-04-01/state_features_candidate.npy`
- **AND** 系统 SHALL 写出 `ic_window_<window>.json` 和 `correlation.csv`
- **AND** 系统 SHALL NOT 在 candidate 阶段写出标准 `df.feather`
- **AND** 系统 SHALL NOT 在 candidate 阶段写出标准 `state_features.npy`

#### Scenario: union finalize 生成品种级 state features
- **WHEN** `main_contract_summary.json` 包含合约 `fu2601` 和 `fu2605`
- **AND** 两个合约均已生成 `state_features_candidate.npy`
- **THEN** union finalize 阶段 SHALL 按 summary 合约列表读取所有 candidate feature 文件
- **AND** 系统 SHALL 去重合并候选特征并保持稳定顺序
- **AND** 系统 SHALL 写出 `PREPROCESS_DATASET/commodity-futures/FEATURE_UNION/fu/5min/2026-01-01-2026-04-01/state_features.npy`
- **AND** 系统 SHALL 写出 `feature_union_manifest.json`，记录每个合约 candidate 路径、candidate 特征数、union 特征数和最终合约输出路径

#### Scenario: union finalize 生成每个合约过滤后的 IC_RESULT
- **WHEN** union state features 为 `["f1", "f2", "f3"]`
- **AND** 合约 `fu2601` 和 `fu2605` 的 `ALL_FEATURE` 均包含 reward/execution 列和 `f1`、`f2`、`f3`
- **THEN** 系统 SHALL 为每个合约读取对应 `ALL_FEATURE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}.feather`
- **AND** 系统 SHALL 为每个合约写出 `IC_RESULT/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather`
- **AND** 每个合约的 `df.feather` SHALL 包含 `reward_features + union_state_features`
- **AND** 系统 SHALL 为每个合约写出标准 `state_features.npy`
- **AND** 每个合约标准 `state_features.npy` SHALL 与品种级 `FEATURE_UNION/state_features.npy` 内容一致

#### Scenario: union 特征缺列 fail-fast
- **WHEN** union state features 包含 `f3`
- **AND** 合约 `fu2605` 的 `ALL_FEATURE` 不包含 `f3`
- **THEN** union finalize SHALL 报错并停止
- **AND** 错误信息 SHALL 包含合约 `fu2605` 和缺失特征 `f3`
- **AND** 系统 SHALL NOT 静默丢弃 `f3`
- **AND** 系统 SHALL NOT 降级为使用 `fu2605` 自身 candidate 特征

#### Scenario: scale save 继续消费标准 IC_RESULT
- **WHEN** union finalize 已为合约 `fu2601` 写出标准 `IC_RESULT/fu/fu2601/5min/2026-01-01-2026-04-01/df.feather` 和 `state_features.npy`
- **THEN** `scale_save.py` SHALL 按现有接口读取该合约标准 `IC_RESULT` 输出
- **AND** `scale_save.py` SHALL 继续只负责缩放 state features 并保存 `SCALE_SAVE`
- **AND** `scale_save.py` SHALL NOT 负责生成 union、补齐缺列或降级选择合约自身 candidate 特征

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

#### Scenario: full process 按 summary 合约循环并在 union 后执行 scale save
- **WHEN** `main_contract_summary.json` 中包含合约 `fu2601` 和 `fu2605`
- **THEN** `fu_full_process.sh` SHALL 从 summary 读取合约列表
- **AND** `fu_full_process.sh` SHALL 分别为 `fu2601` 和 `fu2605` 调用 `cross_section`、`merge`、`concat`、`time_feature`、`merge_clean` 和 `ic_candidate`
- **AND** 每次合约级调用 SHALL 传递 `--symbols fu --contract <contract>`
- **AND** 所有合约 `ic_candidate` 完成后，`fu_full_process.sh` SHALL 调用品种级 `ic_union_finalize`
- **AND** `ic_union_finalize` 完成后，`fu_full_process.sh` SHALL 分别为每个合约调用 `scale_save`
- **AND** `fu_full_process.sh` SHALL NOT 保留独立后置的旧 `feature_union` 步骤
