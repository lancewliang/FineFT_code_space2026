## ADDED Requirements

### Requirement: 商品 FineFT 数据集装配输入契约
系统 SHALL 将 `FineFT/datahandler/commodity_contract_dataset.py` 收窄为 FineFT 商品数据集装配工具，从 `dataset_split_manifest.json` 获取阶段和合约元数据，并从阶段化 `SCALE_SAVE` 读取真实数据文件。

#### Scenario: 使用 dataset split manifest 代替 main contract summary
- **WHEN** 用户运行 `FineFT/datahandler/commodity_contract_dataset.py`
- **THEN** CLI SHALL 接收 `--dataset_split_manifest_path`
- **AND** 系统 SHALL 从该路径读取 `PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/{target_freq}/{symbol}/dataset_split_manifest.json`
- **AND** 系统 SHALL 使用该 manifest 中的 `sets.train.contracts`、`sets.valid.contracts` 和 `sets.test.contracts` 作为阶段合约列表
- **AND** 系统 SHALL NOT 读取 `main_contract_summary.json` 作为 FineFT 数据集生成输入

#### Scenario: 从阶段化 SCALE_SAVE 读取真实数据
- **WHEN** dataset split manifest 声明 `train` 阶段包含合约 `fu2508`
- **AND** CLI 传入 `--input_root PREPROCESS_DATASET/commodity-futures/SCALE_SAVE`、`--symbol fu` 和 `--target_freq 10min`
- **THEN** 系统 SHALL 读取 `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/10min/train/fu2508.feather`
- **AND** 系统 SHALL NOT 使用 manifest 中的旧 `input_path` 作为真实数据来源
- **AND** 系统 SHALL 对 `valid` 和 `test` 阶段使用同样的 `{input_root}/{symbol}/{target_freq}/{stage}/{contract}.feather` 路径规则

#### Scenario: 停止 FineFT 内部阶段切分
- **WHEN** `commodity_contract_dataset.py` 读取 dataset split manifest 和阶段化 SCALE_SAVE 文件
- **THEN** 系统 SHALL NOT 计算 train、valid 或 test split boundaries
- **AND** 系统 SHALL NOT 使用 `train_ratio`、`valid_ratio` 或 `test_ratio` 决定阶段归属
- **AND** 系统 SHALL NOT 按 `trading_days` 过滤 feather 行
- **AND** 系统 SHALL 直接复制已分阶段的合约文件作为 FineFT 阶段数据

#### Scenario: state features 路径使用训练阶段特征选择结果
- **WHEN** 用户运行 `commodity_contract_dataset.py`
- **THEN** CLI SHALL 接收 `--state_features_path`
- **AND** 推荐调用路径 SHALL 为 `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/{target_freq}/{symbol}/train/state_features.npy`
- **AND** 系统 SHALL 复制该文件到 `dataset/{target_freq}/{symbol}/state_features.npy`
- **AND** CLI SHALL NOT 要求 `--feature_union_path`

#### Scenario: 输入契约不匹配 fail-fast
- **WHEN** `--dataset_split_manifest_path` 不存在、manifest 中 `symbol` 或 `target_freq` 与 CLI 不一致、`--state_features_path` 不存在、state feature 清单为空，或 manifest 声明的 SCALE_SAVE 阶段合约文件不存在
- **THEN** 系统 SHALL 报错并停止 FineFT 数据集生成
- **AND** 错误信息 SHALL 包含缺失或不匹配的路径、stage、contract、symbol 或 target_freq

## MODIFIED Requirements

### Requirement: 商品 FineFT 数据集 manifest
系统 SHALL 为商品 FineFT 多合约数据集写出 `dataset_manifest.json`，描述 dataset split manifest 来源、集合归属、输入路径、输出路径、输出行数、state feature 清单和切片计划。

#### Scenario: 写出 manifest 来源和集合信息
- **WHEN** 商品 FineFT 数据集装配工具完成阶段合约文件复制
- **THEN** 系统 SHALL 写出 `dataset/{target_freq}/{symbol}/dataset_manifest.json`
- **AND** manifest SHALL 包含 `symbol`、`target_freq`、`dataset_split_manifest_path`、`sets` 和 `state_features_path`
- **AND** manifest SHALL NOT 要求 `split_ratio` 或 `boundaries` 字段驱动 FineFT 数据集生成
- **AND** manifest MAY 复制 dataset split manifest 中的 `range`、`trading_days`、`skipped_contracts`、`contracts_total_count` 或其他审计信息

#### Scenario: manifest 记录合约级输入输出
- **WHEN** dataset split manifest 在 `sets.train.contracts` 中声明合约 `fu2508`
- **THEN** FineFT manifest SHALL 在 `sets.train.contracts` 中记录 `contract=fu2508`
- **AND** 该记录 SHALL 包含输入 `SCALE_SAVE/{symbol}/{target_freq}/train/fu2508.feather` 路径
- **AND** 该记录 SHALL 包含阶段输出 `dataset/{target_freq}/{symbol}/train/fu2508.feather` 路径
- **AND** train 集合记录 SHALL 包含该合约贡献的 `train/slice/df_*.feather` 编号计划
- **AND** 如 dataset split manifest 提供 `trading_days` 或 `range`，FineFT manifest SHALL 可保留这些字段用于审计，但 SHALL NOT 使用它们过滤数据

#### Scenario: manifest 记录阶段输出行数
- **WHEN** 商品 FineFT 数据集装配工具写出 `train/fu2508.feather`、`valid/fu2508.feather` 或 `test/fu2508.feather`
- **THEN** 每个包含 `output_path` 的合约记录 SHALL 包含 `output_row_count`
- **AND** `output_row_count` SHALL 等于该 `output_path` feather 文件的实际行数
- **AND** 每个集合 SHALL 包含 `contracts_total_count`
- **AND** `contracts_total_count` SHALL 等于该集合内所有合约 `output_row_count` 之和
- **AND** 系统 SHALL 在 `dataset_manifest.json` 中写出这些行数，使调用方无需读取 feather 文件即可知道单文件和集合总行数

#### Scenario: manifest 记录空命中或跳过原因
- **WHEN** dataset split manifest 中某合约不属于 valid 集合或在 `sets.valid.skipped_contracts` 中记录跳过原因
- **THEN** 系统 SHALL NOT 要求该合约存在 `SCALE_SAVE/{symbol}/{target_freq}/valid/{contract}.feather`
- **AND** FineFT manifest SHALL 保留该合约在 valid 集合为空命中或被跳过的原因

### Requirement: 商品 FineFT 阶段数据集生成
系统 SHALL 从阶段化 `SCALE_SAVE` 输出装配商品 FineFT 阶段数据集，保留合约级 train、valid 和 test 文件。

#### Scenario: 复制合约级阶段数据文件
- **WHEN** dataset split manifest 声明合约 `fu2508` 在 train、valid 和 test 集合均存在
- **THEN** 系统 SHALL 读取 `SCALE_SAVE/fu/10min/train/fu2508.feather`
- **AND** 系统 SHALL 复制并写出 `dataset/10min/fu/train/fu2508.feather`
- **AND** 系统 SHALL 复制并写出 `dataset/10min/fu/valid/fu2508.feather`
- **AND** 系统 SHALL 复制并写出 `dataset/10min/fu/test/fu2508.feather`
- **AND** 输出 SHALL 保留输入 feather 的所有列和行
- **AND** 系统 SHALL NOT 额外写出 `dataset/{target_freq}/{symbol}/train.feather`、`valid.feather` 或 `test.feather`

#### Scenario: 复制训练阶段 state features
- **WHEN** `FEATURE_SELECTION/10min/fu/train/state_features.npy` 存在且非空
- **THEN** 系统 SHALL 将其复制为 `dataset/10min/fu/state_features.npy`
- **AND** 系统 SHALL NOT 要求旧 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy` 存在
- **AND** 系统 SHALL NOT 读取 `state_features.npy` 来筛选阶段 feather 输出列

#### Scenario: 缺少必要 SCALE_SAVE 输入 fail-fast
- **WHEN** dataset split manifest 声明 `train` 阶段需要合约 `fu2508`，但 `SCALE_SAVE/{symbol}/{target_freq}/train/fu2508.feather` 不存在
- **THEN** 系统 SHALL 报错并停止
- **AND** 错误信息 SHALL 包含缺失合约、stage 和缺失路径

#### Scenario: 复制后空数据 fail-fast
- **WHEN** 系统复制某个阶段合约 feather 后读取行数为 0
- **THEN** 系统 SHALL 报错并停止
- **AND** 错误信息 SHALL 包含合约、stage 和输出路径

### Requirement: 商品 FineFT 训练切片生成
系统 SHALL 从商品 train 阶段合约数据生成真正用于低层训练的 `train/slice/df_*.feather` 文件，切片连续编号且不跨合约、不跨 train 阶段文件。

#### Scenario: train slice 连续编号
- **WHEN** `train/fu2508.feather` 和 `train/fu2509.feather` 均可切出训练片段
- **THEN** 系统 SHALL 在 `dataset/{target_freq}/{symbol}/train/slice/` 下写出 `df_0.feather`、`df_1.feather`、`df_2.feather` 等连续编号文件
- **AND** 编号 SHALL 从 0 开始且不跳号
- **AND** manifest SHALL 记录每个 slice 编号对应的 contract、源阶段文件和行范围
- **AND** manifest SHALL 记录每个 slice 输出文件的 `output_row_count`

#### Scenario: train short slice 不丢弃
- **WHEN** 合约 `fu2508` 的 train 阶段数据行数少于 `chunk_length`
- **THEN** 系统 SHALL 仍然写出一个 `train/slice/df_*.feather`
- **AND** 该 slice SHALL 只包含 `fu2508` 的 train 阶段数据
- **AND** manifest SHALL 记录该 slice 的 `output_row_count`
- **WHEN** 合约 train 阶段数据在完整 `chunk_length` 切片后仍有不足 `chunk_length` 的尾部行
- **THEN** 系统 SHALL 将该尾部行写出为短 slice
- **AND** 系统 SHALL NOT 为补齐短 slice 从其他合约、valid 或 test 阶段追加数据

#### Scenario: train slice 不跨合约
- **WHEN** 一个训练 slice 从 `train/fu2508.feather` 生成
- **THEN** 该 slice SHALL 只包含 `fu2508` 的行
- **AND** 该 slice SHALL NOT 包含任何其他合约的行

#### Scenario: early_stop 不跨 train 阶段文件
- **WHEN** `chunk_length` 后追加 `early_stop` 行会越过同一合约的 train 阶段数据末尾
- **THEN** 系统 SHALL 将 slice 截断在同一合约 train 阶段数据内
- **AND** 系统 SHALL NOT 从 valid 或 test 阶段追加任何行
- **AND** 如果截断后 slice 为空，系统 SHALL 跳过该 slice

### Requirement: 商品 FineFT valid 动态切片生成
系统 SHALL 通过商品 data handler shell 对商品 valid 阶段数据逐合约执行市场动态切片，输出 `valid/<contract>/label_*/df_*.feather`，并保证动态片段不跨合约。

#### Scenario: 数据集工具不调用 slice model
- **WHEN** 商品 FineFT 数据集装配工具生成 `dataset_manifest.json`、阶段数据和 train slice
- **THEN** `commodity_contract_dataset.py` SHALL NOT import or call `slice_model.py`
- **AND** `commodity_contract_dataset.py` SHALL NOT write valid dynamic slice files
- **AND** valid 动态切片 SHALL 留给商品 data handler shell 的后续独立阶段执行

#### Scenario: shell 逐合约调度 valid 动态切片
- **WHEN** `valid/fu2508.feather` 和 `valid/fu2509.feather` 均存在
- **THEN** 商品 data handler shell SHALL 分别对两个合约文件调用 `FineFT/datahandler/slice_model.py`
- **AND** 每次调用的 `--data_path` SHALL 指向 `dataset/{target_freq}/{symbol}/valid/{contract}.feather`
- **AND** 系统 SHALL NOT 在切片前把两个合约拼接成一个连续 valid DataFrame
- **AND** 输出的每个 `valid/<contract>/label_*/df_*.feather` SHALL 只包含单一合约的数据

#### Scenario: valid 动态切片保持 label 目录格式
- **WHEN** 动态标签数量为 5
- **THEN** 系统 SHALL 在 `dataset/{target_freq}/{symbol}/valid/<contract>/label_0` 到 `label_4` 下写出动态片段文件
- **AND** 文件编号 SHALL NOT 覆盖其他合约产生的片段
- **AND** manifest SHALL 记录每个 valid 动态片段对应的 contract、label 和输出路径

#### Scenario: valid processed 文件按合约隔离
- **WHEN** 商品 data handler shell 分别处理 `valid/fu2508.feather` 和 `valid/fu2509.feather`
- **THEN** `slice_model.py` SHALL 写出 `valid/processed/valid_processed_fu2508.feather`
- **AND** `slice_model.py` SHALL 写出 `valid/processed/valid_processed_fu2509.feather`
- **AND** 系统 SHALL NOT 对多个合约共用 `valid/valid_processed.feather`
- **AND** 系统 SHALL NOT 将不同合约的 label 片段写到同一个 `valid/label_*` 目录

#### Scenario: valid 数据不足时不跨合约补齐
- **WHEN** 某合约 valid 数据不足以执行动态切片
- **THEN** 系统 SHALL 跳过该合约的 valid 动态切片并在 manifest 中记录原因
- **AND** 系统 SHALL NOT 将其他合约数据拼接进该合约 valid 数据以满足最小长度

#### Scenario: valid slope 标签支持少量 segment
- **WHEN** 某合约 valid 动态切片在合并后只剩少量 segment
- **THEN** slope 标签阈值计算 SHALL NOT 因 segment 数量少而抛出 `IndexError`
- **AND** 系统 SHALL 使用该合约已有 segment 生成动态标签
- **AND** 系统 SHALL NOT 拼接其他合约数据来补齐 segment 数量

#### Scenario: valid 动态切片 manifest 记录合约和 label 行数
- **WHEN** `slice_model.py` 为合约 `fu2508` 写出 `valid/fu2508/label_0/df_0.feather`
- **THEN** 系统 SHALL 更新 `valid/slice_manifest.json`
- **AND** manifest SHALL 在合约视角记录 `fu2508` 每个非空 label 的文件路径、文件行数、文件数和总行数
- **AND** manifest SHALL 在 label 视角记录每个非空 label 跨合约的文件路径、合约、文件行数、文件数和总行数
- **AND** manifest SHALL NOT 记录没有生成文件的空 label
- **AND** 多个合约顺序调用 `slice_model.py` SHALL 累积更新 manifest，且同一合约重跑 SHALL 替换该合约旧记录

### Requirement: 商品 FineFT data handler 脚本入口
系统 SHALL 直接升级现有商品 data handler 脚本，让燃料油和铝的 FineFT 商品数据准备使用 dataset split manifest 和阶段化 SCALE_SAVE 的多合约装配流程。

#### Scenario: 燃料油 data handler 调用新工具
- **WHEN** 用户运行 `FineFT/script/data/commodity_data_handler_fu.sh`
- **THEN** 脚本 SHALL 激活 `finetf` conda 环境
- **AND** 脚本 SHALL 调用商品多合约数据集工具
- **AND** 脚本 SHALL 传递 `--symbol fu`、`--dataset_split_manifest_path PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/{target_freq}/fu/dataset_split_manifest.json`、`--input_root PREPROCESS_DATASET/commodity-futures/SCALE_SAVE`、`--state_features_path PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/{target_freq}/fu/train/state_features.npy`、输出根目录、`target_freq`、`chunk_length` 和 `early_stop`
- **AND** 脚本 SHALL 在数据集工具完成后逐合约调用 `FineFT/datahandler/slice_model.py --data_path dataset/{target_freq}/fu/valid/{contract}.feather --timestamp timestamp`
- **AND** 脚本 SHALL NOT 调用旧的 `FineFT/datahandler/preprocess_data.py --trading_pair fu`
- **AND** 脚本 SHALL NOT 调用旧的 `FineFT/datahandler/slice_model.py --data_path dataset/fu/valid.feather`

#### Scenario: 铝 data handler 调用新工具
- **WHEN** 用户运行 `FineFT/script/data/commodity_data_handler_al.sh`
- **THEN** 脚本 SHALL 激活 `finetf` conda 环境
- **AND** 脚本 SHALL 调用商品多合约数据集工具
- **AND** 脚本 SHALL 传递 `--symbol al`、`--dataset_split_manifest_path PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/{target_freq}/al/dataset_split_manifest.json`、`--input_root PREPROCESS_DATASET/commodity-futures/SCALE_SAVE`、`--state_features_path PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/{target_freq}/al/train/state_features.npy`、输出根目录、`target_freq`、`chunk_length` 和 `early_stop`
- **AND** 脚本 SHALL 在数据集工具完成后逐合约调用 `FineFT/datahandler/slice_model.py --data_path dataset/{target_freq}/al/valid/{contract}.feather --timestamp timestamp`
- **AND** 脚本 SHALL NOT 调用旧的 `FineFT/datahandler/preprocess_data.py --trading_pair al`
- **AND** 脚本 SHALL NOT 调用旧的 `FineFT/datahandler/slice_model.py --data_path dataset/al/valid.feather`

#### Scenario: VAE 数据生成读取新 valid/test 结构
- **WHEN** 商品 data handler 完成多合约阶段数据和 valid 动态切片
- **THEN** 后续 VAE 数据生成 SHALL 从 `valid/<contract>/label_*/df_*.feather` 读取训练用动态片段
- **AND** 后续 VAE 数据生成 SHALL 写出 `VAE_data/<contract>/label_*.npy`
- **AND** 后续 VAE 数据生成 SHALL NOT 将不同合约的同一 label 聚合为单个 `VAE_data/label_*.npy`
- **AND** 后续 VAE 数据生成 SHALL 从 `test/{contract}.feather` 读取测试特征数组
- **AND** 后续 VAE 数据生成 SHALL 写出 `VAE_data/test/test_<contract>.npy`

## REMOVED Requirements

### Requirement: 商品 FineFT 多合约数据集边界计算
系统 SHALL 从商品主力合约 summary 计算 FineFT 商品多合约数据集的全局 train、valid、test 日期边界，保证三个集合时序递进且日期不重叠。

#### Scenario: 按 5:3:2 计算全局边界
- **WHEN** `main_contract_summary.json` 包含多个合约及其有效 `trading_days[].date`
- **THEN** 系统 SHALL 对所有合约有效交易日取去重升序并集作为全局交易日轴
- **AND** 系统 SHALL 按 `train:valid:test = 5:3:2` 计算 `start`、`a`、`b`、`c`
- **AND** 系统 SHALL 生成左闭右开的集合范围：`train=[start,a)`、`valid=[a,b)`、`test=[b,c)`
- **AND** `start < a < b < c` SHALL 成立
- **AND** 三个集合的日期范围 SHALL 不重叠

#### Scenario: 合约集合归属由全局边界求交决定
- **WHEN** 合约 `fu2601` 的有效交易日跨越全局边界 `a` 和 `b`
- **THEN** 系统 SHALL 使用该合约有效交易日分别与 `[start,a)`、`[a,b)`、`[b,c)` 求交
- **AND** 合约同一个交易日 SHALL 最多归属于一个集合
- **AND** 系统 SHALL NOT 按行数或拼接后的连续行情重新计算该合约的集合归属

#### Scenario: 无法形成有效边界 fail-fast
- **WHEN** summary 中有效交易日不足以形成非空 train、valid 和 test 集合
- **THEN** 系统 SHALL 报错并停止数据集生成
- **AND** 错误信息 SHALL 说明无法满足 `start < a < b < c`
