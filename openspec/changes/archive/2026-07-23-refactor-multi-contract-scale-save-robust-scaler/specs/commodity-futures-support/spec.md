## MODIFIED Requirements

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
- **THEN** `muti_contract_scale_save.py` SHALL 使用 train `state_features.npy` 作为 state feature 清单
- **AND** `muti_contract_scale_save.py` SHALL 只从 train split 全量拟合 robust scaler，一次生成整套 `center`、`scale`、`fallback` 和 `clip` 参数
- **AND** `muti_contract_scale_save.py` SHALL 将同一套 scaler 参数应用到 train、valid 和 test split 文件
- **AND** `muti_contract_scale_save.py` SHALL NOT 在 valid 或 test split 上重新拟合 scaler
- **AND** `muti_contract_scale_save.py` SHALL 处理存在的 `train/fu2601.feather`
- **AND** `muti_contract_scale_save.py` SHALL NOT 要求为缺失的 `valid/fu2601.feather` 生成输出
- **AND** `muti_contract_scale_save.py` SHALL 继续处理扫描到的其他存在阶段合约文件
- **AND** `muti_contract_scale_save.py` SHALL NOT 使用 valid 阶段产生的特征清单
- **AND** `muti_contract_scale_save.py` SHALL 写出 `SCALE_SAVE/fu/5min/scaler_manifest.json`
- **AND** `muti_contract_scale_save.py` SHALL 写出 `SCALE_SAVE/fu/5min/scale_diagnostics.csv`

#### Scenario: scale-save 输出只包含训练集选中特征
- **WHEN** `muti_contract_scale_save.py` 处理任一存在的 split 阶段合约文件
- **THEN** `muti_contract_scale_save.py` SHALL 写出 `SCALE_SAVE/fu/5min/{stage}/fu2601.feather`
- **AND** `muti_contract_scale_save.py` SHALL 同步写出 `SCALE_SAVE/fu/5min/{stage}/fu2601.csv`
- **AND** 输出 feather 和 csv SHALL 包含商品 reward/execution 列、train `state_features.npy` 中的 state features 和 `symbol`
- **AND** 系统 SHALL 将 state features 按 train-only robust scaler 进行标准化并默认裁剪到 `[-20, 20]`
- **AND** 系统 SHALL NOT 将未入选 state features 写入 scale-save 输出 feather 或 csv

#### Scenario: split-stage robust scaler fail-fast
- **WHEN** train split 输入目录不存在、没有合约 feather、train feature universe 为空、train 筛选结果为空、required feature column 缺失、clip bounds 无效或拟合统计量非有限
- **THEN** `muti_contract_scale_save.py` SHALL 报错并停止当前阶段
- **AND** 错误信息 SHALL 包含阶段名、缺失或为空的资源路径，以及相关合约或特征名
- **AND** 系统 SHALL NOT 静默跳过该合约
- **AND** 系统 SHALL NOT 写出下游可消费的 train `state_features.npy`

#### Scenario: 特征选择 fail-fast
- **WHEN** train 或 valid split 输入目录不存在、没有合约 feather、train feature universe 为空、train 筛选结果为空或 required feature column 缺失
- **THEN** feature selection SHALL 报错并停止当前阶段
- **AND** 错误信息 SHALL 包含阶段名、缺失或为空的资源路径，以及相关合约或特征名
- **AND** 系统 SHALL NOT 静默跳过该合约
- **AND** 系统 SHALL NOT 写出下游可消费的 train `state_features.npy`
