# adjust-commodity-feature-selection-pipeline

## 背景与目标

当前 `fu_full_process.sh` 在 `dataset_split` 之后没有独立的特征评估与筛选阶段，`scale_save` 也仍然依赖旧的前置产物。现在需要把商品期货特征选择改成一条更清晰的流水线：

1. 先按合约切分出 `train/valid/test`
2. 对 `train` 做特征评估与筛选，产出唯一的最终训练特征清单
3. 对 `valid` 只做特征评估和报告，不再做二次筛选
4. 最后由 `scale_save` 使用训练集产生的特征清单处理 split 后各阶段数据

目标是把“特征统计评估”和“特征筛选”从旧的 `IC_RESULT / SCALE_SAVE` 步骤中拆出来，统一沉淀到 `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/${target_freq}`，同时保留原有 `SCALE_SAVE` 作为最终缩放输出目录。验证集评估报告只用于观察训练集特征在 valid 阶段的表现，不参与后续特征清单决策。

## 用户场景

### 场景 1：训练集特征评估与筛选
- 用户运行商品期货全流程后，`dataset_split` 已生成 `SPLIT-TRAIN-VALID-TEST/${target_freq}/train/*.feather`
- 系统读取 `train/*.feather` 中所有 state 特征
- 系统按合约计算 `Permutation Importance`、`CatBoost Importance`、`IC`、`RankIC`、`Sharpe`
- 系统使用以 `RankIC` 为第一步硬过滤依据的有序筛选流程
- 系统汇总训练集统计并输出最终特征清单 `state_features.npy`

### 场景 2：验证集独立复评和报告
- 用户继续运行 `valid` 特征评估
- 系统读取 `valid/*.feather`
- 系统仅使用 `train` 阶段得到的 `state_features.npy` 特征清单
- 系统重复评估并输出 per-contract 明细、汇总统计和 manifest/report
- 系统不执行 `Hard Filter`、`Stability Filter`、`Composite Score` 或 `Correlation Filter`
- 系统不输出供下游采用的新特征清单

### 场景 3：筛选后重新生成最终可用数据
- 系统使用 `FEATURE_SELECTION/${target_freq}/${symbol}/train/state_features.npy` 作为唯一 state feature 清单
- 系统读取 `SPLIT-TRAIN-VALID-TEST/${target_freq}/${symbol}/{train|valid|test}/{contract}.feather`
- 某个合约在某个 split 阶段不存在时，`scale_save` 支持跳过该合约阶段并继续处理其他存在的阶段
- 最终数据仍保留在 `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/...`，并按 split stage 区分输出目录

## 设计方向

采用方案 B：新增独立的多合约特征评估与筛选模块，shell 只负责调度。

### 调度顺序
`fu_full_process.sh` 调整为：
1. `dataset_split`
2. `feature_selection(train)`
3. `feature_selection(valid)`
4. `scale_save`
5. `maintenance_margin_dict`

### 目录边界
- `PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/${target_freq}`：承载 split 后的 `train/valid/test` 合约数据
- `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/${target_freq}`：承载特征评估、筛选、manifest、训练集最终特征清单，以及验证集评估报告
- `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE`：承载最终 scale 后的可用数据

### 模块边界
新增 `data_preprocess/operator_futures/feature_selection/muti_contract/*.py` 下的多合约流程模块，职责分成三段：
- 评估：逐合约计算 `Permutation Importance`、`CatBoost Importance`、`IC`、`RankIC`、`Sharpe`
- 汇总：保存 `Mean / Std / Median` 统计
- 筛选：训练阶段依次执行 `Hard Filter`、`Stability Filter`、`Composite Score`、`Correlation Filter`；验证阶段只做评估报告

### Sharpe 口径
`Sharpe` 采用单特征伪策略收益：
- 先对特征在当前输入内做列内 z-score
- 再与未来收益相乘得到伪收益序列
- 再对伪收益序列计算 Sharpe

## 关键决策

- `train` 输入为 `train/*.feather`，使用全部 state 特征，输出唯一最终特征清单 `state_features.npy`
- `valid` 输入为 `valid/*.feather`，仅使用 `train` 输出的 `state_features.npy` 做评估和报告，不做筛选，不产生下游采用的特征清单
- 每个合约都要输出 `Permutation Importance`、`CatBoost Importance`、`IC`、`RankIC`、`Sharpe`
- 汇总层必须保存 `Mean / Std / Median` 统计
- `FEATURE_SELECTION/${target_freq}` 是特征评估、筛选和最终中间产物的统一根目录
- `Hard Filter` 第一步必须使用 `abs(RankIC_Mean)`，不再使用 `abs(IC_Mean)`
- `scale_save` 必须后移，且读取 split 后各阶段合约文件，并始终使用训练集产生的 `state_features.npy`
- 旧的 `SCALE_SAVE` 目录保留，作为最终缩放结果目录，不再作为特征评估输入源

## 范围边界

**包含：**
- 调整 `fu_full_process.sh` 的步骤顺序
- 新增 `train` 特征评估与筛选流程
- 新增 `valid` 评估与报告流程，不做筛选
- 新增多合约特征评估模块
- 新增 per-contract 明细、汇总统计、manifest 和训练集最终特征清单输出
- 调整 `scale_save` 输入来源
- 增强 `scale_save` 对 split 阶段目录中合约缺失的处理

**不包含（本次）：**
- 修改商品期货环境、交易撮合、手续费、保证金逻辑
- 改变 `dataset_split` 的分割规则本身
- 改变 `SCALE_SAVE` 的缩放算法
- 改变已有的主力合约拼接逻辑

## 验收标准

- [ ] `fu_full_process.sh` 中步骤顺序变为 `dataset_split -> feature_selection(train) -> feature_selection(valid) -> scale_save -> maintenance_margin_dict`
- [ ] `dataset_split` 输出根目录为 `PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/${target_freq}`
- [ ] `FEATURE_SELECTION/${target_freq}/train` 能产出 per-contract 明细、汇总统计、manifest、`state_features.npy`
- [ ] `FEATURE_SELECTION/${target_freq}/valid` 能产出 per-contract 明细、汇总统计和 manifest/report，但不做筛选，不产生下游采用的新特征清单
- [ ] `train` 和 `valid` 两段都包含 `Permutation Importance`、`CatBoost Importance`、`IC`、`RankIC`、`Sharpe`
- [ ] `_ordered_filter_features` 第一步 hard filter 使用 `RankIC_Mean` 而不是 `IC_Mean`
- [ ] `scale_save` 使用训练集产生的 `state_features.npy` 处理 split 后存在的合约阶段，最终结果仍保留在 `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/{symbol}/{contract}/{target_freq}/{stage}/{start_date}-{end_date}/`
- [ ] 某个合约在某个 split 阶段不存在时，`scale_save` 跳过该缺失阶段并继续处理其他存在阶段
- [ ] 若输入缺失、训练特征清单为空或筛选结果为空，流程必须 fail-fast

## Amendments

### 2026-07-18: 明确指标计算和综合筛选口径
- 原因：实现过程中修正了 IC、RankIC、CatBoost Importance、多窗口和 Composite Score 的行为，需要把代码中的关键口径回写到 OpenSpec，避免规格只描述指标名称。
- 摘要：feature selection 默认按 `[1, 6, 12]` 窗口计算；IC、RankIC、CatBoost Importance 与原始 `ic_correlation.py`、`rank_ic_correlation.py`、`catbooost.py` 口径对齐；Composite Score 采用 RankIC 第一优先级、Sharpe/Permutation 第二优先级、CatBoost Importance 第三优先级，并删除排序后 10% 的低分因子且至少保留 1 个。

### 2026-07-19: 训练集决定最终特征清单，验证集只评估
- 原因：split 后每个阶段的合约覆盖不完全一致，使用 valid 再筛选会让最终清单依赖验证集可用合约，且与训练阶段筛选结果不一致。
- 摘要：`feature_selection_train` 输出的特征清单改名为最终 `state_features.npy`；`feature_selection_valid` 只做评估和报告，不做筛选；`_ordered_filter_features` 第一步 hard filter 使用 `RankIC_Mean`；后续 `scale_save` 只使用训练集产生的 `state_features.npy`，并支持某合约在某 split 阶段不存在时跳过该阶段。
