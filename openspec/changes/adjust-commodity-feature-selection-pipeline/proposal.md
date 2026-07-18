# adjust-commodity-feature-selection-pipeline

## 背景与目标

当前 `fu_full_process.sh` 在 `dataset_split` 之后没有独立的特征评估与筛选阶段，`scale_save` 也仍然依赖旧的前置产物。现在需要把商品期货特征选择改成一条更清晰的流水线：

1. 先按合约切分出 `train/valid/test`
2. 再分别对 `train` 和 `valid` 做独立特征评估与筛选
3. 先产出候选特征，再产出最终特征
4. 最后对筛选后的新合约 `df.feather` 执行 `scale_save`

目标是把“特征统计评估”和“特征筛选”从旧的 `IC_RESULT / SCALE_SAVE` 步骤中拆出来，统一沉淀到 `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/${target_freq}`，同时保留原有 `SCALE_SAVE` 作为最终缩放输出目录。

## 用户场景

### 场景 1：训练集特征评估与候选筛选
- 用户运行商品期货全流程后，`dataset_split` 已生成 `SPLIT-TRAIN-VALID-TEST/${target_freq}/train/*.feather`
- 系统读取 `train/*.feather` 中所有 state 特征
- 系统按合约计算 `Permutation Importance`、`CatBoost Importance`、`IC`、`RankIC`、`Sharpe`
- 系统汇总训练集统计并输出 `state_features_candidate.npy`

### 场景 2：验证集独立复评与最终筛选
- 用户继续运行 `valid` 特征评估
- 系统读取 `valid/*.feather`
- 系统仅使用 `train` 阶段得到的候选特征集
- 系统重复评估、汇总和筛选，最终输出 `state_features.npy`

### 场景 3：筛选后重新生成最终可用数据
- 系统把 `FEATURE_SELECTION/${target_freq}/valid` 中筛完的特征子集重新写成合约级 `df.feather`
- 系统再基于这些新文件执行 `scale_save`
- 最终数据仍保留在 `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/...`

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
- `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/${target_freq}`：承载特征评估、筛选、manifest、候选集、最终特征集，以及重新生成的合约级 `df.feather`
- `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE`：承载最终 scale 后的可用数据

### 模块边界
新增 `data_preprocess/operator_futures/feature_selection/muti_contract/*.py` 下的多合约流程模块，职责分成三段：
- 评估：逐合约计算 `Permutation Importance`、`CatBoost Importance`、`IC`、`RankIC`、`Sharpe`
- 汇总：保存 `Mean / Std / Median` 统计
- 筛选：依次执行 `Hard Filter`、`Stability Filter`、`Composite Score`、`Correlation Filter`

### Sharpe 口径
`Sharpe` 采用单特征伪策略收益：
- 先对特征在当前输入内做列内 z-score
- 再与未来收益相乘得到伪收益序列
- 再对伪收益序列计算 Sharpe

## 关键决策

- `train` 和 `valid` 必须独立评估与筛选，不共享最终筛选结果
- `train` 输入为 `train/*.feather`，使用全部 state 特征
- `valid` 输入为 `valid/*.feather`，仅使用 `train` 输出的候选特征
- 每个合约都要输出 `Permutation Importance`、`CatBoost Importance`、`IC`、`RankIC`、`Sharpe`
- 汇总层必须保存 `Mean / Std / Median` 统计
- `FEATURE_SELECTION/${target_freq}` 是特征评估、筛选和最终中间产物的统一根目录
- `scale_save` 必须后移，且读取筛选后重新生成的合约 `df.feather`
- 旧的 `SCALE_SAVE` 目录保留，作为最终缩放结果目录，不再作为特征评估输入源

## 范围边界

**包含：**
- 调整 `fu_full_process.sh` 的步骤顺序
- 新增 `train / valid` 两段独立特征评估与筛选流程
- 新增多合约特征评估模块
- 新增 per-contract 明细、汇总统计、manifest、候选特征集和最终特征集输出
- 重新生成筛选后的合约级 `df.feather`
- 调整 `scale_save` 输入来源

**不包含（本次）：**
- 修改商品期货环境、交易撮合、手续费、保证金逻辑
- 改变 `dataset_split` 的分割规则本身
- 改变 `SCALE_SAVE` 的缩放算法
- 改变已有的主力合约拼接逻辑

## 验收标准

- [ ] `fu_full_process.sh` 中步骤顺序变为 `dataset_split -> feature_selection(train) -> feature_selection(valid) -> scale_save -> maintenance_margin_dict`
- [ ] `dataset_split` 输出根目录为 `PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST/${target_freq}`
- [ ] `FEATURE_SELECTION/${target_freq}/train` 能产出 per-contract 明细、汇总统计、manifest、`state_features_candidate.npy`
- [ ] `FEATURE_SELECTION/${target_freq}/valid` 能产出 per-contract 明细、汇总统计、manifest、`state_features.npy`
- [ ] `train` 和 `valid` 两段都包含 `Permutation Importance`、`CatBoost Importance`、`IC`、`RankIC`、`Sharpe`
- [ ] `FEATURE_SELECTION/${target_freq}` 下能生成筛选后的新合约 `df.feather`
- [ ] `scale_save` 使用筛选后的新合约 `df.feather`，最终结果仍保留在 `PREPROCESS_DATASET/commodity-futures/SCALE_SAVE`
- [ ] 若输入缺失、候选集为空或筛选结果为空，流程必须 fail-fast

## Amendments

### 2026-07-18: 明确指标计算和综合筛选口径
- 原因：实现过程中修正了 IC、RankIC、CatBoost Importance、多窗口和 Composite Score 的行为，需要把代码中的关键口径回写到 OpenSpec，避免规格只描述指标名称。
- 摘要：feature selection 默认按 `[1, 6, 12]` 窗口计算；IC、RankIC、CatBoost Importance 与原始 `ic_correlation.py`、`rank_ic_correlation.py`、`catbooost.py` 口径对齐；Composite Score 采用 RankIC 第一优先级、Sharpe/Permutation 第二优先级、CatBoost Importance 第三优先级，并删除排序后 10% 的低分因子且至少保留 1 个。
