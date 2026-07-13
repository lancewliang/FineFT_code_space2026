# refactor-commodity-feature-selection-union

## 背景与目标

商品期货多合约预处理当前在每个合约内连续执行 `ic_correlation` 和 `scale_save`。`ic_correlation` 会立即用该合约自身选出的 `state_features` 过滤并写出 `IC_RESULT/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather`。所有合约完成后才执行的 `feature_union` 已经无法恢复其他合约需要但被提前裁掉的特征，导致多个合约不能可靠使用同一组状态特征。

目标是把商品期货 feature selection 改成两阶段流程：先逐合约计算候选特征，等所有合约完成后生成 union，再用 union 回读每个合约的全量 `ALL_FEATURE` 并生成过滤后的合约数据文件。后续 `scale_save` 继续消费标准 `IC_RESULT` 输出。

## 用户场景

- 用户运行 `fu_full_process.sh` 处理商品期货多合约数据时，希望所有合约最终使用同一份 union 后的 `state_features.npy`。
- 用户需要每个合约仍然拥有过滤完特征的 `IC_RESULT/{contract}/df.feather`，供 `scale_save` 按现有接口继续执行。
- 用户需要通过 manifest 审计每个合约候选特征、union 特征和最终过滤后文件的路径与规模。

## 设计方向

采用两阶段 IC selection：

1. 合约循环内执行 candidate 阶段。`ic_correlation` 对单个合约读取 `ALL_FEATURE`，计算 IC、窗口报告和相关性去重结果，只写 `state_features_candidate.npy`、`ic_window_*.json`、`correlation.csv` 和候选 manifest，不写标准 `df.feather`，也不写标准 `state_features.npy`。
2. 所有合约 candidate 完成后执行 union finalize。该阶段读取 summary 中所有合约的 `state_features_candidate.npy`，生成品种级 union，并写入 `FEATURE_UNION/{symbol}/{target_freq}/{start_date}-{end_date}/state_features.npy` 和 `feature_union_manifest.json`。
3. union finalize 随后逐合约回读 `ALL_FEATURE/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}.feather`，按 `reward_features + union_state_features` 写出标准 `IC_RESULT/{symbol}/{contract}/{target_freq}/{start_date}-{end_date}/df.feather` 和 `state_features.npy`。
4. `scale_save` 保持职责不变，只读取每个合约标准 `IC_RESULT` 输出并缩放保存。
5. `fu_full_process.sh` 调整顺序为：每个合约执行到 `ic_candidate`，所有合约完成后执行 `ic_union_finalize`，再逐合约执行 `scale_save`。旧的独立后置 `feature_union` 步骤不再保留为单独阶段，因为 union 和过滤输出由 finalize 阶段完成。

## 关键决策

- `feature_union` 的能力提前到 IC selection finalize 阶段，但不省略生成过滤后合约 `df.feather` 的步骤。
- candidate 阶段不写标准 `state_features.npy`，避免后续步骤误读尚未 union 的合约级候选特征。
- 标准 `IC_RESULT/{contract}/state_features.npy` 必须等于品种级 union `state_features.npy`。
- union finalize 对缺失 candidate、空 union、union 特征在任一合约 `ALL_FEATURE` 中缺列等问题直接失败。
- `scale_save` 不承担 union、补列或特征降级逻辑，继续只做缩放和保存。

## 范围边界

**包含：**
- 为商品期货 `ic_correlation` 增加 candidate-only 输出模式。
- 扩展或调整 `contract_feature_union`，支持从 IC candidate 读取候选特征、生成 union、并逐合约写出过滤后的标准 `IC_RESULT` 文件。
- 调整 `fu_full_process.sh` 的商品期货执行顺序，使 `scale_save` 在 union finalize 后执行。
- 更新相关测试，覆盖 candidate-only、union finalize、缺列失败和 shell 阶段顺序。

**不包含（本次）：**
- 不修改 IC、Rank IC、CatBoost、Lasso 的特征重要性算法。
- 不修改状态特征公式、奖励特征公式或时间滚动特征生成逻辑。
- 不修改 `scale_save` 的缩放算法和输出契约。
- 不修改 crypto futures 的默认单合约预处理流程，除非为了保持兼容需要做参数默认值保护。
- 不修改 FineFT 训练算法。

## 验收标准

- [ ] 商品期货每个合约运行 candidate 阶段后，只生成候选特征和报告，不生成标准 `df.feather` 或标准 `state_features.npy`。
- [ ] union finalize 读取所有合约 candidate 后，生成品种级 `FEATURE_UNION/state_features.npy` 和 manifest。
- [ ] union finalize 为 summary 中每个合约生成过滤后的 `IC_RESULT/{contract}/df.feather`，列包含 `reward_features + union_state_features`。
- [ ] 每个合约标准 `IC_RESULT/{contract}/state_features.npy` 与品种级 union `state_features.npy` 内容一致。
- [ ] union 中任一特征在某个合约 `ALL_FEATURE` 缺失时，finalize 失败并指出缺失合约和缺失特征。
- [ ] `fu_full_process.sh` 不再在旧位置单独调用后置 `feature_union`；`scale_save` 在 `ic_union_finalize` 后按合约执行。
- [ ] 相关 data_preprocess 测试通过，至少覆盖商品 feature pipeline、商品主合约 CLI 和 feature selection Polars 测试。
