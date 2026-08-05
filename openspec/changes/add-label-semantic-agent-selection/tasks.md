## 1. Specification

- [x] 1.1 确认 `label_semantics.json` 的最终存放位置和 picker 参数名。
- [x] 1.2 明确 `slope`、`quantile`、`DTW` 三种 label 方法下的语义生成/输入规则。
- [x] 1.3 确认方向行为指标字段是否直接写入 `analysis_result.npy/csv`，避免 picker 依赖大体积逐步明细 CSV。
- [x] 1.4 确认涨跌停语义字段 `limit_state`、`limit_state_sign` 与商品期货 `LowerLimitPrice`、`UpperLimitPrice`、`limit_up_single_sided_ratio`、`limit_down_single_sided_ratio` 的映射规则。
- [x] 1.5 固化当前编号约定：`label_0` 为跌停/接近跌停，`label_<label_number-1>` 即 `label_{n+1}` 为涨停/接近涨停，`label_1..label_n` 为普通动态 label。

## 2. Implementation

- [x] 2.1 在 `FineFT/RL/DiHFT/low_level/test_agent_index.py` 聚合每个验证 slice 的方向行为指标：平均仓位、多头步数占比、空头步数占比、空仓步数占比、多头收益、空头收益、净敞口。
- [x] 2.2 在 `FineFT/RL/DiHFT/low_level/test_agent_index.py` 聚合每个验证 slice 的涨跌停行为指标：涨停步数占比、跌停步数占比、涨停多头收益、跌停空头收益、涨停反向空头占比、跌停反向多头占比。
- [x] 2.3 更新 `analysis_result.npy` 和 `analysis_result.csv` schema，使新增数组字段与 `contract`、`df_path`、`reward_sum`、`df_length` 对齐。
- [x] 2.4 在 `FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py` 增加 `label_semantics.json` 读取、覆盖校验、DTW 显式语义校验、首尾 label 涨跌停约定校验和涨跌停语义校验。
- [x] 2.5 在 picker 的候选转换阶段计算 candidate-level 行为摘要，例如 `candidate_mean_exposure`、`candidate_long_ratio`、`candidate_short_ratio`、`candidate_long_reward_mean`、`candidate_short_reward_mean`、`candidate_limit_up_long_reward_mean`、`candidate_limit_down_short_reward_mean`。
- [x] 2.6 在第一阶段和最终阶段选择前应用 label 语义硬过滤：上涨/大涨要求偏多且多头正收益，下跌/大跌要求偏空且空头正收益，震荡要求净敞口受控，涨停要求涨停多头盈利且低反向空头占比，跌停要求跌停空头盈利且低反向多头占比。
- [x] 2.7 扩展 `selection_manifest.json`，记录每个 label 的方向语义、涨跌停语义、过滤阈值、候选行为摘要、拒绝原因摘要和最终选择原因。
- [x] 2.8 更新商品期货相关脚本参数，支持传入 `--label_semantics_path` 或生成默认 slope/quantile 语义 manifest。

## 3. Tests

- [x] 3.1 为低层测试聚合结果新增单元测试，验证方向行为指标字段存在、数组长度对齐、CSV JSON 单元格可解析。
- [x] 3.2 为低层测试聚合结果新增单元测试，验证涨跌停行为指标字段存在、数组长度对齐、CSV JSON 单元格可解析。
- [x] 3.3 为 picker 新增大涨 label 用例：偏空高收益候选被拒绝，偏多正收益候选被选中。
- [x] 3.4 为 picker 新增大跌 label 用例：偏多高收益候选被拒绝，偏空正收益候选被选中。
- [x] 3.5 为 picker 新增震荡 label 用例：强趋势候选被拒绝，低净敞口候选被选中。
- [x] 3.6 为 picker 新增涨停 label 用例：涨停反向空头占比过高的候选被拒绝，涨停多头盈利候选被选中。
- [x] 3.7 为 picker 新增跌停 label 用例：跌停反向多头占比过高的候选被拒绝，跌停空头盈利候选被选中。
- [x] 3.8 为首尾 label 约定新增测试：`label_0` 非跌停类时失败，最后一个 label 非涨停类时失败。
- [x] 3.9 为失败路径新增测试：缺失 label 语义、DTW 无显式语义、缺失方向行为指标、缺失涨跌停行为指标、无候选通过语义过滤。
- [x] 3.10 保留并更新现有 `FineFT/tests/rl/test_test_agent_index.py` 和 `FineFT/tests/analysis/test_pick_agent.py`。

## 4. Verification

- [x] 4.1 运行 `conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/analysis/test_pick_agent.py -q`。
- [x] 4.2 运行 `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`。
- [x] 4.3 运行 `openspec validate add-label-semantic-agent-selection --strict`。
- [x] 4.4 若缺少真实商品期货验证数据或 GPU，记录未运行的全量训练/回测步骤；本变更的核心验证应通过 focused tests 完成。

## 5. Rollback

- [ ] 5.1 如需回滚，移除新增语义过滤参数和新增行为指标消费逻辑，保留原有 `trans_reward_mean` 选择路径。
- [ ] 5.2 回滚后删除或忽略新增 `selection_manifest.json` 语义字段，不改变旧 `model.pth` 读取方式。
