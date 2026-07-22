# 实现计划：select-cross-contract-low-level-agents

## 来源
- 提案：openspec/changes/select-cross-contract-low-level-agents/proposal.md
- 设计：openspec/changes/select-cross-contract-low-level-agents/design.md
- 规格：openspec/changes/select-cross-contract-low-level-agents/specs/
- 任务：openspec/changes/select-cross-contract-low-level-agents/tasks.md

## 实现步骤

### Task 1: Update low-level test-agent discovery tests and helper fixtures to model `valid/<contract>/<label>/df_*.feather`.
- [ ] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：让现有 `test_agent_index.py` 测试使用商品期货三层 valid 目录，先覆盖新 schema 的 RED 用例。
- 改动文件：`FineFT/tests/rl/test_test_agent_index.py`
- 验证方式：运行 `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py::test_weighted_trader_handles_nested_contract_label_directories -q`，预期当前实现因旧 schema 失败。

### Task 2: Refactor `test_agent_index.py` validation file discovery and aggregate result construction to emit pure `label`, aligned `contract`, and contract-relative `df_path` arrays.
- [ ] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：只发现 `valid/<contract>/<label>/df_*.feather`，按纯 `label + initial_action + bin_index` 聚合所有合约 slice。
- 改动文件：`FineFT/RL/DiHFT/low_level/test_agent_index.py`
- 验证方式：运行 Task 1 的 focused pytest，确认 npy 中 `label == label_2`、`contract == ["fu2507"]`、`df_path == ["fu2507/label_2/df_0.feather"]`。

### Task 3: Update aggregate CSV serialization and Chinese headers to include `合约` while preserving JSON array cells.
- [ ] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：将 `contract` 纳入 JSON 数组列，并让汇总 CSV 表头加入 `合约`。
- 改动文件：`FineFT/RL/DiHFT/low_level/test_agent_index.py`、`FineFT/tests/rl/test_test_agent_index.py`
- 验证方式：运行 `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py -q`，确认 `合约`、`数据文件` 等 JSON 单元格可解析。

### Task 4: Add picker schema-validation and sample-equal scoring tests for the new cross-contract result schema and legacy schema rejection.
- [ ] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：为 picker 添加新 schema 校验、旧 schema 拒绝、样本等权评分和最终聚合逻辑测试。
- 改动文件：`FineFT/tests/analysis/test_pick_agent.py`
- 验证方式：运行 `source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/analysis/test_pick_agent.py -q`，预期当前实现缺少校验/manifest 能力而失败。

### Task 5: Refactor `FineFT_single_agent_with_different_position.py` to validate new schema, preserve current first-stage and final-stage selection logic, and fail fast on invalid labels or metrics.
- [ ] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：在 picker 中加入 schema/label/metric 校验，并保持第一阶段与第二阶段当前选择逻辑不变。
- 改动文件：`FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`
- 验证方式：运行 Task 4 的 focused pytest，确认新 schema 通过、旧 schema 和非法指标 fail-fast。

### Task 6: Add `selection_manifest.json` output and enforce label-order model assembly for `model.pth`.
- [ ] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：让最终模型按 `label_0...label_N` 顺序组装，并在 analysis 输出目录写入选择 manifest。
- 改动文件：`FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`、`FineFT/tests/analysis/test_pick_agent.py`
- 验证方式：运行 picker focused tests，确认 manifest 字段完整，且每个 label 都有唯一选择后才写入。

### Task 7: Update commodity low-level test and picker shell scripts so `fu` runs pass the required base path, experiment name, position choices, and label count.
- [ ] **任务完成**（与 superpowers plan `Task 7`、`tasks.md` 对应条目同步勾选）
- 目标：让商品期货 `fu` 测试和筛选脚本使用同一套 `dataset/10min`、`experiment_name`、`position_choices` 和 `num_label` 参数。
- 改动文件：`FineFT/script/test/DiHFT/low_level/test_util_fu.sh`、`FineFT/script/analysis/pick_agent/low_level_fu.sh`
- 验证方式：运行 `bash -n FineFT/script/test/DiHFT/low_level/test_util_fu.sh FineFT/script/analysis/pick_agent/low_level_fu.sh`。

### Task 8: Run `conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/analysis/test_pick_agent.py -q`.
- [ ] **任务完成**（与 superpowers plan `Task 8`、`tasks.md` 对应条目同步勾选）
- 目标：执行本变更直接相关的 focused test suite。
- 改动文件：无代码改动；验证命令执行。
- 验证方式：`source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/analysis/test_pick_agent.py -q` 返回 0。

### Task 9: Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`.
- [ ] **任务完成**（与 superpowers plan `Task 9`、`tasks.md` 对应条目同步勾选）
- 目标：确认修改后的两个 Python 入口语法有效。
- 改动文件：无代码改动；验证命令执行。
- 验证方式：`source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py` 返回 0。

### Task 10: Run `openspec validate select-cross-contract-low-level-agents --strict`.
- [ ] **任务完成**（与 superpowers plan `Task 10`、`tasks.md` 对应条目同步勾选）
- 目标：确认实现后规格仍严格有效。
- 改动文件：无代码改动；验证命令执行。
- 验证方式：`openspec validate select-cross-contract-low-level-agents --strict` 输出 `Change 'select-cross-contract-low-level-agents' is valid`。

