## 1. Implementation

- [x] 1.1 Update low-level test-agent discovery tests and helper fixtures to model `valid/<contract>/<label>/df_*.feather`. <!-- 已实现: 测试 fixture 改为三层 valid 目录，并覆盖新 schema RED 用例 -->
- [x] 1.2 Refactor `test_agent_index.py` validation file discovery and aggregate result construction to emit pure `label`, aligned `contract`, and contract-relative `df_path` arrays. <!-- 已实现: discovery 仅扫描合约/label slice，并输出纯 label 与对齐的 contract/df_path -->
- [x] 1.3 Update aggregate CSV serialization and Chinese headers to include `合约` while preserving JSON array cells. <!-- 已实现: 汇总 CSV 新增 合约 列，JSON 数组单元格可解析 -->
- [x] 1.4 Add picker schema-validation and sample-equal scoring tests for the new cross-contract result schema and legacy schema rejection. <!-- 已实现: picker 新增样本等权、旧 schema、label 覆盖和 manifest 测试 -->
- [x] 1.5 Refactor `FineFT_single_agent_with_different_position.py` to validate new schema, preserve current first-stage and final-stage selection logic, and fail fast on invalid labels or metrics. <!-- 已实现: picker 加入 schema/label/metric 校验并保留当前两阶段选择逻辑 -->
- [x] 1.6 Add `selection_manifest.json` output and enforce label-order model assembly for `model.pth`. <!-- 已实现: 输出 selection_manifest.json 并按 label 顺序组装 model.pth -->
- [x] 1.7 Update commodity low-level test and picker shell scripts so `fu` runs pass the required base path, experiment name, position choices, and label count. <!-- 已实现: 商品脚本参数与日志路径已对齐 -->

## 2. Verification

- [x] 2.1 Run `conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/analysis/test_pick_agent.py -q`. <!-- 已实现: focused pytest 12 passed -->
- [x] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`. <!-- 已实现: py_compile 通过 -->
- [x] 2.3 Run `openspec validate select-cross-contract-low-level-agents --strict`. <!-- 已实现: strict validation 通过 -->
