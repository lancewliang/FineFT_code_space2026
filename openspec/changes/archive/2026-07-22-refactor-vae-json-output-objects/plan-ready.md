# 实现计划：refactor-vae-json-output-objects

## 来源
- 提案：openspec/changes/refactor-vae-json-output-objects/proposal.md
- 设计：openspec/changes/refactor-vae-json-output-objects/design.md
- 规格：openspec/changes/refactor-vae-json-output-objects/specs/
- 任务：openspec/changes/refactor-vae-json-output-objects/tasks.md

## 实现步骤

### Task 1: Add focused VAE JSON object tests
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：先用 RED 测试定义 VAE JSON 输出对象接口，覆盖 `LabelTrainingManifest`、`LabelSummary`、`RoutingSummary`、对象属性访问、`maybe_write_routing_summary_after_analysis()` 返回类型，以及 `to_dict()` 与写出 JSON 的一致性。
- 改动文件：`FineFT/tests/rl/test_commodity_vae_cross_contract.py`
- 验证方式：运行 `conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py -q`，预期在实现前因缺少 `manifests.py` 或返回值仍为 dict 而失败。
- 来源：`tasks.md` → `- [ ] 1.1 Add focused tests for VAE JSON output objects covering `LabelTrainingManifest`, `LabelSummary`, `RoutingSummary`, object attribute access, `maybe_write_routing_summary_after_analysis()` return type, and `to_dict()` equality with written JSON files.`

### Task 2: Add VAE manifest dataclasses
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：新增 `FineFT/RL/DiHFT/VAE/manifests.py`，集中定义训练 manifest、logpx summary、sample integrity、acceptance、winner summary、contract routing summary 和 routing summary dataclass，并提供兼容 JSON 的 `to_dict()`。
- 改动文件：`FineFT/RL/DiHFT/VAE/manifests.py`
- 验证方式：运行 `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/manifests.py`，并重跑 focused tests，预期类型导入相关失败消失，调用方返回 dict 的失败仍存在。
- 来源：`tasks.md` → `- [ ] 1.2 Add `FineFT/RL/DiHFT/VAE/manifests.py` with dataclass models for training manifest, logpx summary, sample integrity, acceptance, winner summary, contract routing summary, and routing summary serialization.`

### Task 3: Refactor VAE train manifest flow to objects
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：将 `materialize_label_training_data()` 改为返回 `LabelTrainingManifest`，并将 `main.py` 训练模式和 analyze-only 模式中的训练数据传递改为对象属性访问，保留 `label_k_manifest.json` 兼容结构。
- 改动文件：`FineFT/RL/DiHFT/VAE/merge_vae_train.py`、`FineFT/RL/DiHFT/VAE/main.py`
- 验证方式：运行 `conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_materialize_label_training_data_merges_contract_arrays_and_writes_manifest -q`，预期通过，并确认 JSON 文件内容等于 `result.to_dict()`。
- 来源：`tasks.md` → `- [ ] 1.3 Refactor `FineFT/RL/DiHFT/VAE/merge_vae_train.py` and `main.py` so materialized train data is represented and passed as objects, not dicts, while preserving `label_k_manifest.json`.`

### Task 4: Refactor VAE summary and routing flow to objects
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：将 `process.py` 传给 `summary.py` 的 per-contract analysis result 和 train baseline、以及 `summary.py` 的 logpx stats、sample integrity、per-contract summary、label summary、winner summary 和 routing summary 改为对象传递与返回，保留 `summary.json` 和 `routing_summary.json` 兼容结构。
- 改动文件：`FineFT/RL/DiHFT/VAE/process.py`、`FineFT/RL/DiHFT/VAE/summary.py`、`FineFT/tests/rl/test_commodity_vae_cross_contract.py`
- 验证方式：运行 focused summary/routing tests：`conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_contract_logpx_outputs_writes_per_contract_and_aggregate_files FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_contract_logpx_outputs_includes_enhanced_summary_metrics FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_routing_summary_compares_labels_by_contract FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_main_writes_routing_summary_after_analysis_when_all_labels_ready -q`。
- 来源：`tasks.md` → `- [ ] 1.4 Refactor `FineFT/RL/DiHFT/VAE/process.py` and `summary.py` so per-label summary inputs, summary outputs, and routing summary data are represented and returned as objects, not dicts, while preserving `summary.json` and `routing_summary.json`.`

### Task 5: Run focused VAE tests
- [x] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：运行 VAE focused test，验证对象接口、JSON 兼容、现有 VAE shell/parser 测试未回归。
- 改动文件：无代码改动；验证 `FineFT/tests/rl/test_commodity_vae_cross_contract.py`
- 验证方式：运行 `conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py`，预期通过。
- 来源：`tasks.md` → `- [ ] 2.1 Run `conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py`.`

### Task 6: Run VAE py_compile
- [x] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：对新增对象模块和改动的 VAE Python 文件运行语法验证。
- 改动文件：无代码改动；验证 `FineFT/RL/DiHFT/VAE/manifests.py`、`merge_vae_train.py`、`main.py`、`summary.py`
- 验证方式：运行 `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/manifests.py FineFT/RL/DiHFT/VAE/merge_vae_train.py FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/summary.py`，预期无输出且退出码为 0。
- 来源：`tasks.md` → `- [ ] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/manifests.py FineFT/RL/DiHFT/VAE/merge_vae_train.py FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/summary.py`.`

### Task 7: Run OpenSpec strict validation
- [x] **任务完成**（与 superpowers plan `Task 7`、`tasks.md` 对应条目同步勾选）
- 目标：确认 OpenSpec 变更制品仍通过 strict validation。
- 改动文件：无代码改动；验证 `openspec/changes/refactor-vae-json-output-objects/`
- 验证方式：运行 `openspec validate refactor-vae-json-output-objects --strict`，预期输出 `Change 'refactor-vae-json-output-objects' is valid`。
- 来源：`tasks.md` → `- [ ] 2.3 Run `openspec validate refactor-vae-json-output-objects --strict`.`

## Amendments

### 2026-07-22: 扩大 VAE 内部 source/loader 对象化范围
- 原因：build 完成后复查发现，VAE 内部仍在通过 `list[dict]` 传递 label source、test source 和 contract loader 元数据，这与“用对象而不是 dict 书写代码”的目标不一致。
- 影响规格：`openspec/changes/refactor-vae-json-output-objects/specs/fineft-vae-json-outputs/spec.md`
- 影响任务：新增 `1.5`、`1.6`、`2.4`、`2.5`、`2.6`

### Task 8: Add focused source/loader object tests
- [x] **任务完成**（与 superpowers plan `Task 8`、`tasks.md` 对应条目同步勾选）
- 目标：先用 RED 测试定义 VAE label source、test source 和 contract loader 的对象化接口，覆盖属性访问和现有 discovery/validation 行为。
- 改动文件：`FineFT/tests/rl/test_commodity_vae_cross_contract.py`
- 验证方式：运行 `eval "$(conda shell.bash hook)" && conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py -q`，预期在实现前因新的对象类型尚未接入而失败。
- 来源：`tasks.md` → `- [ ] 1.5 Add focused tests for VAE source discovery and loader preparation objects covering `LabelArraySource`, `TestContractSource`, `ContractDatasetLoader`, object attribute access, and preservation of existing discovery/validation behavior.`

### Task 9: Refactor VAE source and loader flow to objects
- [x] **任务完成**（与 superpowers plan `Task 9`、`tasks.md` 对应条目同步勾选）
- 目标：将 `discover_label_sources()`、`discover_test_sources()`、`prepare_contract_dataset_loader_list()` 和 `analyze_contract_tests()` 改为对象传递，不再在内部方法之间传递 `list[dict]`。
- 改动文件：`FineFT/RL/DiHFT/VAE/merge_vae_train.py`、`FineFT/RL/DiHFT/VAE/main.py`、`FineFT/RL/DiHFT/VAE/process.py`
- 验证方式：运行 focused source/loader tests 以及现有 VAE focused test module，预期通过并保持 JSON 输出兼容。
- 来源：`tasks.md` → `- [ ] 1.6 Refactor `FineFT/RL/DiHFT/VAE/merge_vae_train.py`, `main.py`, and `process.py` so label source discovery, test source discovery, and contract loader preparation return and pass dataclass objects instead of `list[dict]`.`

### Task 10: Re-run verification for the expanded scope
- [x] **任务完成**（与 superpowers plan `Task 10`、`tasks.md` 对应条目同步勾选）
- 目标：对扩大范围后的 VAE 对象化重构重新执行 focused tests、py_compile 和 OpenSpec 校验。
- 改动文件：无代码改动；验证 `FineFT/tests/rl/test_commodity_vae_cross_contract.py` 与 VAE Python 文件
- 验证方式：运行 `eval "$(conda shell.bash hook)" && conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py`、`eval "$(conda shell.bash hook)" && conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/manifests.py FineFT/RL/DiHFT/VAE/merge_vae_train.py FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/process.py FineFT/RL/DiHFT/VAE/summary.py`、`openspec validate refactor-vae-json-output-objects --strict`。
- 来源：`tasks.md` → `- [ ] 2.4 Re-run `conda activate finetf && pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py`.` | `- [ ] 2.5 Re-run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/manifests.py FineFT/RL/DiHFT/VAE/merge_vae_train.py FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/process.py FineFT/RL/DiHFT/VAE/summary.py`.` | `- [ ] 2.6 Re-run `openspec validate refactor-vae-json-output-objects --strict`.`
