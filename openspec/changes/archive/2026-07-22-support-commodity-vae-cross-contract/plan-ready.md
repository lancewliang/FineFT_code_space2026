# 实现计划：support-commodity-vae-cross-contract

## 来源
- 提案：openspec/changes/support-commodity-vae-cross-contract/proposal.md
- 设计：openspec/changes/support-commodity-vae-cross-contract/design.md
- 规格：openspec/changes/support-commodity-vae-cross-contract/specs/
- 任务：openspec/changes/support-commodity-vae-cross-contract/tasks.md

## 实现步骤

### Task 1: Add cross-contract VAE training materialization tests
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：用 RED 测试定义 `VAE_data/<contract>/label_k.npy` 合并、`VAE_data/train/label_k.npy`、manifest、缺失 label 和维度校验。
- 改动文件：`FineFT/tests/rl/test_commodity_vae_cross_contract.py`
- 验证方式：`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py -q`，预期在实现前因缺少 helper/API 失败。

### Task 2: Implement VAE data discovery and train-set materialization
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：在 `merge_vae_train.py` 中实现商品多合约 VAE 数据发现、训练集合并、manifest 写入和 fail-fast 校验，并由 `main.py` 调用。
- 改动文件：`FineFT/RL/DiHFT/VAE/merge_vae_train.py`、`FineFT/RL/DiHFT/VAE/main.py`
- 验证方式：运行 Task 1 的 focused pytest，确认训练集合并相关测试通过。

### Task 3: Add per-contract VAE analysis output tests
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：用 mock VAE 分析结果定义 `ood_logpx_<contract>.npy/.csv`、`ood_logpx_all.npy/.csv` 和 `summary.json` 的输出契约。
- 改动文件：`FineFT/tests/rl/test_commodity_vae_cross_contract.py`
- 验证方式：运行 focused pytest，预期在实现前因缺少分合约分析函数失败。

### Task 4: Implement per-contract test analysis and aggregate outputs
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：改造 `process.py`、`main.py` 和 `summary.py`，逐合约读取 `VAE_data/test/test_<contract>.npy`，由 `summary.py` 输出分合约和总体结果。
- 改动文件：`FineFT/RL/DiHFT/VAE/process.py`、`FineFT/RL/DiHFT/VAE/main.py`、`FineFT/RL/DiHFT/VAE/summary.py`
- 验证方式：运行 Task 3 的 focused pytest，确认 `.npy`、CSV 和 summary 输出符合规格。

### Task 5: Replace ambiguous VAE CLI booleans with explicit workflow flags
- [x] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：新增 `--train` 和 `--analyze-only`，训练模式执行合并、训练、训练后分合约分析；分析模式只加载 `model_latest.pth` 并重跑分析。
- 改动文件：`FineFT/RL/DiHFT/VAE/main.py`、`FineFT/tests/rl/test_commodity_vae_cross_contract.py`
- 验证方式：运行 focused pytest 中的 CLI 参数测试，并确保旧 `--if_train True` 不再是商品训练入口要求。

### Task 6: Update fu VAE shell entry
- [x] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：让 `VAE_util_fu.sh` 激活 `finetf`、设置 `PYTHONPATH`、传 `--dataset_name fu`、`--data_base_path dataset/10min`、`--train`，启动 `label_0..label_4`，并默认最多同时运行 2 个训练进程。
- 改动文件：`FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh`
- 验证方式：`bash -n FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh` 返回 0，并通过 focused pytest 确认关键参数、并发上限和无效 `MAX_PARALLEL_JOBS` fail-fast。

### Task 7: Run focused VAE tests or full FineFT tests
- [x] **任务完成**（与 superpowers plan `Task 7`、`tasks.md` 对应条目同步勾选）
- 目标：验证商品 VAE 跨合约训练与分析测试整体通过；如果全量测试被外部 fixture 阻塞，记录阻塞并运行 focused VAE 测试。
- 改动文件：无代码改动；执行验证命令。
- 验证方式：优先运行 `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests -q`；若被无关外部数据阻塞，运行 `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py -q`。

### Task 8: Run VAE py_compile
- [x] **任务完成**（与 superpowers plan `Task 8`、`tasks.md` 对应条目同步勾选）
- 目标：确认改造后的 VAE Python 入口语法有效。
- 改动文件：无代码改动；执行验证命令。
- 验证方式：`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/process.py FineFT/RL/DiHFT/VAE/merge_vae_train.py` 返回 0。

### Task 9: Run shell syntax validation
- [x] **任务完成**（与 superpowers plan `Task 9`、`tasks.md` 对应条目同步勾选）
- 目标：确认 `VAE_util_fu.sh` shell 语法有效。
- 改动文件：无代码改动；执行验证命令。
- 验证方式：`bash -n FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh` 返回 0。

### Task 10: Run OpenSpec strict validation
- [x] **任务完成**（与 superpowers plan `Task 10`、`tasks.md` 对应条目同步勾选）
- 目标：确认规格在实现后仍严格有效。
- 改动文件：无代码改动；执行验证命令。
- 验证方式：`openspec validate support-commodity-vae-cross-contract --strict` 输出 `Change 'support-commodity-vae-cross-contract' is valid`。

## Amendments

### 2026-07-22: 增强 VAE summary 门控分析指标
- 原因：现有 `summary.json` 只提供 raw `logpx` 基础统计，不足以分析按行情 label 训练的门控模型；需要补充训练基准、分位数、接受率、样本完整性和跨 label routing 汇总。
- 影响规格：`openspec/changes/support-commodity-vae-cross-contract/specs/commodity-futures-support/spec.md`
- 影响任务：`tasks.md` 1.7、1.8、1.9、1.10、2.5

## 追加实现步骤

### Task 11: Add enhanced per-label summary metric tests
- [x] **任务完成**（与 superpowers plan `Task 11`、`tasks.md` 1.7 同步勾选）
- 目标：用 focused tests 定义 `summary.json` 的 train baseline、test quantiles、acceptance 和 sample integrity 字段。
- 改动文件：`FineFT/tests/rl/test_commodity_vae_cross_contract.py`
- 验证方式：`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_contract_logpx_outputs_includes_enhanced_summary_metrics -q` 先失败，随后实现后通过。

### Task 12: Implement enhanced per-label summary metrics
- [x] **任务完成**（与 superpowers plan `Task 12`、`tasks.md` 1.8 同步勾选）
- 目标：在 `summary.py` 中输出训练基准、分位数、接受率和样本完整性，并从 `main.py` 分析流程传入训练集 logpx 基准；`process.py` 只收集分析结果并委托 summary writer。
- 改动文件：`FineFT/RL/DiHFT/VAE/summary.py`、`FineFT/RL/DiHFT/VAE/process.py`、`FineFT/RL/DiHFT/VAE/main.py`
- 验证方式：运行 Task 11 focused pytest，确认 enhanced summary 测试通过。

### Task 13: Add cross-label routing summary tests
- [x] **任务完成**（与 superpowers plan `Task 13`、`tasks.md` 1.9 同步勾选）
- 目标：用小数组定义 `routing_summary.json` 的 winner distribution、top1/top2 margin、low-margin rate 和 mismatched label sample counts。
- 改动文件：`FineFT/tests/rl/test_commodity_vae_cross_contract.py`
- 验证方式：`source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_commodity_vae_cross_contract.py::test_write_routing_summary_compares_labels_by_contract -q` 先失败，随后实现后通过。

### Task 14: Implement cross-label routing summary generation
- [x] **任务完成**（与 superpowers plan `Task 14`、`tasks.md` 1.10 同步勾选）
- 目标：在 `summary.py` 新增 helper 读取多个 `label_k/ood_logpx_<contract>.npy`，生成 `result/DiHFT/vae_results/<dataset_name>/routing_summary.json`，并在 `piplinerunner.analyze_contracts()` 后检查所有 label 输出齐全时自动调用；不增加 `--routing-summary` workflow flag，不依赖 shell 后处理 snippet。
- 改动文件：`FineFT/RL/DiHFT/VAE/summary.py`、`FineFT/RL/DiHFT/VAE/main.py`、`FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh`
- 验证方式：运行 Task 13 focused pytest，并用 focused parser/shell 测试确认无 `--routing-summary` 入口；再用 `bash -n FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh` 校验 shell 入口。

### Task 15: Re-run summary extension verification
- [x] **任务完成**（与 superpowers plan `Task 15`、`tasks.md` 2.5 同步勾选）
- 目标：验证增强 summary 与 routing summary 改动未破坏现有商品 VAE 工作流。
- 改动文件：无代码改动；执行验证命令。
- 验证方式：运行 focused commodity VAE tests、包含 `summary.py` 的 VAE py_compile、shell syntax validation 和 `openspec validate support-commodity-vae-cross-contract --strict`。
