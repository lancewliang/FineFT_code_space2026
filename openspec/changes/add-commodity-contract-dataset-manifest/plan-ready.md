# 实现计划：add-commodity-contract-dataset-manifest

## 来源
- 提案：openspec/changes/add-commodity-contract-dataset-manifest/proposal.md
- 设计：openspec/changes/add-commodity-contract-dataset-manifest/design.md
- 规格：openspec/changes/add-commodity-contract-dataset-manifest/specs/
- 任务：openspec/changes/add-commodity-contract-dataset-manifest/tasks.md

## 实现步骤

### Task 1: Commodity split boundary tests
- [x] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：用最小多合约 summary fixture 固化 `5:3:2` 全局交易日边界、左闭右开集合和无重叠约束。
- 改动文件：`FineFT/tests/datahandler/test_commodity_contract_dataset.py`。
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`，新增边界测试先失败后通过。

### Task 2: Manifest builder tests
- [x] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：测试 manifest 记录 `boundaries`、`sets`、合约交易日求交、输入路径、阶段输出路径和 train slice 计划。
- 改动文件：`FineFT/tests/datahandler/test_commodity_contract_dataset.py`。
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`，manifest 相关断言通过。

### Task 3: Boundary and manifest implementation
- [x] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：新增 `FineFT/datahandler/commodity_contract_dataset.py`，实现 summary 读取、边界计算、manifest 构建与 CLI 参数解析。
- 改动文件：`FineFT/datahandler/commodity_contract_dataset.py`、`FineFT/tests/datahandler/test_commodity_contract_dataset.py`。
- 验证方式：运行聚焦 pytest，确认 Task 1 和 Task 2 的测试通过。

### Task 4: Contract stage dataset writer
- [x] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：按 manifest 过滤合约级 `df.feather`，写出 `train/df_<contract>.feather`、`valid/df_<contract>.feather`、`test/df_<contract>.feather`，复制 state features，并确认不生成旧集合大文件。
- 改动文件：`FineFT/datahandler/commodity_contract_dataset.py`、`FineFT/tests/datahandler/test_commodity_contract_dataset.py`。
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q`。

### Task 5: Train slice writer
- [x] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：从合约级 train 阶段文件写出连续编号的 `train/slice/df_*.feather`，保证不跨合约、不跨 train 边界，并记录 slice provenance。
- 改动文件：`FineFT/datahandler/commodity_contract_dataset.py`、`FineFT/tests/datahandler/test_commodity_contract_dataset.py`。
- 验证方式：聚焦 pytest 校验编号连续、slice 只含单合约、`early_stop` 被截断在 train 内。

### Task 6: Valid dynamic slice writer
- [x] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：逐合约执行 valid 动态切片，输出到 `valid/label_*/df_*.feather`，片段编号不覆盖且不跨合约。
- 改动文件：`FineFT/datahandler/commodity_contract_dataset.py`、`FineFT/tests/datahandler/test_commodity_contract_dataset.py`，必要时对 `FineFT/datahandler/slice_model.py` 增加可复用入口。
- 验证方式：聚焦 pytest 使用 stub labeler 或轻量标签器验证多个合约的 label 输出。

### Task 7: VAE data creation compatibility
- [x] **任务完成**（与 superpowers plan `Task 7`、`tasks.md` 对应条目同步勾选）
- 目标：让 `vae_data_creation.py` 在没有 `test.feather` 时读取 `test/df_<contract>.feather` 并合并测试数组，同时保持旧单文件行为。
- 改动文件：`FineFT/datahandler/vae_data_creation.py`、`FineFT/tests/datahandler/test_vae_data_creation.py`。
- 验证方式：`conda activate finetf && pytest FineFT/tests/datahandler/test_vae_data_creation.py -q`。

### Task 8: Commodity data handler scripts
- [x] **任务完成**（与 superpowers plan `Task 8`、`tasks.md` 对应条目同步勾选）
- 目标：直接升级 `commodity_data_handler_fu.sh` 和 `commodity_data_handler_al.sh` 调用新工具，传入 summary、SCALE_SAVE、FEATURE_UNION、输出根目录、日期范围和切片参数，并移除旧单文件 preprocess/slice 调用。
- 改动文件：`FineFT/script/data/commodity_data_handler_fu.sh`、`FineFT/script/data/commodity_data_handler_al.sh`、`FineFT/tests/datahandler/test_commodity_contract_dataset.py` 或新增脚本断言测试。
- 验证方式：聚焦测试或 shell 文本断言确认脚本包含 `commodity_contract_dataset.py` 且不包含旧 `preprocess_data.py --trading_pair` 和 `slice_model.py --data_path dataset/<symbol>/valid.feather`。

### Task 9: Documentation and final validation
- [x] **任务完成**（与 superpowers plan `Task 9`、`tasks.md` 对应条目同步勾选）
- 目标：更新 datahandler 文档，运行 OpenSpec 和聚焦测试，确认规格、计划和实现入口一致。
- 改动文件：`docs/datahandler/data_preparation_analysis.zh_cn.md`、`openspec/changes/add-commodity-contract-dataset-manifest/*`。
- 验证方式：`openspec validate add-commodity-contract-dataset-manifest --strict`，以及 `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_vae_data_creation.py -q`。

## Amendments

### 2026-07-13: Shell-orchestrated valid slicing
- 原因：用户要求 `commodity_contract_dataset.py` 不调用 `slice_model.py`，valid 动态切片改由 shell 独立调度。
- 影响规格：openspec/changes/add-commodity-contract-dataset-manifest/specs/commodity-futures-support/spec.md
- 影响任务：tasks.md 1.10、1.11、1.12

### 2026-07-13: Manifest row counts
- 原因：用户要求 `dataset_manifest.json` 直接给出每个阶段输出文件行数，以及 train/valid/test 阶段总行数。
- 影响规格：openspec/changes/add-commodity-contract-dataset-manifest/specs/commodity-futures-support/spec.md
- 影响任务：tasks.md 1.13

### 2026-07-13: Train short slices and slice row counts
- 原因：用户要求少于 `chunk_length` 的 train 数据也参与训练，并在 manifest 记录切片文件行数。
- 影响规格：openspec/changes/add-commodity-contract-dataset-manifest/specs/commodity-futures-support/spec.md
- 影响任务：tasks.md 1.14

### 2026-07-13: Contract-scoped valid dynamic slices
- 原因：用户要求 valid processed 和 label 切片按合约隔离，避免多个合约之间覆盖。
- 影响规格：openspec/changes/add-commodity-contract-dataset-manifest/specs/commodity-futures-support/spec.md
- 影响任务：tasks.md 1.15

### 2026-07-13: Robust slope labels for small valid segment counts
- 原因：用户运行 valid 动态切片时，部分合约 segment 数量少导致 `Dynamic_labeler` slope 阈值越界。
- 影响规格：openspec/changes/add-commodity-contract-dataset-manifest/specs/commodity-futures-support/spec.md
- 影响任务：tasks.md 1.16

### 2026-07-13: Valid slice manifest
- 原因：用户要求 valid 动态切片后有 manifest 描述每个合约/label 的输出文件和行数，并提供 label 汇总。
- 影响规格：openspec/changes/add-commodity-contract-dataset-manifest/specs/commodity-futures-support/spec.md
- 影响任务：tasks.md 1.17

### 2026-07-13: Contract-scoped VAE test arrays
- 原因：用户要求商品多合约 test npy 不合并，按合约分别写到 `VAE_data/test/`。
- 影响规格：openspec/changes/add-commodity-contract-dataset-manifest/specs/commodity-futures-support/spec.md
- 影响任务：tasks.md 1.18

### 2026-07-13: Contract-scoped VAE valid arrays
- 原因：用户要求商品多合约 valid npy 不跨合约聚合，按合约分别写到 `VAE_data/<contract>/`。
- 影响规格：openspec/changes/add-commodity-contract-dataset-manifest/specs/commodity-futures-support/spec.md
- 影响任务：tasks.md 1.19

## 追加实现步骤

### Task 10: Remove slice model coupling from dataset tool
- [x] **任务完成**（与 superpowers plan `Task 10`、`tasks.md` 对应条目同步勾选）
- 目标：删除 `commodity_contract_dataset.py` 中对 `slice_model.py` 的 import/call、`build_valid_labeler` 和 `write_valid_dynamic_slices` 职责；数据集工具只输出 manifest、阶段数据和 train slices。
- 改动文件：`FineFT/datahandler/commodity_contract_dataset.py`、`FineFT/datahandler/slice_model.py`、`FineFT/tests/datahandler/test_commodity_contract_dataset.py`。
- 验证方式：新增/更新测试断言 `commodity_contract_dataset.py` 不包含 `slice_model` 引用，且 run_dataset_generation 不生成 `valid/label_*`。

### Task 11: Shell orchestrates per-contract slice_model
- [x] **任务完成**（与 superpowers plan `Task 11`、`tasks.md` 对应条目同步勾选）
- 目标：让 `commodity_data_handler_fu.sh` 和 `commodity_data_handler_al.sh` 在新工具完成后，逐个 `valid/df_<contract>.feather` 调用 `slice_model.py --timestamp timestamp`，并避免旧 `dataset/<symbol>/valid.feather` 调用。
- 改动文件：`FineFT/script/data/commodity_data_handler_fu.sh`、`FineFT/script/data/commodity_data_handler_al.sh`、`FineFT/tests/datahandler/test_commodity_contract_dataset.py`。
- 验证方式：脚本文本测试确认包含 per-contract loop 和 `slice_model.py --data_path "${valid_file}"`，且不包含旧 `dataset/<symbol>/valid.feather`。

### Task 12: Update docs and validate amend
- [x] **任务完成**（与 superpowers plan `Task 12`、`tasks.md` 对应条目同步勾选）
- 目标：更新 datahandler 文档描述 shell 调度 valid 动态切片，运行 OpenSpec 和聚焦测试。
- 改动文件：`docs/datahandler/data_preparation_analysis.zh_cn.md`、`openspec/changes/add-commodity-contract-dataset-manifest/*`。
- 验证方式：`openspec validate add-commodity-contract-dataset-manifest --strict`，以及聚焦 datahandler/script 测试。

### Task 13: Manifest output row counts
- [x] **任务完成**（与 superpowers plan `Task 13`、`tasks.md` 对应条目同步勾选）
- 目标：在阶段数据写出后回填 manifest，使每个合约 `output_path` 记录 `output_row_count`，每个集合记录 `contracts_total_count`。
- 改动文件：`FineFT/datahandler/commodity_contract_dataset.py`、`FineFT/tests/datahandler/test_commodity_contract_dataset.py`、`docs/datahandler/data_preparation_analysis.zh_cn.md`。
- 验证方式：先新增失败测试确认 manifest 缺少行数字段，再实现后运行 `conda activate finetf && pytest FineFT/tests/datahandler/test_commodity_contract_dataset.py -q` 和 OpenSpec 校验。

### Task 14: Train short slices and slice row counts
- [x] **任务完成**（与 superpowers plan `Task 14`、`tasks.md` 对应条目同步勾选）
- 目标：`train/slice` 保留不足 `chunk_length` 的合约数据和尾部数据，并在 `slice_outputs` 中记录每个 slice 文件行数。
- 改动文件：`FineFT/datahandler/commodity_contract_dataset.py`、`FineFT/tests/datahandler/test_commodity_contract_dataset.py`、`docs/datahandler/data_preparation_analysis.zh_cn.md`。
- 验证方式：新增失败测试覆盖短 slice 和尾部 slice，修复后运行聚焦 pytest 与 OpenSpec 校验。

### Task 15: Contract-scoped valid dynamic slices
- [x] **任务完成**（与 superpowers plan `Task 15`、`tasks.md` 对应条目同步勾选）
- 目标：`slice_model.py` 将 processed 文件写到 `valid/processed/valid_processed_<contract>.feather`，将 label 切片写到 `valid/<contract>/label_*/df_*.feather`，并让 VAE 递归读取新结构。
- 改动文件：`FineFT/datahandler/slice_model.py`、`FineFT/datahandler/vae_data_creation.py`、`FineFT/tests/datahandler/test_slice_model.py`、`FineFT/tests/datahandler/test_vae_data_creation.py`、`docs/datahandler/data_preparation_analysis.zh_cn.md`。
- 验证方式：新增失败测试覆盖两个合约不覆盖输出和 VAE 读取合约级 label 目录，修复后运行聚焦 pytest 与 OpenSpec 校验。

### Task 16: Robust small-segment slope labels
- [x] **任务完成**（与 superpowers plan `Task 16`、`tasks.md` 对应条目同步勾选）
- 目标：修复 `Dynamic_labeler` 在 slope 模式下 segment 数量少时的 `IndexError`，让小样本 valid 合约仍可切出动态标签。
- 改动文件：`FineFT/datahandler/label_util.py`、`FineFT/tests/datahandler/test_slice_model.py`。
- 验证方式：新增小 segment 数量回归测试，运行聚焦 slice/datahandler 测试与 OpenSpec 校验。

### Task 17: Valid slice manifest
- [x] **任务完成**（与 superpowers plan `Task 17`、`tasks.md` 对应条目同步勾选）
- 目标：`slice_model.py` 生成并增量更新 `valid/slice_manifest.json`，记录 contract 和 label 两个视角下的文件、文件数、文件行数与总行数，跳过空 label。
- 改动文件：`FineFT/datahandler/slice_model.py`、`FineFT/tests/datahandler/test_slice_model.py`、`docs/datahandler/data_preparation_analysis.zh_cn.md`。
- 验证方式：新增 manifest 回归测试，运行聚焦 slice/datahandler 测试与 OpenSpec 校验。

### Task 18: Contract-scoped VAE test arrays
- [x] **任务完成**（与 superpowers plan `Task 18`、`tasks.md` 对应条目同步勾选）
- 目标：商品多合约 `test/df_<contract>.feather` 逐个转换为 `VAE_data/test/test_<contract>.npy`，不再合并成 `VAE_data/test.npy`。
- 改动文件：`FineFT/datahandler/vae_data_creation.py`、`FineFT/tests/datahandler/test_vae_data_creation.py`、`docs/datahandler/data_preparation_analysis.zh_cn.md`。
- 验证方式：新增/更新 VAE 多合约 test 输出测试，运行聚焦 datahandler 测试与 OpenSpec 校验。

### Task 19: Contract-scoped VAE valid arrays
- [x] **任务完成**（与 superpowers plan `Task 19`、`tasks.md` 对应条目同步勾选）
- 目标：商品多合约 `valid/<contract>/label_*/df_*.feather` 逐合约转换为 `VAE_data/<contract>/label_*.npy`，不再跨合约聚合成 `VAE_data/label_*.npy`。
- 改动文件：`FineFT/datahandler/vae_data_creation.py`、`FineFT/tests/datahandler/test_vae_data_creation.py`、`docs/datahandler/data_preparation_analysis.zh_cn.md`。
- 验证方式：新增/更新 VAE 多合约 valid 输出测试，运行聚焦 datahandler 测试与 OpenSpec 校验。
