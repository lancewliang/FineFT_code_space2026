## 1. Implementation

- [x] 1.1 Add tests for commodity VAE cross-contract train-set materialization, manifest contents, missing label handling, and array dimension validation. <!-- 已实现: 新增商品 VAE 多合约训练集合并、manifest、缺失 label 和维度校验测试 -->
- [x] 1.2 Implement commodity VAE data discovery and train-set materialization in `FineFT/RL/DiHFT/VAE/merge_vae_train.py` and integrate it from `main.py`. <!-- 已实现: merge_vae_train.py 支持发现合约 label 数组、合并训练集并写 manifest；main.py 负责调用编排 -->
- [x] 1.3 Add tests for per-contract VAE analysis outputs, traceable CSV rows, aggregate `ood_logpx_all` outputs, and `summary.json` statistics. <!-- 已实现: 新增测试覆盖 test source discovery、分合约 logpx npy/csv、总体输出和 summary -->
- [x] 1.4 Refactor `FineFT/RL/DiHFT/VAE/process.py`, `main.py`, and `summary.py` analysis flow to read `VAE_data/test/test_<contract>.npy`, emit per-contract `.npy/.csv`, aggregate `.npy/.csv`, and `summary.json`. <!-- 已实现: process.py/main.py 支持逐合约测试 loader，summary.py 负责分析输出与总体汇总写出 -->
- [x] 1.5 Replace ambiguous boolean CLI usage in `FineFT/RL/DiHFT/VAE/main.py` with explicit `--train` and `--analyze-only` behavior for the commodity VAE workflow. <!-- 已实现: main.py 新增明确 workflow flags 并移除商品入口对 if_train/if_cross_analyze 的依赖 -->
- [x] 1.6 Update `FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh` to activate `finetf`, set `PYTHONPATH`, pass `--dataset_name fu`, pass `--data_base_path dataset/10min`, and launch `label_0..label_4` with the explicit training flag and max-2 default concurrency. <!-- 已实现: fu VAE shell 入口激活环境、设置 PYTHONPATH、传 --train 多 label 启动，并默认最多 2 个并发训练进程 -->
- [x] 1.7 Add tests for enhanced per-label `summary.json` metrics: train baseline, test quantiles, acceptance against train thresholds, and input/analyzed sample integrity. <!-- 已实现: 新增 enhanced summary 测试覆盖 train_baseline、quantiles、acceptance 和 sample_mismatch -->
- [x] 1.8 Implement enhanced per-label summary statistics in `FineFT/RL/DiHFT/VAE/summary.py` and wire training baseline logpx collection from `main.py`. <!-- 已实现: summary.py 输出 train_baseline、quantiles、acceptance、sample_mismatch，main.py 分析训练集 baseline，process.py 只委托写出 -->
- [x] 1.9 Add tests for cross-label `routing_summary.json` winner distribution, top1/top2 margin, low-margin rate, and mismatched label sample counts. <!-- 已实现: 新增 routing_summary 测试覆盖 winner_counts、margin、low_margin 和 sample_mismatch -->
- [x] 1.10 Implement cross-label routing summary generation after label analysis outputs are available. <!-- 已实现: summary.py 生成 routing_summary.json，main.py 在 analyze_contracts() 后检查所有 label 输出齐全时自动写出；无 args.routing_summary 或 shell 后处理 snippet -->

## 2. Verification

- [x] 2.1 Run `conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests -q` or the focused VAE tests if the full suite is blocked by unrelated environment fixtures. <!-- 已实现: 全量 FineFT tests 88 passed -->
- [x] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/VAE/main.py FineFT/RL/DiHFT/VAE/process.py FineFT/RL/DiHFT/VAE/merge_vae_train.py FineFT/RL/DiHFT/VAE/summary.py`. <!-- 已实现: VAE main.py/process.py/merge_vae_train.py/summary.py py_compile 通过 -->
- [x] 2.3 Run `bash -n FineFT/script/train/DiHFT/low_level/VAE_util_fu.sh`. <!-- 已实现: shell syntax 校验通过 -->
- [x] 2.4 Run `openspec validate support-commodity-vae-cross-contract --strict`. <!-- 已实现: OpenSpec strict validation 通过 -->
- [x] 2.5 Re-run focused commodity VAE tests, VAE py_compile, shell syntax validation, and OpenSpec strict validation after the summary metric extension. <!-- 已实现: focused tests、py_compile、shell syntax 和 OpenSpec strict validation 均通过 -->
