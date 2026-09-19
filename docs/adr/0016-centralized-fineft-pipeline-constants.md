# FineFT 管道公共常量与产出物命名契约体系

为了消除各阶段 Python 脚本中散落的硬编码字符串，增强上下游代码的数据血缘与依赖可见性，我们在 `FineFT/common/` 下建立模块化的强类型常量契约体系，统一管理所有文件产出物与 DataFrame 字段。

## 决策背景与上下文

在 `fu.readme.md` 梳理的 30min 完整流水线中，包含了从数据准备、低层强化学习训练、回测评估、低层筛选、VAE 训练、高层 Optuna 路由寻优到最终启发式选拔与可视化的 8 个主要阶段（步骤 2~8）。

目前，这些阶段的中间产出物文件名（如 `analysis_result.csv`、`optuna_results.csv`、`reward_history.npy` 等）与评估/动作 DataFrame 字段名（如 `tr`、`mdd`、`annual_sr`、`turnover`、`position` 等）均以散落的裸字符串硬编码在各个脚本中：
1. **依赖隐晦**：下游脚本（如 `DiHFT_high_level_heurstic.py`）无法通过静态 `import` 显示声明它依赖于上游（如 `test_agent_index.py` 或 `vae_routing_util.py`）产出的具体文件，代码间调用关系割裂。
2. **笔误隐患**：例如历史遗留的 `maintain_marigine_history.npy`（多了一个字符 `i`）被跨脚本手工复制多次，一旦某处重命名或拼写不一致便会引发难以排查的 `FileNotFoundError` 或 `KeyError`。
3. **动态格式分散**：像 `df_{index}.feather`、`trading_action_detail_epoch_{epoch}.csv` 等带有占位符的文件名，在多个文件中使用不同的 f-string 拼接，缺乏集中校验。

## 决策内容

1. **作用域严格限定在 FineFT**：
   重构仅作用于 `FineFT/` 目录下的 Python 脚本相互引用；`data_preprocess/` 算子库维持原有解耦状态，其产出物（如 `dataset_split_manifest.json`、`state_features.npy`）在 FineFT 侧以标准化输入契约形式引用。

2. **分类分文件独立存储 (`FineFT/common/`)**：
   在 `FineFT/common/` 目录下按关注点划分为 3 个独立的 Python 常量文件：
   - `artifacts.py`：管理所有磁盘静态产出物文件名（`ArtifactNames`，如 `TRAINED_MODEL_PKL`、`DATASET_MANIFEST_JSON`、`ANALYSIS_RESULT_CSV`、`SELECTION_MANIFEST_JSON`、`OPTUNA_RESULTS_CSV`、`HIGH_LEVEL_AGENT_PARA_TXT`）、仿真历史向量文件名（`HistoryArtifactNames`，如 `REWARD_HISTORY_NPY`、`WALLET_BALANCE_HISTORY_NPY` 等）以及动态文件名的纯函数构造器（如 `get_df_chunk_filename`、`get_trading_detail_csv_filename` 等）。
   - `metric_columns.py`：管理所有财务绩效、选拔评估与 Optuna 调优相关的 DataFrame 列名（`MetricColumns`，如 `TR = "tr"`、`PORTFOLIO_TR = "portfolio_tr"`、`MDD = "mdd"`、`ANNUAL_SR = "annual_sr"`、`DAILY_CR = "daily_cr"`、`REQUIRED_MONEY = "required_money"` 等）。
   - `trade_columns.py`：管理所有单 Step 执行、持仓动作、交易明细与行情约束列名（`TradeColumns`，如 `POSITION = "position"`、`TURNOVER = "turnover"`、`REALIZED_PNL_STEP = "realized_pnl_step"`、`IS_LIMIT_UP = "is_limit_up"` 等），并集中维护双语导出字典 `CSV_HEADER_LABELS`。

3. **门面统一导出 (Facade Export)**：
   `FineFT/common/__init__.py` 统一暴露上述命名空间类及关键构造辅助函数，既支持直观的顶层门面导入（`from common import ArtifactNames, MetricColumns`），又完全支持精准按子模块导入（`from common.artifacts import ArtifactNames`）。

4. **显式扩展名变量命名**：
   静态产出物常量变量名显式包含扩展名后缀（如 `ANALYSIS_RESULT_CSV` vs `ANALYSIS_RESULT_NPY`），彻底消除同名不同格式文件的语义冲突。

5. **历史字面量 100% 兼容**：
   对于 `maintain_marigine_history.npy` 等包含历史拼写偏差的字面量，常量变量名采用规范命名 `MAINTAIN_MARGIN_HISTORY_NPY`，底层字符串值严格保持不变，确保与存量磁盘实验数据完全兼容。

6. **全链路覆盖 fu 脚本**：
   全面覆盖 `fu.readme.md` 步骤 2~8 涉及的核心脚本（`commodity_contract_dataset.py`, `valid_cross_contract_label_calibration.py`, `vae_data_creation.py`, `weight_advantage_pretrain.py`, `test_agent_index.py`, `FineFT_single_agent_with_different_position.py`, `VAE/main.py`, `vae_routing_optuna.py`, `vae_routing_util.py`, `DiHFT_high_level_heurstic.py`）。

## 影响与后果

- **开发体验**：IDE 补全与静态代码分析全面支持，下游代码通过 `import` 语句即可清晰展现整条流水线的数据血缘。
- **运行安全**：彻底杜绝由于字符串手滑或参数拼写不一致引发的隐蔽运行时缺陷。
- **系统边界**：业务脚本不再散落任何字面量文件名和 DataFrame 字段名，契约变更只需在 `FineFT/common/` 集中修改。
