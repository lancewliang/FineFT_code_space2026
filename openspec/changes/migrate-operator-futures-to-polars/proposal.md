# migrate-operator-futures-to-polars

## 背景与目标

`data_preprocess/operator_futures` 当前核心预处理链路大量依赖 pandas。随着数据规模增大，CSV/Feather 读写、时间窗口聚合、跨日拼接、特征合并和前向填充等步骤性能不足。目标是在不改变外部使用方式和输出结构的前提下，将该目录下核心预处理实现原地迁移到 Polars，并使端到端预处理耗时预期至少降低 30%。

## 用户场景

- 研究或训练前，用户按现有脚本入口和 CLI 参数运行 Binance futures 预处理，希望更快生成同名中间 Feather 和最终特征文件。
- 用户按现有商品期货流程运行主力合约拼接、商品降采样、cross-section、scale、feature selection 等步骤，希望商品链路同样使用 Polars 加速。
- 下游训练、分析和已有调度脚本继续读取相同目录结构、文件名、列名、列顺序和 timestamp 语义的产物。

## 设计方向

采用原地一次性迁移方案：新建独立 OpenSpec 变更 `migrate-operator-futures-to-polars`，实现阶段直接替换 `data_preprocess/operator_futures` 核心预处理代码为 Polars，不新增 pandas legacy 文件，不提供 pandas/Polars 双实现开关。旧 pandas 实现仅通过 git 历史和迁移前测试结果作为参考。

外部接口保持不变，包括脚本路径、模块入口、CLI 参数、默认路径、输出目录结构和 Feather 文件命名。`polars` 作为 `data_preprocess` 运行环境的必需依赖。

迁移范围覆盖两条核心链路：

- Binance futures 链路：`orderbook_25`、`derivative_ticker`、`features_related`、`cross_section`、`merge_concat`、`merge_all`、`scale_describe_save`、`feature_selection` 中核心预处理相关 pandas 用法。
- 商品期货链路：`operator_futures/commodity` 下 `main_contract.py`、`stitch_main_contract.py`、`downscale.py`、`downscale_single_day.py`、`downscale_continuous_by_trading_day.py`、`schema.py`、`config.py` 中涉及 DataFrame、CSV、Feather、聚合、拼接和 schema 的逻辑。

共享入口和 `market_type` 分支也纳入迁移范围，例如 `cross_section/create_feature.py`、`scale_describe_save/scale_save.py`、`feature_selection/ic_correlation.py`。

输出兼容性优先于重构：同样输入和同样 CLI 参数下，输出应尽量保持文件级兼容，包括列名、列顺序、timestamp 语义、行数、排序、Feather schema 和 dtype。浮点数值验证允许 `rtol=1e-12, atol=1e-12`。

如果迁移中发现 pandas 旧行为疑似 bug，或 Polars 难以完全复刻既有 schema/dtype 行为，不直接静默改变。应形成兼容性清单，列明文件、函数、输入场景、pandas 输出、Polars 预期输出和建议处理方式，再由用户逐项确认。

错误处理沿用现有脚本的直接失败风格。缺文件、缺列、非法日期范围、无可交易主力合约、商品盘口 best quote 校验失败等情况继续抛出明确异常。商品期货现有校验语义和异常信息尽量保持不变。

测试以现有小样例和已有测试数据为边界，不新增大样本 benchmark 脚本。实现阶段应补强输出兼容断言，并用现有样例跑通 commodity 主力合约拼接、commodity downscale、Binance futures 核心 downscale/merge/cross-section 的最短链路。

## 关键决策

- 新建独立变更 `migrate-operator-futures-to-polars`，不并入现有 `add-commodity-futures-support`。
- 原地一次性迁移 `data_preprocess/operator_futures` 核心预处理代码到 Polars。
- 明确覆盖 Binance futures 链路和 `operator_futures/commodity` 商品期货专用链路。
- 保持脚本路径、模块入口、CLI 参数、默认路径、输出目录、Feather 文件名兼容。
- 将 `polars` 作为必需依赖，不保留 pandas legacy 文件。
- 输出尽量文件级兼容；浮点比较允许 `rtol=1e-12, atol=1e-12`。
- 疑似旧 bug 或无法完全复刻的 schema/dtype 差异先进入兼容性清单，由用户逐项决定。
- 不新增 benchmark 脚本；性能目标作为验收标准，用人工命令记录端到端耗时。

## 范围边界

**包含：**

- `data_preprocess/operator_futures/commodity` 商品期货专用预处理链路。
- `data_preprocess/operator_futures/orderbook_25`、`derivative_ticker` 降采样链路。
- `data_preprocess/operator_futures/features_related` 特征计算链路。
- `data_preprocess/operator_futures/cross_section` 横截面特征链路。
- `data_preprocess/operator_futures/time_operator` 时间特征链路。
- `data_preprocess/operator_futures/merge_concat`、`merge_all` 特征合并与跨日拼接链路。
- `data_preprocess/operator_futures/scale_describe_save`、`feature_selection` 中核心预处理相关 pandas 用法。
- `crypto_futures` 和 `commodity_futures` 相关 `market_type` 分支的兼容处理。
- `data_preprocess` 依赖声明中加入必需的 `polars`。
- 基于现有小样例和已有测试数据的兼容性测试调整。

**不包含（本次）：**

- 训练目录 `FineFT/**` 的 pandas 迁移。
- 文档中的 pandas 示例迁移，除非测试直接依赖。
- 纯分析脚本或 notebook 的 pandas 迁移。
- 新增长期 benchmark 框架或大样本性能测试数据。
- 引入 pandas/Polars 双实现开关或 legacy pandas 模块。
- 改变预处理输出目录结构、文件命名、CLI 参数或下游读取契约。

## 验收标准

- [ ] `polars` 已作为 `data_preprocess` 运行环境的必需依赖声明。
- [ ] `data_preprocess/operator_futures` 核心预处理路径不再依赖 pandas 作为主要 DataFrame 引擎。
- [ ] Binance futures 核心 downscale、feature、cross-section、merge、concat、merge_clean 链路使用 Polars 实现并保持现有 CLI 和输出路径兼容。
- [ ] 商品期货 `operator_futures/commodity` 主力合约、连续拼接、降采样、schema 相关链路使用 Polars 实现并保持现有 CLI 和输出路径兼容。
- [ ] 同样输入和 CLI 参数下，输出列名、列顺序、timestamp 语义、排序、行数和关键 Feather schema/dtype 尽量与迁移前一致。
- [ ] 浮点数值输出与迁移前结果按 `rtol=1e-12, atol=1e-12` 对齐。
- [ ] 疑似 pandas 旧 bug 或无法完全复刻的 schema/dtype 差异已记录到兼容性清单，并等待用户逐项确认。
- [ ] 现有 `data_preprocess/tests/test_commodity_*` 相关测试通过。
- [ ] 基于现有小样例的最短 Binance futures 预处理链路和商品期货预处理链路能生成预期文件。
- [ ] 人工记录一次迁移前后端到端预处理耗时，预期总耗时至少降低 30%；不新增 benchmark 脚本。
