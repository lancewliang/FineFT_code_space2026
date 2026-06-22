# migrate-commodity-preprocess-to-polars

## 背景与目标

商品主流程已经在前段完成 Polars 化，但后半段仍有共享模块保留 pandas，实现上继续依赖 rolling、concat、corr 和 feather I/O。当前目标是把这些共享后半段迁移到 Polars 原生实现，统一商品与通用预处理链路的数据处理方式，减少 pandas 依赖并保持现有输出兼容。

## 用户场景

- 商品主流程从连续合约拼接、按日 downscale、time feature、IC 选择到 scale/save 能完整跑通。
- 5 档商品盘口与 25 档通用盘口都能在同一套后半段逻辑中处理。
- 迁移后，目标文件不再 import pandas，但对外 CLI、文件路径、列名保持兼容。

## 设计方向

采用分层 Polars 原生迁移：

- `time_operator/create_feature_multi_processing.py` 改为 Polars 实现，不再保留 multiprocessing 外壳，使用 Polars rolling / expression 生成 time feature。
- `time_operator/multi_processing_util.py` 收敛为 Polars helper 层，承接 OHLCV、OHLC、price/size rolling 的通用逻辑。
- `feature_selection` 目录整体迁移到 Polars，包含 `ic_correlation.py`、`rank_ic_correlation.py`、`catbooost.py`、`remove_duplicates_feature.py`、`lasso_linear.py`、`cor_util.py`。
- `scale_describe_save/scale_save.py` 改为 Polars 读写与缩放实现。
- 允许在 CatBoost / sklearn 等外部模型边界前做 numpy 或库原生结构转换，但不在目标文件中引入 pandas。

## 关键决策

- 迁移范围覆盖共享后半段模块，不只商品链路。
- 目标文件迁移后不再 `import pandas`。
- `time_operator` 不保留 multiprocessing 外壳。
- CLI 参数、输出路径、文件名、列名保持兼容。
- 数值允许浮点微小误差，不要求逐列完全一致。
- 不设置性能验收指标，只要求正确性、兼容性和端到端可跑通。

## 范围边界

**包含：**
- `data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py`
- `data_preprocess/operator_futures/time_operator/multi_processing_util.py`
- `data_preprocess/operator_futures/feature_selection/*`
- `data_preprocess/operator_futures/scale_describe_save/scale_save.py`
- 相关测试与商品脚本入口参数调整

**不包含（本次）：**
- 其他未涉及的预处理模块重构
- 训练模型本身的算法改造
- 性能基准体系建设

## 验收标准

- [ ] 目标文件不再 `import pandas`
- [ ] 商品 5 天默认 `main.sh` 可完整跑通
- [ ] 现有 CLI 输出路径和文件名保持兼容
- [ ] 商品 5 档、通用 25 档都能通过回归测试
- [ ] 端到端流程无 `Traceback / FileNotFound / KeyError`
