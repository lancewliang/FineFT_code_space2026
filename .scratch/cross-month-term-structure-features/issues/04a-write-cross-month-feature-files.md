# 04a — 写出 CROSS_MONTH_FEATURE 文件

**要构建什么:** 为 `cross_month_feature.py` 增加可执行生成入口，并在商品期货 full-process shell 中调用它，使每个交易日、每个合约在 daily merge 前都有对应的 `CROSS_MONTH_FEATURE` 文件。生成入口必须读取 main contract summary，使用交易日 T 之前最近一个可用交易日的 `main/sub/other` 角色，加载当前合约、主力合约、次主力合约和活跃交割月合约的 target-frequency bars，生成固定宽度跨月合约结构特征并写出 feather。

**被阻塞于:** 03 — Generate Delivery-Month Sequence Features.

**Status:** done

- [x] `cross_month_feature.py` 提供 CLI 或批处理入口，能按 symbol、contract、target_freq、date 和 summary 生成单日 `CROSS_MONTH_FEATURE`。
- [x] 生成入口调用 `resolve_previous_main_sub_role`，确保 T 日特征使用 T 之前最近一个可用交易日的角色。
- [x] 输出路径为 `PREPROCESS_DATASET/commodity-futures/CROSS_MONTH_FEATURE/{symbol}/{contract}/{target_freq}/{YYYY-MM-DD}.feather`。
- [x] 输出包含 `timestamp` 和固定宽度 `CROSS_MONTH_FEATURE_COLUMNS`，且不包含 Reward/Execution 列。
- [x] 商品期货 full-process shell 在 daily merge 前运行跨月特征生成步骤。
- [x] 商品期货 full-process shell 先完成所有合约的 `cross_section`，再遍历所有合约运行 `cross_month_feature` 和后续流程。
- [x] 测试验证 CLI/shell 能实际触发生成，并且 daily merge required 模式能消费该文件。
