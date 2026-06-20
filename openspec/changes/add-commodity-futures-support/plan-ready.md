# 实现计划：add-commodity-futures-support

## 来源

- 提案：openspec/changes/add-commodity-futures-support/proposal.md
- 设计：openspec/changes/add-commodity-futures-support/design.md
- 规格：openspec/changes/add-commodity-futures-support/specs/commodity-futures-support/spec.md
- 任务：openspec/changes/add-commodity-futures-support/tasks.md

## 实现步骤

### Task 1: 商品期货数据契约与配置

- [ ] **任务完成**（与 superpowers plan `Task 1`、`tasks.md` 对应条目同步勾选）
- 目标：建立商品期货配置、深度列名、时间聚合和 reward/execution manifest 基础能力，使后续模块拥有稳定接口。
- 改动文件：新增 `data_preprocess/operator_futures/commodity/config.py`、`data_preprocess/operator_futures/commodity/schema.py`、`data_preprocess/operator_futures/commodity/__init__.py`；新增测试 `data_preprocess/tests/test_commodity_config_schema.py`。
- 验证方式：运行 `python -m pytest data_preprocess/tests/test_commodity_config_schema.py -q`，确认 `fu` 配置、depth=5 列名、右闭右标 resample 参数和 manifest 选择通过。

### Task 2: 主力合约拼接

- [ ] **任务完成**（与 superpowers plan `Task 2`、`tasks.md` 对应条目同步勾选）
- 目标：将现有主力识别算法整理为可调用商品期货模块，从 `data/原始下载/{品种中文名}/{YYYY}` 扫描原始数据，按 `TradingDay` 选择日级主力，使用 `ActionDay + UpdateTime` 生成真实时间戳，并输出审计元数据。
- 改动文件：新增 `data_preprocess/operator_futures/commodity/main_contract.py`；按需迁移 `openspec/specs/support_shanghai_futures/step2_extract_main_contract_data.py` 的算法；新增测试 `data_preprocess/tests/test_commodity_main_contract.py`。
- 验证方式：运行 `python -m pytest data_preprocess/tests/test_commodity_main_contract.py -q`，覆盖前一交易日选择、当前日 fallback、夜盘时间戳和不复权拼接。

### Task 3: 商品期货下采样

- [ ] **任务完成**（与 superpowers plan `Task 3`、`tasks.md` 对应条目同步勾选）
- 目标：实现三类商品期货下采样输出：参考价/funding 兼容列、真实 5 档盘口、秒频成交估计和 quote 聚合特征。
- 改动文件：新增 `data_preprocess/operator_futures/commodity/downscale.py`；按需新增 CLI 包装脚本 `data_preprocess/operator_futures/commodity/downscale_single_day.py`；新增测试 `data_preprocess/tests/test_commodity_downscale.py`。
- 验证方式：运行 `python -m pytest data_preprocess/tests/test_commodity_downscale.py -q`，使用 `docs/上海商品交易所/fu2302.csv` 验证 LastPrice/midprice、depth=5、`Turnover.diff()/Volume.diff()`、tick rule、quote 空窗口报错和右闭右标。

### Task 4: 特征管线适配

- [ ] **任务完成**（与 superpowers plan `Task 4`、`tasks.md` 对应条目同步勾选）
- 目标：让 cross-section、merge/concat、time feature、feature selection、scale/save 可读取商品期货 manifest 和 depth=5，移除商品路径中的 25 档与前 106 列硬编码。
- 改动文件：修改 `data_preprocess/operator_futures/cross_section/base_feature_util.py`、`data_preprocess/operator_futures/cross_section/create_feature.py`、`data_preprocess/operator_futures/merge_concat/merge.py`、`data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py`、`data_preprocess/operator_futures/feature_selection/ic_correlation.py`、`data_preprocess/operator_futures/scale_describe_save/scale_save.py`；新增测试 `data_preprocess/tests/test_commodity_feature_pipeline.py`。
- 验证方式：运行 `python -m pytest data_preprocess/tests/test_commodity_feature_pipeline.py -q`，确认 depth=5 snapshot/KLINE/QUOTE 特征生成、6-25 档不生成、价差 target 保持不变、scale/save 使用 manifest。

### Task 5: 商品期货环境支持

- [ ] **任务完成**（与 superpowers plan `Task 5`、`tasks.md` 对应条目同步勾选）
- 目标：让 FineFT 环境可用 depth=5 商品数据初始化和 step，关闭 funding，按买入/卖出费率扣交易成本，并在深度不足时报错。
- 改动文件：新增或修改 `FineFT/env/env_initiate/commodity_initiate.py`、`FineFT/env/env_class/commodity_env.py`、`FineFT/env/env_class/futures_util.py`；新增测试 `FineFT/env/test_commodity_env.py`。
- 验证方式：运行 `python -m pytest FineFT/env/test_commodity_env.py -q`，确认 `reset()`、一次 `step()`、买卖手续费、无 funding countdown 和深度不足 fail-fast。

### Task 6: 脚本入口与文档

- [ ] **任务完成**（与 superpowers plan `Task 6`、`tasks.md` 对应条目同步勾选）
- 目标：提供商品期货端到端脚本入口和文档，使用户按本地目录放置数据后能复现从主力连续化到 scale/save 的流程。
- 改动文件：新增或修改 `data_preprocess/script_preprocess/future_upgraded/commodity/*.sh`、`data_preprocess/README.zh_CN.md`、`data_preprocess/README.md`、按需新增 `docs/上海商品交易所/commodity_futures_preprocess.md`。
- 验证方式：运行脚本语法检查 `bash -n data_preprocess/script_preprocess/future_upgraded/commodity/*.sh`，并确认文档列出输入目录、输出 `fu`、废弃特征、estimated 特征和 smoke-test 命令。

### Task 7: 验证与回归

- [ ] **任务完成**（与 superpowers plan `Task 7`、`tasks.md` 对应条目同步勾选）
- 目标：运行 OpenSpec、商品期货单元/烟测和加密货币回归烟测，记录受限于完整原始数据或 GPU 的跳过项。
- 改动文件：更新 `openspec/changes/add-commodity-futures-support/tasks.md` checkbox 状态；按需新增验证记录 `openspec/changes/add-commodity-futures-support/verification.md`。
- 验证方式：运行 `openspec validate add-commodity-futures-support --strict`、`python -m pytest data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_main_contract.py data_preprocess/tests/test_commodity_downscale.py data_preprocess/tests/test_commodity_feature_pipeline.py FineFT/env/test_commodity_env.py -q`，并运行一个现有 crypto 环境 smoke test 或记录缺少数据时的跳过原因。
