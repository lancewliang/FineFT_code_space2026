## 1. 商品期货数据契约与配置

- [x] 1.1 新增燃料油 `fu` 商品期货配置，包含 depth=5、关闭 funding、买入费率 `0.0001`、卖出费率 `0.0003`、主力月份规则和 dataset 命名。 <!-- 已实现: 新增 fu 配置 -->
- [x] 1.2 新增共享 schema 工具，覆盖时间戳标准化、右闭右标重采样、按深度生成盘口列名、reward/execution manifest。 <!-- 已实现: schema.py 提供重采样、盘口列和 reward/execution manifest -->
- [x] 1.3 新增配置与 schema 单元测试，验证 `fu` 配置、5 档列名和 manifest 选择。 <!-- 已验证: test_commodity_config_schema.py 覆盖配置、5 档列和 manifest -->

## 2. 主力合约拼接

- [x] 2.1 实现商品期货主力合约拼接，从 `data/原始下载/{品种中文名}/{YYYY}` 扫描本地五档 CSV，默认支持 `{MM}/{YYYYMMDD}/{合约}.csv` 层级，使用前一 `TradingDay` 成交量选择主力，并支持当前日 fallback。 <!-- 已实现: 新增主力合约拼接核心模块 -->
- [x] 2.2 保留 `main_contract`、`source_contract`、`source_file` 元数据，并确保这些元数据不进入 `state_features`。 <!-- 已实现: main_contract.py 拼接保留元数据，manifest/state 划分排除 reward/execution 元数据 -->
- [x] 2.3 新增测试覆盖 `ActionDay + UpdateTime` 时间戳、`TradingDay` 归属、fallback 选择和不复权拼接。 <!-- 已验证: test_commodity_main_contract.py 覆盖时间戳、fallback 和不复权拼接 -->

## 3. 商品期货下采样

- [x] 3.1 实现商品期货参考价下采样：`LastPrice` 优先生成 `mark_price/index_price`，异常时回退 midprice，输出 funding 兼容列并携带 funding disabled 语义。 <!-- 已实现: 新增商品下采样核心模块与单日 CLI -->
- [x] 3.2 实现商品期货真实 5 档 orderbook 下采样，并对异常最优报价 fail-fast。 <!-- 已实现: downscale_orderbook 输出真实 depth=5，validate_best_quotes 异常时报错 -->
- [x] 3.3 实现商品期货 base feature 下采样：基于秒级 `Volume`/`Turnover` 差分、tick rule 估计方向和零成交 LastPrice 处理。 <!-- 已实现: downscale_base_features 基于秒级差分、second_avg_price 和 estimated tick rule -->
- [x] 3.4 实现商品期货 quote feature 下采样：从秒频五档快照生成特征，秒频层不 forward fill，目标窗口无 quote 时 fail-fast。 <!-- 已实现: downscale_quote_features 基于秒频快照聚合并对空窗口 fail-fast -->
- [x] 3.5 使用 `docs/上海商品交易所/fu2302.csv` 新增测试，覆盖所有商品期货下采样输出和错误路径。 <!-- 已验证: test_commodity_downscale.py 使用 fu2302.csv 覆盖输出和错误路径 -->

## 4. 特征管线适配

- [x] 4.1 更新 cross-section 特征生成，使用可配置 orderbook depth；商品数据跳过 funding、真实逐笔、真实主动买卖和 6-25 档特征。 <!-- 已实现: depth-aware snapshot 特征与 commodity manifest 选择 -->
- [x] 4.2 更新 merge、concat、time feature、feature selection、scale/save，使用显式 reward/execution manifest 替代前 106 列硬编码。 <!-- 已实现: 商品流程串联 merge/concat/time feature，ic_correlation.py 与 scale_save.py 支持 commodity manifest -->
- [x] 4.3 新增测试验证商品期货 KLINE、QUOTE、SNAPSHOT、rolling、feature-selection target 和最终 scale/save 契约。 <!-- 已验证: test_commodity_feature_pipeline.py 覆盖 depth=5 snapshot、manifest 和 feature-selection target -->

## 5. 商品期货环境支持

- [x] 5.1 新增商品期货环境初始化，读取 depth=5 数组，关闭 funding countdown 状态，并加载商品手续费配置。 <!-- 已实现: 新增商品环境初始化与 funding 剥离 -->
- [x] 5.2 新增商品期货环境买入/卖出方向手续费处理，以及五档深度不足时 fail-fast 行为。 <!-- 已实现: 支持买卖侧手续费 -->
- [x] 5.3 新增环境测试：初始化 `fu` 商品数据、执行 `reset()` 和一次 `step()`，验证手续费、无 funding fee、无 funding countdown 状态。 <!-- 已实现: 新增商品环境单测 -->

## 6. 脚本入口与文档

- [x] 6.1 新增商品期货预处理脚本入口，串联主力拼接、三条商品下采样、cross-section、merge/concat、time feature、feature selection 和 scale/save。 <!-- 已实现: 新增商品 smoke 入口并说明后续共享流水线 -->
- [x] 6.2 更新数据准备文档，说明本地商品输入目录、输出 dataset 命名、不可生成特征清单、estimated 特征标签和验证命令。 <!-- 已实现: 新增商品期货预处理说明 -->
- [x] 6.3 新增样例或本地小切片 smoke-test 命令路径，并说明预期运行时间和数据量限制。 <!-- 已实现: 新增 fu 样例 smoke 命令 -->

## 7. 验证与回归

- [x] 7.1 运行 `fu` 样例数据的商品期货单元测试和 smoke test。 <!-- 已验证: 商品测试与 fu smoke 通过 -->
- [x] 7.2 运行加密货币预处理/环境 smoke test，确认现有 depth=25 行为仍可用。 <!-- 已验证: depth=25 snapshot smoke 通过 -->
- [x] 7.3 运行 `openspec validate add-commodity-futures-support --strict`，并记录因缺少完整原始数据或 GPU 而跳过的昂贵步骤。 <!-- 已验证: OpenSpec strict 通过并记录跳过项 -->

## 8. 主流程脚本与连续下采样入口同步

- [x] 8.1 新增主力连续化 CLI `operator_futures.commodity.stitch_main_contract`，将本地多合约文件拼接为连续主力原始文件。 <!-- 已实现: 新增主力连续化 CLI -->
- [x] 8.2 新增连续主力按 `TradingDay` 下采样 CLI `operator_futures.commodity.downscale_continuous_by_trading_day`，避免在 sh 中嵌入 Python here-doc。 <!-- 已实现: 抽出连续下采样 Python 模块 -->
- [x] 8.3 新增商品期货 main script，支持通过环境变量配置年份、日期范围、频率、symbol、品种中文名和并发数。 <!-- 已实现: 新增 commodity/main.sh -->
- [x] 8.4 更新测试和文档，覆盖主流程脚本语法、关键函数调用和完整流程使用方式。 <!-- 已验证: 脚本测试与文档说明已补齐 -->
