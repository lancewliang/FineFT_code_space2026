## 1. 商品期货数据契约与配置

- [ ] 1.1 新增燃料油 `fu` 商品期货配置，包含 depth=5、关闭 funding、买入费率 `0.0001`、卖出费率 `0.0003`、主力月份规则和 dataset 命名。
- [ ] 1.2 新增共享 schema 工具，覆盖时间戳标准化、右闭右标重采样、按深度生成盘口列名、reward/execution manifest。
- [ ] 1.3 新增配置与 schema 单元测试，验证 `fu` 配置、5 档列名和 manifest 选择。

## 2. 主力合约拼接

- [ ] 2.1 实现商品期货主力合约拼接，从本地五档 CSV 读取数据，使用前一 `TradingDay` 成交量选择主力，并支持当前日 fallback。
- [ ] 2.2 保留 `main_contract`、`source_contract`、`source_file` 元数据，并确保这些元数据不进入 `state_features`。
- [ ] 2.3 新增测试覆盖 `ActionDay + UpdateTime` 时间戳、`TradingDay` 归属、fallback 选择和不复权拼接。

## 3. 商品期货下采样

- [ ] 3.1 实现商品期货参考价下采样：`LastPrice` 优先生成 `mark_price/index_price`，异常时回退 midprice，输出 funding 兼容列并携带 funding disabled 语义。
- [ ] 3.2 实现商品期货真实 5 档 orderbook 下采样，并对异常最优报价 fail-fast。
- [ ] 3.3 实现商品期货 base feature 下采样：基于秒级 `Volume`/`Turnover` 差分、tick rule 估计方向和零成交 LastPrice 处理。
- [ ] 3.4 实现商品期货 quote feature 下采样：从秒频五档快照生成特征，秒频层不 forward fill，目标窗口无 quote 时 fail-fast。
- [ ] 3.5 使用 `docs/上海商品交易所/fu2302.csv` 新增测试，覆盖所有商品期货下采样输出和错误路径。

## 4. 特征管线适配

- [ ] 4.1 更新 cross-section 特征生成，使用可配置 orderbook depth；商品数据跳过 funding、真实逐笔、真实主动买卖和 6-25 档特征。
- [ ] 4.2 更新 merge、concat、time feature、feature selection、scale/save，使用显式 reward/execution manifest 替代前 106 列硬编码。
- [ ] 4.3 新增测试验证商品期货 KLINE、QUOTE、SNAPSHOT、rolling、feature-selection target 和最终 scale/save 契约。

## 5. 商品期货环境支持

- [ ] 5.1 新增商品期货环境初始化，读取 depth=5 数组，关闭 funding countdown 状态，并加载商品手续费配置。
- [ ] 5.2 新增商品期货环境买入/卖出方向手续费处理，以及五档深度不足时 fail-fast 行为。
- [ ] 5.3 新增环境测试：初始化 `fu` 商品数据、执行 `reset()` 和一次 `step()`，验证手续费、无 funding fee、无 funding countdown 状态。

## 6. 脚本入口与文档

- [ ] 6.1 新增商品期货预处理脚本入口，串联主力拼接、三条商品下采样、cross-section、merge/concat、time feature、feature selection 和 scale/save。
- [ ] 6.2 更新数据准备文档，说明本地商品输入目录、输出 dataset 命名、不可生成特征清单、estimated 特征标签和验证命令。
- [ ] 6.3 新增样例或本地小切片 smoke-test 命令路径，并说明预期运行时间和数据量限制。

## 7. 验证与回归

- [ ] 7.1 运行 `fu` 样例数据的商品期货单元测试和 smoke test。
- [ ] 7.2 运行加密货币预处理/环境 smoke test，确认现有 depth=25 行为仍可用。
- [ ] 7.3 运行 `openspec validate add-commodity-futures-support --strict`，并记录因缺少完整原始数据或 GPU 而跳过的昂贵步骤。
