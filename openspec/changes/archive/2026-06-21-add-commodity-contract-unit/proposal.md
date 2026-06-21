# add-commodity-contract-unit

## 背景与目标

商品期货成交额 `Turnover` 是按合约交易单位计入的累计成交金额，现有商品期货预处理使用 `Turnover / Volume` 计算价格口径均价，会把燃料油 `fu` 等商品的均价放大一个合约交易单位倍数。目标是在商品配置中保存每个商品的合约交易单位，并将商品期货价格口径均价修正为 `Turnover / Volume / contract_unit`。

## 用户场景

- 用户运行燃料油 `fu` 商品期货预处理，希望由 `Volume` 和 `Turnover` 差分估算出的成交均价回到交易所报价价格口径。
- 用户后续新增其他商品期货品种时，可以在商品配置中声明不同的合约交易单位，而不是修改计算代码。
- 下游仍可读取原始成交额口径的 `tradeval`，同时使用修正后的 OHLC 和 `vwap` 价格特征。

## 设计方向

采用商品配置级 `contract_unit` 方案。`CommodityConfig` 新增必填正数 `contract_unit` 字段，`fu` 配置为 `10`。商品期货下采样计算通过现有 `symbol` 加载商品配置，不新增 CLI 参数，不改变输出列名或目录结构。

价格口径计算统一除以合约交易单位：秒级 `second_avg_price = second_tradeval / second_volume / contract_unit`；聚合后的 `vwap = tradeval / volume / contract_unit`。`second_tradeval` 和输出 `tradeval` 仍保存原始成交额差分，不做归一化，避免破坏下游对成交额字段的理解。

配置错误直接失败：`contract_unit` 缺失、为 `0` 或负数时抛出明确异常；未知 `symbol` 继续沿用现有商品配置错误。成交额差分校验保持原语义，`second_volume > 0` 但 `second_tradeval <= 0/null` 时继续报错并保留 timestamp、contract、`second_volume`、`second_tradeval` 信息。

## 关键决策

- `CommodityConfig` 新增 `contract_unit`，作为每个商品合约交易单位的唯一来源。
- `fu` 的 `contract_unit` 配置为 `10`。
- 仅修正价格口径字段，包含 `second_avg_price`、由其派生的 OHLC 价格和聚合 `vwap`。
- `tradeval` 保持原始成交额差分，不除以合约交易单位。
- 不新增 CLI 参数，外部接口继续通过 `symbol` 选择商品配置。
- `contract_unit` 必须为正数，配置错误不使用默认值兜底。

## 范围边界

**包含：**

- 商品期货配置中新增合约交易单位字段。
- 燃料油 `fu` 配置 `contract_unit=10`。
- 商品期货成交均价和 `vwap` 公式按合约交易单位修正。
- 商品期货相关测试覆盖配置、秒级均价、聚合 `vwap` 和原始 `tradeval` 保持不变。
- 修订后续规格中关于“不引入 contract_multiplier”的旧决策，改为引入商品配置级合约交易单位。

**不包含（本次）：**

- 不改变 Binance/crypto futures 的成交均价或 `vwap` 计算。
- 不把输出 `tradeval` 归一化为价格口径成交额。
- 不新增命令行参数或运行时配置文件。
- 不批量补齐其他商品品种的真实合约交易单位。
- 不修改商品期货主力合约选择规则、盘口校验、funding 行为或输出目录结构。

## 验收标准

- [ ] `CommodityConfig` 包含必填正数 `contract_unit` 字段。
- [ ] `fu` 商品配置的 `contract_unit` 为 `10`。
- [ ] 商品期货秒级成交估算使用 `second_tradeval / second_volume / contract_unit` 计算 `second_avg_price`。
- [ ] 商品期货聚合 `vwap` 使用 `tradeval / volume / contract_unit` 计算。
- [ ] 输出 `tradeval` 仍保持原始成交额差分，不除以合约交易单位。
- [ ] `contract_unit <= 0` 或缺失时抛出明确配置错误。
- [ ] 相关测试通过，并按项目约定使用 `conda activate finetf` 后运行 Python/pytest 验证命令。
