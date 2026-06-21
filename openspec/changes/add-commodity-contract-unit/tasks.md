## 1. 商品配置合约交易单位

- [x] 1.0 商品配置合约交易单位完成（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步） <!-- 已实现: CommodityConfig 增加正数 contract_unit 校验，fu 配置为 10，并覆盖配置测试 -->
- [ ] 1.1 修改 `data_preprocess/operator_futures/commodity/config.py`，为 `CommodityConfig` 新增必填正数 `contract_unit` 字段，并将 `fu` 配置为 `10`。
- [ ] 1.2 增加或调整商品配置测试，断言 `get_commodity_config("fu").contract_unit == 10`，并覆盖 `contract_unit <= 0` 的配置校验。
- [ ] 1.3 验证：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_config_schema.py -q`。回滚时移除新增字段和对应测试断言。

## 2. 商品成交价格口径修正

- [x] 2.0 商品成交价格口径修正完成（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步） <!-- 已实现: 商品成交均价与 vwap 除以 contract_unit，tradeval 保持原始成交额 -->
- [ ] 2.1 修改 `data_preprocess/operator_futures/commodity/downscale.py`、`downscale_single_day.py` 和 `downscale_continuous_by_trading_day.py`，让商品成交估算通过 `symbol` 读取 `contract_unit`，并用 `second_tradeval / second_volume / contract_unit` 计算 `second_avg_price`。
- [ ] 2.2 修改商品基础特征聚合逻辑，使 `vwap = tradeval / volume / contract_unit`，同时保持输出 `tradeval` 为原始成交额差分合计。
- [ ] 2.3 增加或调整商品下采样测试，覆盖 `fu` 的 `contract_unit=10` 时 `Turnover=26000`、`Volume=1` 产生价格口径 `2600`，并断言 `tradeval` 未被归一化。
- [ ] 2.4 验证：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_downscale.py -q`。回滚时恢复原公式和测试期望。

## 3. 规格与回归验证

- [ ] 3.0 规格与回归验证完成（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步）
- [ ] 3.1 如有相关文档或长期规格引用“不引入 contract_multiplier”，更新为商品配置级 `contract_unit` 的新决策，避免与实现语义冲突。
- [ ] 3.2 运行商品期货相关回归测试，确认配置、下采样和 CLI smoke 不受影响。
- [ ] 3.3 验证：运行 `openspec validate add-commodity-contract-unit --strict`、`conda run -n finetf pytest data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_downscale.py -q`、`git diff --check`。回滚时恢复规格和文档中的旧公式描述。
