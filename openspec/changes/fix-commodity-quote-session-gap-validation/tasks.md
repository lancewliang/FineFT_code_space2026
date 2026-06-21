## 1. 商品交易 session 配置

- [x] 1.0 商品交易 session 配置完成（与 `plan-ready.md` Task 1 和 superpowers plan Task 1 同步） <!-- 已实现: CommodityConfig 增加 TradingSession 配置和校验，fu 配置常规交易时段 -->
- [x] 1.1 修改 `data_preprocess/operator_futures/commodity/config.py`，为商品配置增加交易 session 字段，并为 `fu` 配置燃料油常规交易 session。
- [x] 1.2 增加或调整商品配置测试，断言 `fu` 配置包含 quote gap 校验可用的交易 session，并覆盖 session 结构校验。
- [x] 1.3 验证：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_config_schema.py -q`。

## 2. Quote gap session-aware 校验

- [x] 2.0 Quote gap session-aware 校验完成（与 `plan-ready.md` Task 2 和 superpowers plan Task 2 同步） <!-- 已实现: downscale_quote_features 按商品交易 session 校验 gap，跨 session 不误报，同 session 缺口仍 fail-fast -->
- [x] 2.1 修改 `data_preprocess/operator_futures/commodity/downscale.py`，让 `downscale_quote_features()` 根据商品交易 session 判断相邻 quote bar 是否需要连续。
- [x] 2.2 保留同一有效交易 session 内缺少目标频率 quote window 的 fail-fast 行为。
- [x] 2.3 增加或调整商品下采样测试，覆盖夜盘结束后到下一有效交易 session 之间不因 `2025-10-31 23:05:00` 报错。
- [x] 2.4 增加或调整商品下采样测试，覆盖同一交易 session 内中间 quote window 缺失仍报 `Target window has no quote snapshots`。
- [x] 2.5 验证：运行 `conda run -n finetf pytest data_preprocess/tests/test_commodity_downscale.py -q`。

## 3. 规格与回归验证

- [x] 3.0 规格与回归验证完成（与 `plan-ready.md` Task 3 和 superpowers plan Task 3 同步） <!-- 已实现: OpenSpec strict、配置/downscale 回归测试和 diff 检查通过 -->
- [x] 3.1 运行 OpenSpec strict 校验，确认 session-aware quote gap 规格合法。
- [x] 3.2 运行商品期货配置和 downscale 相关回归测试，确认现有 best quote、base feature、orderbook 和 empty-input 行为不回退。
- [x] 3.3 验证：运行 `openspec validate fix-commodity-quote-session-gap-validation --strict`、`conda run -n finetf pytest data_preprocess/tests/test_commodity_config_schema.py data_preprocess/tests/test_commodity_downscale.py -q`、`git diff --check`。
