# Close Issues: add-commodity-futures-support

## 2026-06-21 close 前置检查未通过

### RESOLVED: `tasks.md` 仍有未完成条目

`/sddflow close` 前置条件要求 `openspec/changes/add-commodity-futures-support/tasks.md` 所有任务条目均为 `[x]`。本问题在后续 `/sddflow build` 中已处理：逐条核对实现证据，补齐 `ic_correlation.py` 商品期货 manifest 逻辑，并同步勾选全部任务。

原始阻塞项为以下 10 个未勾选条目：

- `1.2` 新增共享 schema 工具。
- `1.3` 新增配置与 schema 单元测试。
- `2.2` 保留主力拼接元数据并排除 state features。
- `2.3` 新增主力拼接测试覆盖。
- `3.2` 商品期货真实 5 档 orderbook 下采样。
- `3.3` 商品期货 base feature 下采样。
- `3.4` 商品期货 quote feature 下采样。
- `3.5` 使用 `fu2302.csv` 覆盖下采样测试和错误路径。
- `4.2` 更新 merge/concat/time feature/feature selection/scale-save 的 manifest 逻辑。
- `4.3` 新增商品期货特征管线契约测试。

### 当前证据

- `plan-ready.md` 中 Task 1-8 的任务级 checkbox 已全部为 `[x]`。
- `docs/superpowers/plans/2026-06-20-add-commodity-futures-support.md` 中实际 step checkbox 已全部为 `[x]`。
- `tasks.md` 当前已无未勾选任务条目。
- 已补充 `data_preprocess/operator_futures/feature_selection/ic_correlation.py` 的商品期货 reward/state manifest 划分，并在 `fu_full_process.sh` 中传入 `--market_type commodity_futures --orderbook_depth 5`。
- 已运行商品期货数据测试、商品环境测试、脚本语法检查、`finetf` 编译检查和 OpenSpec 严格校验。

### 处理建议

已处理。可以重新进入 `/sddflow close` 执行归档前验证。
