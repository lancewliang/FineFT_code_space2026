# Close Issues: add-commodity-futures-support

## 2026-06-21 close 前置检查未通过

### CRITICAL: `tasks.md` 仍有未完成条目

`/sddflow close` 前置条件要求 `openspec/changes/add-commodity-futures-support/tasks.md` 所有任务条目均为 `[x]`，但当前仍有 10 个实际未勾选条目：

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
- `tasks.md` 中仍存在上述未勾选条目，因此不能进入归档。

### 处理建议

先进入 `/sddflow build`，基于已有实现与测试证据逐项核对这些条目；如果代码已经覆盖，则同步勾选 `tasks.md` 并运行验证；如果存在缺口，则补实现或补测试后再回到 `/sddflow close`。
