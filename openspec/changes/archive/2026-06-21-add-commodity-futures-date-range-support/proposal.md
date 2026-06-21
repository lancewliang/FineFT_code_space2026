## Why

当前商品期货主流程只支持按单一年份拼接主力连续序列，无法直接覆盖 `2023-01-01` 到 `2026-02-28` 这类跨年日期范围。用户需要按真实训练窗口一次性生成连续主力原始文件和后续特征数据，不能手工按年份拆分再合并。

## What Changes

- 将商品期货主流程从“单年输入”改为“日期范围输入”，由 `START_DATE` / `END_DATE` 自动推导需要扫描的年份集合。
- 让主力合约拼接跨年连续工作，不在年边界重置前一交易日主力选择状态。
- 将 `main.sh` 的输出文件名和日志名改为日期范围语义，避免单年命名误导。
- 保持现有左闭右开日期语义不变；要覆盖 `2026-02-28`，仍然需要传 `END_DATE=2026-03-01`。
- **BREAKING** 对旧的单年手工调用方式：`YEAR=...` 不再是必须输入，且单年输出文件名不再作为主入口假设。

## Capabilities

### New Capabilities
- 无。本次变更只扩展既有 `commodity-futures-support` 能力。

### Modified Capabilities
- `commodity-futures-support`: 更新商品期货主流程脚本与主力拼接行为，使其支持跨年日期范围而不是单年。

## Impact

- 影响 `data_preprocess/script_preprocess/future_upgraded/commodity/main.sh` 与 `fu_full_process.sh` 的参数语义和输出命名。
- 影响 `data_preprocess/operator_futures/commodity/main_contract.py` 与 `stitch_main_contract.py` 的年份扫描和主力选择状态传递。
- 影响商品期货端到端 smoke / CLI 测试，需要覆盖跨年样例。
- 不改变商品期货的盘口深度、手续费、funding 关闭或特征契约，只调整输入时间范围的组织方式。
