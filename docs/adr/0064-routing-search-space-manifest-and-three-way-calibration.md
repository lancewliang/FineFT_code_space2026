---
status: accepted
date: 2026-10-26
---

# 0064. 路由搜索空间清单与三分区校准契约

步骤 7 的上游只能稳定产出 `train/valid/test` 和既有 Selection Manifest，无法产出独立的四段时间证据、artifact 哈希清单或 scoring protocol。决定将 `EXPERIMENT_MANIFEST` 收缩为 schema-3 的 Routing Search-Space Manifest，只保存 `strategy_version` 与九维 `search_space`；运行时继续依据 `BASE_PATH`、`DATASET_NAME`、`EXPERIMENT_NAME`、Selection Manifest 以及既有 `train/valid/test` 目录解析数据和模型。`valid` 同时承担低层选择与高层路由校准，`test` 只消费冻结配置，不再对步骤 7 宣称独立 calibration。

artifact、VAE 评分架构、文件集合和固定风控参数不再要求上游写入输入 Manifest。程序仍在运行时计算输入 identity 和内部 raw-score cache identity，以防止冻结配置或量化缓存静默复用于不同的实际输入；这属于运行时一致性检查，不是上游生成证据文件的职责。固定的 MDD、非主力合约防御、动作持续、止损、冷却和断路器参数继续通过 CLI 传入，并随最佳 Trial 写入 frozen config，但不属于 Optuna search space。

## Consequences

- Manifest 可以由上游直接生成，不再依赖上游拥有模型文件哈希、TradingDay 边界或评分批处理协议。
- 当前三分区结果必须按“valid 上的联合选择与校准、test 上的冻结评估”解释；若未来需要独立 calibration，必须新增数据分区或另行建立数据契约。
- 内部 identity 和量化缓存检查仍保留，因此删减输入 Manifest 不等于放弃运行时的输入一致性保护。
