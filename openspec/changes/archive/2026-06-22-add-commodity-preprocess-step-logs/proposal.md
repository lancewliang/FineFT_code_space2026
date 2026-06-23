## Why

商品 preprocess 主流程当前只有总入口日志，后续主要步骤缺少独立日志文件。运行耗时较长时，失败或卡住后只能从总日志里排查，定位成本高。

目标是在商品 preprocess 的主要阶段为每个步骤写入独立日志文件，保留现有主流程调用方式和输出产物路径，提升长流程排障效率。

## What Changes

- 为商品 preprocess 主流程的 9 个主要步骤添加独立日志文件。
- 在总日志中打印每个阶段的开始、成功、失败和对应日志路径。
- 保持并发子任务已有日志目录可用，不移除现有 cross_section/merge 按日期日志。
- 保留现有 `main.sh` 调用方式、默认日期区间、产物目录和总日志行为。
- 阶段日志捕获 stdout 和 stderr；步骤失败时保留原有 `set -euo pipefail` 的失败语义，让主流程停止。

## Capabilities

### New Capabilities

- 无

### Modified Capabilities

- `commodity-futures-support`: 增加商品 preprocess 主流程步骤级独立日志契约。

## Impact

- 影响脚本：`data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
- 影响入口：`data_preprocess/script_preprocess/future_upgraded/commodity/main.sh` 的总日志仍保留
- 影响测试：商品脚本/文档测试需要覆盖步骤日志文件名、阶段输出和失败语义
- 不新增外部依赖，不改变 Python 算法逻辑、数据产物格式或 Binance/crypto futures 脚本

## 用户场景

- 维护者启动 `data_preprocess/script_preprocess/future_upgraded/commodity/main.sh` 后，能在日志目录中按阶段查看每个主要步骤的输出。
- 某个阶段失败时，维护者能直接打开对应步骤日志定位报错，而不需要在总日志中搜索整条流水线。
- 10 分钟以上的主流程运行中，维护者可以观察各步骤日志判断流程卡在哪个阶段。

## 设计方向

采用“步骤级独立日志文件”方案：在 `fu_full_process.sh` 的商品主流程阶段边界处统一包装主要步骤执行，每个主要步骤写入自己的日志文件。总入口日志继续保留，用于记录整体启动、阶段调度和高层状态。

每个主要步骤日志覆盖以下阶段：`stitch_main_contract`、`downscale_continuous_by_trading_day`、`cross_section`、`merge`、`concat`、`time_feature`、`merge_clean`、`ic_correlation`、`scale_save`。

## 关键决策

- 每个主要步骤写入独立日志文件，而不是只在总日志中打印开始/结束。
- 保留现有 `main.sh` 调用方式、默认日期区间、产物目录和总日志行为。
- 日志文件名包含 symbol、target_freq、start_date、end_date 和步骤名，避免不同运行互相覆盖不清。
- 阶段日志应捕获 stdout 和 stderr；步骤失败时保留原有 `set -euo pipefail` 的失败语义，让主流程停止。

## 范围边界

**包含：**
- 为商品 preprocess 主流程的 9 个主要步骤添加独立日志文件。
- 在总日志中打印每个阶段的开始、成功、失败和对应日志路径。
- 保持并发子任务已有日志目录可用，不移除现有 cross_section/merge 按日期日志。

**不包含（本次）：**
- 不改 Python 算法逻辑或数据产物格式。
- 不引入新的日志框架或外部依赖。
- 不改变 Binance/crypto futures 的预处理脚本。
- 不做日志轮转、压缩或长期清理策略。

## 验收标准

- [ ] 运行商品 `main.sh` 后，每个主要步骤都有独立日志文件。
- [ ] 总日志中能看到每个主要步骤的开始、成功/失败和步骤日志路径。
- [ ] 某个步骤失败时，主流程仍然以非 0 状态退出，并且对应步骤日志保留错误输出。
- [ ] 现有商品 preprocess 输出产物路径不变。
