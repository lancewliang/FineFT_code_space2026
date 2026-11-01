# 07: 贯通辅助列与切片坐标

**What to build:** 把 02 的行情辅助列和契约从 SCALE_SAVE 原样传递到 FineFT 合约数据、训练切片、标注切片及后续共享行情输入，保证运行期无需重算窗口。

**Blocked by:** 02 — 在分区行情中预计算因果价格统计量

**Status:** ready-for-agent

**Implementation:** 未实施。此前代码改动已全部回退；本轮只重切任务，不启动代码修改。ready-for-agent 表示规格可实施，不表示依赖已完成。

## 文件范围

FineFT/datahandler/commodity_contract_dataset.py、实际会重写行情的标注/校准入口及数据准备脚本；必要的 manifest 传播。

## 验收标准

- [ ] 按 01 契约验证并保留辅助列及产物信息；合约整表、train/valid/test、slice 和标注重写路径均不丢列，不进入三流市场特征清单。
- [ ] 保留原始行身份及 slice 到合约分区行的对应关系；DataFrame reset_index 不得被误当成原行情坐标。与老师表、df_index/step_index 和 regime 标注保持对齐。
- [ ] 先计算再切片；非分区首部 slice 的首行沿用原有效位，不重复丢弃前 18 行，也不新增历史前缀。分区首部无效行只在消费时禁止交易。
- [ ] 为运行入口提供有效起点、有效行数和原始坐标映射；不存在有效交易数据或不足一次环境转移的片段须明确拒绝，不静默跳过。
- [ ] 旧数据缺列、缺版本或元信息不一致明确失败并提示重新生成数据，不在训练入口重算或伪造有效位；说明需重跑的数据阶段及产物，保留原 DP 算法。
- [ ] 用真实 Feather 写读和切片/标注路径验证多合约、多分区、首 slice、后续 slice、尾部片段的数值和坐标；不要求正式历史全量重建或重训。

## 职责边界

负责数据保真和可定位的有效性信息，不实现环境起点选择、CLI/worker 配置传递、退出事件或 learner。worker 实际消费由 04 验收。

共同遵循 [spec](../spec.md) 第 4–6 节与 [实施依赖总览](../implementation-plan.md)。Python/pytest 使用 `conda activate finetf`。不更改已确认范围，不启动正式重训。

## Comments

- 2026-11-01：按确认后的行情统计预计算、Q8 A 与 exit_price_scale_floor 命名重新切分职责；当前验收项全部未完成。
