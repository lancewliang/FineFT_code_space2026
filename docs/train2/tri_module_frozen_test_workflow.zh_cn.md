# 三模块门控：两折验证、生产冻结与一次性 Test

本文说明 `tri_module` 的最终训练和评价流程。它与 `legacy` 是两套彼此独立的门控实现；默认值仍是 `legacy`，本流程不会自动切换默认门控，也不会删除旧门控。

## 先分清三种“次数”

只有 **2 个外层评价 Fold**：

1. 初始历史：2024-07-23 至 2024-10-31。这里只提供历史，不产生外层成绩。
2. 外层 Fold 1：用此前历史训练，评价 2024-11-01 至 2025-02-28。
3. 外层 Fold 2：把 Fold 1 时段加入历史后重新训练，评价 2025-03-01 至 2025-06-27。

每个外层 Fold 内部可以有多次 inner prequential 拟合。它们的用途是生成未见过的 OOF 分数，用于选择 λ、拟合 Isotonic 和 Platt；这些拟合次数不是新的外层 Fold。

两个外层 Fold 完成后，还会用全部 valid 数据拟合最终 Selector 和四头 Utility Model。这个“全 valid 生产拟合”也不是第三个外层 Fold。最终校准器只使用两次外层积累的未见分数，不能使用全 valid 模型对自己训练数据的预测。

所有 Fold 边界都按官方 `TradingDay` 判断，执行 `embargo >= H = 3`，并且一个持仓 episode 不得跨边界拆分。初始历史之后的每条合格 valid 样本最多进入一次外层未见评价。

## 训练与冻结

Utility Model 有独立训练入口：

```bash
export UTILITY_EPISODES_CSV=/absolute/path/candidate_episodes.csv
export UTILITY_OUTPUT_DIR=/absolute/path/utility_run
export UTILITY_FEATURE_NAMES=feature_a,feature_b
export UTILITY_CODE_VERSION=<immutable-git-revision>
bash FineFT/script/train/DiHFT/trading_utility_fu_10.sh
```

训练报告必须显示：`outer_fold_count = 2`、`final_calibration_source = outer_unseen_predictions`、`test_accessed = false`。

调用 `freeze_production_assets(...)` 前，需要准备：

- 最终全 valid Selector manifest 及其 lineage；
- Utility runtime manifest（四个模型头、Isotonic/Platt、eligibility 和残差证据）；
- 固定成本模型 manifest；
- VAE 权重、train scaler、train-ID likelihood reference；
- 与上述资产一致的不可变代码版本。

冻结时会核对两折协议指纹、valid 范围、全 valid Selector 身份、特征 schema、λ 网格、成本口径、OOD 阈值和所有文件哈希。不一致时直接失败，不会生成冻结目录。冻结成功后产出 `frozen_production_manifest.json`，其中保留 `legacy_retained = true` 和 `default_gating_stack = legacy`。

## 唯一一次正式 Test

正式 test 范围固定为 2025-06-30 至 2026-02-05。调用 `run_one_shot_test_comparison(...)` 时：

- legacy evaluator 只收到 test 数据，不会收到或加载 Utility Model；
- tri-module evaluator 收到同一份 test 数据和已冻结 manifest；
- 两者必须使用相同的成本版本和审计 schema `1.1.0`；
- legacy 的 utility 专属字段必须为 `null`，不能为了填字段而运行效用推理；
- 冻结资产在 test 前后重新校验哈希；
- 结果只发布到冻结目录的 `official_test/`，同一冻结身份已有结果时拒绝覆盖或重跑；
- evaluator 或校验失败时，临时目录会清理，不留下看似完成的正式报告。

对照报告分别保留 performance、risk、coverage、OOD、route、utility rejection、channel support 和 audit integrity。报告是供人判断的证据，不承诺消灭亏损交易，也不会根据结果修改 Selector、模型、校准器、eligibility、λ、阈值或协议。

## 端到端审计与故障语义

正式报告保存两条独立的真实决策链。每条链都使用相同公共 envelope，并能由 `freeze_id`、`prediction_id`、`decision_id`、`trade_id` 和 account snapshot ID 追溯：

```text
gate stack + frozen assets
  -> route slot
  -> candidate / NullCandidate
  -> utility admission（仅 tri_module）
  -> final action
  -> real Trade
  -> Account snapshot
```

以下故障必须 fail-closed：缺少任一模型 head、文件 hash 损坏、feature schema 不符、eligibility 协议过期、Fold 指纹不符。tri-module 不得在失败时回退到 legacy。

审计语义保持明确：

- OOD 或执行数据缺失且无法成交时，不生成虚假成交；
- Utility rejection 是决策层拒绝，不标成 exchange rejection；
- shadow/counterfactual fill 只用于离线标签，禁止进入真实 Trade/Account ledger；
- 开仓、同向加仓、反手、NullCandidate、ineligible channel、`U_lower <= 0`、硬止损和断路器应分别由端到端测试覆盖。

## 什么变化会让旧外层证据失效

冻结协议中的任一项变化，都产生新的协议指纹，旧的两折结果不能继续使用：特征或 schema、模型族或搜索空间、λ 网格、分位规则、eligibility 证据门槛、显著性/单调性规则、OOD 进入或恢复规则、Fold 日期、H 或 embargo。发生这些变化时，必须从外层 Fold 1 重新开始；不能只重跑 Fold 2，也不能复用原正式 test 身份。

正式 test 的优劣不会触发第二次 test。是否删除 legacy、是否更改默认门控，仍需用户今后另行明确决定。
