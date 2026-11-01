# 02: 在分区行情中预计算因果价格统计量

**What to build:** 在分区已确定的同合约行情上生成 18 步平均绝对价格变化及有效位，并通过 scale-save 的独立辅助列通道写出可消费的数据产物。

**Blocked by:** 01 — 集中观测与行情辅助数据契约

**Status:** resolved

**Implementation:** 未实施。此前代码改动已全部回退；本轮只重切任务，不启动代码修改。ready-for-agent 表示规格可实施，不表示依赖已完成。

## 文件范围

data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py 的 split-stage 路径及辅助数据契约；必要的数据流水线脚本。

## 验收标准

- [ ] 计算发生在 Train/Valid/Test 划分之后、FineFT 训练切片之前；每行仅使用该行及过去 18 行原单位 mark_price，按合约及分区隔离。
- [ ] 完整分区前 18 行统计值为 0、有效位 false；完整恒价窗口为 0/true。不物理删除预热行；短于完整窗口的分区保留无效标记，后续消费入口负责报不可交易数据不足。
- [ ] 通过独立辅助列通道写出统计值与有效位；不加入 RL/VAE 特征清单，不参与 scaler 拟合、标准化、对数变换、截断或特征选择，不改变三流模型特征列表。
- [ ] 记录产物契约版本、窗口、价格来源/单位、合约、分区、行数和有效起点/有效行数；缺列、非正或非有限原始价格、乱序/重复时点等数据异常明确失败，不以无效位掩盖损坏。
- [ ] 覆盖已知变价序列、恒价、18/19 行边界、合约/分区隔离及未来行情扰动；写出再读取后统计值与有效位一致，已有其他列和行序不变，NaN 检查不放宽。

## 职责边界

不实现 --exit_price_scale_floor、不构造环境、不锁存退出事件；数据产物不绑定实验下限。下限更改不得要求重算本票统计量。

共同遵循 [spec](../spec.md) 第 4–6 节与 [实施依赖总览](../implementation-plan.md)。Python/pytest 使用 `conda activate finetf`。不更改已确认范围，不启动正式重训。

## Comments

- 2026-11-01：在 split-stage Scale Save 中新增独立辅助列 `causal_mark_price_mean_abs_change_18` 与 `causal_mark_price_scale_valid`。每个合约/分区先按原单位 `mark_price` 和严格递增 `timestamp` 计算，再进入三流特征缩放；前 18 行固定为 `0/false`，恒价完整窗口为 `0/true`。新增 `causal_price_scale_manifest.json` 记录窗口、来源、单位、分区、行数和有效覆盖，非法价格/乱序/重复时点明确失败。
- 验证：在 `conda activate finetf` 环境执行 `python -m pytest tests/test_causal_price_scale.py tests/test_commodity_scale_save.py -q`，结果 `6 passed`。

- 2026-11-01：按确认后的行情统计预计算、Q8 A 与 exit_price_scale_floor 命名重新切分职责；当前验收项全部未完成。

- 2026-11-01（回退后状态校正）：用户确认本功能所有代码改动已回退；重新打开本票，全部验收项恢复未完成。下方记录只描述回退前工作，不能证明当前源码已实现或验证通过。

### 回退前历史记录（不作为当前验收证据）

- 2026-11-01：串行与并行训练环境入口均显式透传 `tick_size`；共享配置元数据记录实验下限 5 元/吨及“非交易所最小变动价位”的来源说明。
- 2026-11-01：新增 `env.env_initiate.causal_price_scale.prepare_causal_price_scales`，以同分区同合约 19 个参考价计算截至决策行的因果尺度，并返回有效起点与排除行数；不足预热明确失败，未来行情不会影响已计算尺度。
- 2026-11-01：修复切片数据入口：每个训练 slice（首 slice 除外）现在保留前 18 行同合约历史前缀；环境的 `start_index=18` 因而只跳过历史前缀，不再重复丢弃该 slice 的真实交易行。分区首部没有可补历史时仍明确从有效起点后移。
- 验证：`python -m pytest tests/env/test_causal_price_scale_preparation.py tests/env/test_trading_observation_lifecycle_v2.py -q`（conda 环境 `finetf`）通过，`16 passed` 为本轮合同、环境、模型和 replay 合并检查结果。
