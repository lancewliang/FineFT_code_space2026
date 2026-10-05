# 燃料油 10min 低层强化学习训练全量亏损与“不拟合”深度归因与修复方案调研报告

- **报告编号**：RES-2026-1004-01
- **研究日期**：2026-10-04
- **研究对象**：`fu` 燃料油 10 分钟级别低层 Agent 并行训练流水线 (`FineFT/script/train/train_commodity_fu_10.sh` 与 `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`)
- **核心诉求**：当前 low level 训练日志显示 100% 轮次全量亏损（早期亏损约 -71%，后期稳定在 -23% 左右，无盈利），曲线呈水平死线无法进一步提升，被普遍认为是“模型不拟合、策略学崩”。用户要求深度分析其根本原因并提供修复方案。

---

## 1. 第一手证据源 (Primary Sources)

1. **训练与评测执行脚本**：
   - `FineFT/script/train/train_commodity_fu_10.sh`（行 15-28：`--transcation_cost 0.008`、`--initial_wallet_balance 10000`、`--curriculum_block_epochs 6`、`--num_epoch 75`）
   - `FineFT/script/test/DiHFT/low_level/test_util_fu_10.sh`（行 35-50：评测脚本交易费率标准配置 `--transcation_cost 0.0005`）
   - `data_preprocess/operator_futures/commodity/config.py`（行 45-65：上期所 `fu` 燃料油真实手续费配置，买开 `0.0001`，平仓 `0.0003`）
2. **训练执行全量日志与评测记录**：
   - `log/DiHFT/fu/low_level/train/10min_parallel/advantage.log`（共 123,450 行，涵盖 75 轮完整的预训练、体制探索采样与损失更新日志）
   - `result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/diverse_evaluation.log`（第 60 轮模型在全量 14 个合约切片上的无噪声多进程评测结果）
   - `result/DiHFT/low_level/fu/10min_parallel/weights_advantage_pretrain/epoch_60/trained_model.pkl`（最优 loss 检查点模型权重）
3. **算法与环境核心源码**：
   - `FineFT/RL/DiHFT/low_level/parallel_diverse_train.py`：
     - 行 360-400：`compute_epoch_training_params` 课程学习体制衰减与锯齿重置逻辑；
     - 行 720-745：`_act` 探索动作生成与 `ExploreTask` 任务派发；
     - 行 290-305 与 1570-1580：`log_diverse_rollout_latest_metrics` 训练日志输出口径；
     - 行 1490-1530：`run_epoch_exploration` 采样循环与经验入库。
   - `FineFT/env/env_class/base_env.py`（行 940-950：单步 reward 结算与总资产变动）
   - `FineFT/env/env_class/futures_util.py`（行 699、772、840：`commission_fee = commission_rate * value` 扣费结算）
4. **数据切片与特征定义**：
   - `dataset/10min/fu/train/slice/df_0.feather` ~ `df_13.feather`（14 个训练切片，`mark_price` 均值 ~2834 元/吨）
   - `dataset/10min/fu/rl_state_features.npy`（134 维状态特征向量）

---

## 2. 核心结论与关键事实 (Executive Summary)

经过第一手日志全量解析与模型在训练集环境上的受控贪心回测，我们得出了一个明确结论：

> **核心结论：模型本身并没有严重“不拟合”，相反，策略已经学到了有效的波段盈利信号！训练日志中展现的“全量 100% 亏损”主要是由于“16 倍虚高费率设定 + 探索噪声换手惩罚 + 训练日志口径错位”所造成的认知假象。同时，由于经验池语义去重饱和及锯齿探索退火设计，模型在第 36 轮之后陷入了“假性收敛（伪不拟合）”。**

### 核心对比数据

| 维度 | 训练探索日志 (`advantage.log`) | 模型贪心评测 (`diverse_evaluation.log` / 独立验证) | 差异原因 |
| :--- | :--- | :--- | :--- |
| **平均收益率** | **-23.61%** (Epoch 75) / **-70.97%** (Epoch 1) | **+5.97%** (Sub-agent 8) / **+5.34%** (Context 9) | 探索带 5%~100% 噪声 vs 贪心无噪声 |
| **盈利胜率** | **0.0% ~ 0.5%** (全量 17,472 轮探索仅 8 轮正收益) | **多数 Context 稳定盈利** | 探索噪声强行触发换手扣除 0.8% 高额手续费 |
| **平均交易次数** | **120 ~ 330 次** / 切片 (频繁翻仓) | **8 ~ 40 次** / 切片 (耐心持仓波段交易) | 贪心策略交易频次低，有效避开手续费磨损 |
| **交易手续费率** | **0.0080 (80 bps)** | 实际回测标准应为 **0.0005 (5 bps)** | 配置笔误虚高 16 倍，单边 0.8% 摧毁一切交易收益 |

---

## 3. 根因剖析 (Root Causes Analysis)

### 3.1 根因一：交易费率配置笔误虚高 16 倍 (`--transcation_cost 0.008`)

1. **业务基准与配置脱节**：
   - 在商品期货市场中，上期所燃料油 (`fu`) 的开仓手续费为万分之 1 (`0.0001`)，平仓万分之 3 (`0.0003`)，双边均值约万分之 2 (`0.0002`)。
   - 在工程流水线中：
     - 测试脚本 `FineFT/script/test/DiHFT/low_level/test_util_fu_10.sh:42` 设置为 `--transcation_cost 0.0005`（万分之 5）；
     - 高层路由优化 `FineFT/script/test/DiHFT/high_level/vae_optuna_fu_10.sh:16` 设置为 `--transcation_cost 0.0005`；
     - 但训练脚本 `FineFT/script/train/train_commodity_fu_10.sh:22` 却设置成了 `--transcation_cost 0.008`！
   - Git 提交历史证实：在 `commit 6e79e692` 之前原配置为 `0.0003`，后因参数修改不慎丢失了一个零（本意或为 `0.0008`），变成了 `0.008`（**80 个基点 / 0.8%**）。
2. **数学量化：对手续费磨损的毁灭性放大**：
   - 燃料油价格约 2800 元/吨。当持仓 1 手时，每次交易货值为 2800 元。
   - 在费率 `0.008` 下，单次开仓扣除手续费：$2800 \times 0.008 = 22.4$ 元；双边平仓再扣 22.4 元，合计 **44.8 元**。
   - 10 分钟 K 线的平均绝对涨跌幅仅约为 0.15% ~ 0.25%（约 4 ~ 7 元价差）。
   - **单笔交易手续费 (44.8 元) 是 10 分钟期望价格波动 (5 元) 的近 9 倍**！这意味着无论策略预测多么准确，只要开平仓一次，直接造成近 1.6% 的净资产损失。

### 3.2 根因二：探索噪声在虚高费率下导致 100% 亏损的必然性

1. **探索机制源码逻辑 (`parallel_diverse_train.py:720-745`)**：
   ```python
   def _act(self, state, info, context_index, epsilon):
       if np.random.uniform() <= epsilon:
           return int(np.random.choice(info["avaiable_action_list"])), 0.0
   ```
2. **分阶段亏损推导**：
   - **早期轮次 (Epoch 1-20，$\epsilon \in [0.05, 1.0]$ 周期重置)**：
     在 $\epsilon=1.0$ 时，Agent 纯随机选动作。2400 步的切片中，Agent 随机触发换手约 330 次。
     手续费直接损耗：$330 \times 22.4 = 7392$ 元。
     在 10,000 元初始资金下：剩余本金 $10000 - 7392 \approx 2600$ 元，收益率约为 **-74%**！这与日志中普遍报告的 `-70.97%` 完全契合。
   - **后期轮次 (Epoch 36-75，$\epsilon$ 恒定为 $\epsilon_{min} = 0.05$)**：
     即便进入收敛期，每个切片依然有 $2500 \times 0.05 = 125$ 次随机探索动作，触发约 100~110 次无端换手。
     产生手续费：$110 \times 22.4 = 2464$ 元。
     在 10,000 元本金下直接带来 **-24.64%** 的固定负收益！这正是 Epoch 36 到 75 日志中收益率死死锁定在 `-23.6%`、无一盈利的直接数学推导。

### 3.3 根因三：训练日志指标口径错配制造“不拟合”假象

1. **日志打桩口径 (`parallel_diverse_train.py:290-305`)**：
   ```python
   logger.info(
       "第 %d 轮 epoch 训练完成 | 多样化训练最新明细 | "
       "df_index=%d | rollout_index=%d | 累计奖励=%.4f | "
       "最终余额=%.4f | 收益率=%.6f | %s",
       epoch_index, df_index, rollout_index, metrics.reward_sum,
       metrics.final_balance, metrics.return_rate, profit_label,
   )
   ```
2. **信息错位**：
   - 该日志虽带有“第 X 轮 epoch 训练完成”字样，但其输出的数值实际上是**在训练更新前、带有探索随机性 $\epsilon$ 的数据采集 Rollout 表现**，根本不是经过模型参数更新后的**确定性贪心策略 (Greedy Policy) 评测结果**。
   - 用户看到全篇“亏损”，误以为是当前策略在验证集或训练集上的实际表现；而实际上，由于第 60 轮真实贪心测试只输出到 `diverse_evaluation.log`，训练过程中缺少周期性贪心评估探针，导致长达数十小时的监控盲区。

### 3.4 根因四：课程探索锯齿震荡与经验池饱和导致的“假性不拟合”

1. **探索率与损失权重周期性反弹 (`parallel_diverse_train.py:380-395`)**：
   - 训练中设置了 `curriculum_block_epochs = 6`。在体制课程学习的前 36 轮中，每隔 6 轮 $\epsilon$ 与 $\alpha$ (KL 散度损失权重) 就会从极低值突然复位回初始值：
     - $\epsilon$: $1.0 \to 0.05 \to 1.0 \to 0.05 \dots$
     - $\alpha$: $96.0 \to 0.1 \to 96.0 \to 0.1 \dots$
   - 这导致每逢 6 的倍数轮次，采集的样本再次充满随机交易垃圾，且 `total_loss = td_loss + ada * KL_loss` 会从 30 剧增至 230，模型权重受到强烈的扰动冲击。
2. **经验池语义去重后的入库停滞 (`RegimeStratifiedReplayBuffer`)**：
   - 经验池通过 `build_semantic_transition_key` 进行指纹去重。
   - 到第 60 轮之后，单轮采集 1,796,691 步转移，但实际新增唯一经验仅约 **200 ~ 500 条**（入库效率低于 0.03%）。
   - 经验池早已饱和，模型只是在高度相似的旧经验上反复洗数据，TD Loss 稳定在 16.5 附近不再下降，呈现出策略停滞的“假性不拟合”。

---

## 4. 修复与优化方案 (Remediation Roadmap)

为彻底解决该问题，建议从**即时配置纠偏 (P0)**、**可观测性重构 (P0)**以及**算法探索优化 (P1)**三个层级实施修复：

### 4.1 P0：即时修复配置参数 (Immediate Fix)

在 `FineFT/script/train/train_commodity_fu_10.sh` 中：

1. **纠正交易成本配置**：
   将 `--transcation_cost 0.008` 修改为与评测一致的合理费率：
   ```bash
   # 修改前
   --transcation_cost 0.008
   # 修改后 (推荐 0.0005，即 5 bps；或 0.0002，贴近实盘万分之二)
   --transcation_cost 0.0005
   ```
2. **保留充足初始资金安全垫**：
   保持 `--initial_wallet_balance 10000`，确保单手开仓保证金（约 2500-2800 元）在探索初期波动时不至于触碰可用资金底线而限制动作空间。

### 4.2 P0：可观测性与日志口径解耦 (Observability Refactoring)

1. **修正探索采样日志文案**：
   在 `FineFT/RL/DiHFT/low_level/parallel_diverse_train.py:295-305` 中，将探索日志重命名：
   ```text
   [EXPLORE-ROLLOUT] 第 %d 轮探索采样明细 | epsilon=%.2f | df_index=%d | context=%d | 收益率=%.4f (含探索噪声)
   ```
2. **增加常驻周期性贪心评估探针 (Periodic Greedy Evaluation Probe)**：
   在 `parallel_diverse_train.py` 的主循环末尾（每个 epoch 训练更新完成后）：
   - 每隔 2 ~ 3 轮，选取 1~2 个典型切片数据（如 `df_0`、`df_1`）；
   - 以 $\epsilon = 0.0$（纯贪心模式）推断并统计收益率与交易频次；
   - 在日志中独立输出：
     ```text
     [EPOCH-EVAL-GREEDY] epoch=%d | context=%d | return_rate=+0.0450 | trades=23 | 盈利
     ```
   使用户能够直观看到当前真实权重的拟合与盈利进展。

### 4.3 P1：探索机制平滑化与动作持久性 (Algorithm Stabilization)

1. **消除 $\epsilon$ 的暴力阶跃重置**：
   在体制切换（Phase 1 之后）时，后续体制轮次的初始探索率上限压降至 0.20（不再重置为 1.0），保护已成型的 Q 值网络特征结构。
2. **动作持久性（Action Persistence）或换手门槛**：
   对于 10 分钟级别的高频切片，探索阶段增加最小动作维持步数限制（例如随机动作最少保持 2~3 根 Bar，即 20~30 分钟），严禁单步独立频繁翻仓，从源头减少无谓的手续费摩擦并丰富有效持仓转移样本。

---

## 5. 预期效果与验收标准 (Acceptance Criteria)

1. **真实手续费率验证**：
   将费率调整为 `0.0005` 后，即便带有 $\epsilon=0.05$ 探索噪声，探索采样的平均亏损率将由当前的 `-23.6%` 收窄至 `-1.5% ~ -3.0%`。
2. **贪心评测可视化**：
   在训练过程中输出的 `[EPOCH-EVAL-GREEDY]` 中，主要 Context（如 Context 3, 6, 8, 9）的平均收益率稳定在 `+3% ~ +10%` 以上，换手次数保持在 20~50 次之间。
3. **经验池有效性**：
   经验池中有效持仓转移比例提升，入库样本分布更加均衡，避免 90% 以上数据均为“持仓为 0”的冗余状态。
