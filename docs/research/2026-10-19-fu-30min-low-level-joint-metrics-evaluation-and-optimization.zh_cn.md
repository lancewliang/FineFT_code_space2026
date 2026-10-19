# fu 30min 低层二维联合评估产物 `two_dimensional_joint_metrics.csv` 质量评估与优化方案研究报告

- **研究日期**：2026-10-19（Asia/Shanghai）
- **研究对象**：
  1. `analysis_result/DiHFT/low_level/fu/30min_parallel/two_dimensional_selection/two_dimensional_joint_metrics.csv`
  2. `analysis_result/DiHFT/low_level/fu/30min_parallel/two_dimensional_selection/two_dimensional_selection.csv`
  3. `analysis_result/DiHFT/low_level/fu/30min_parallel/two_dimensional_selection/two_dimensional_candidate_rankings.csv`
  4. `analysis_result/DiHFT/low_level/fu/30min_parallel/two_dimensional_selection/two_dimensional_selection_manifest.json`
  5. `FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py`
  6. `FineFT/RL/DiHFT/high_level/vae_routing_util.py`
  7. `analysis_result/DiHFT/high_level_heurstic/fu/30min_parallel/diagnostics/diagnostics_summary.md`
- **报告归档位置**：`docs/research/2026-10-19-fu-30min-low-level-joint-metrics-evaluation-and-optimization.zh_cn.md`

---

## 1. 核心结论与执行摘要 (Executive Summary)

本研究针对用户提出的“评估 `two_dimensional_joint_metrics.csv`，查看 30min 策略训练结果的质量，是否存在优化方案”展开全链路、穿透式的一手数据与源码审计。核心结论如下：

1. **训练质量总体概览：结构性分化严重，局部高度繁荣与系统性盲区并存**：
   - 候选模型在 38 个 Epoch（35~72）与 11 个集成专家分箱（`bin_0`~`bin_10`）共 418 个模型中，整体单步均值收益为 **+0.0390**，但方差极大（std 达 0.5634，最小 -1.3875，最大 +2.2539）。
   - **做多与震荡能力强劲**：在上行趋势（`slope_label == label_2`）和横盘震荡（`slope_label == label_1`）中，候选模型正收益占比高达 **76.9% ~ 90.2%**，最强模型单步收益超 2.25。
   - **做空能力系统性衰竭**：在下行趋势（`slope_label == label_0`）中发生系统性溃败，平均收益为 **-0.39 ~ -0.65**，亏损候选占比高达 **80.4% ~ 96.1%**。9 个格子中用于应对下跌行情的 Slot 0 和 Slot 6 各自仅有 **1 个候选模型（0.24%）** 勉强过线，容错率几乎为零。

2. **发现第一大核心病理：Slot 2 选模空洞（Empty Model Trap）源于选模器除零统计缺陷，而非模型无盈利能力**：
   - 最终选模结果中，Slot 2（`label_0 x label_2`，低波动慢牛）被标记为 `empty_model`，在实盘/回测中被强制执行空仓拒单（`flat_position`）。
   - **数据实证穿透**：Slot 2 中全部 418 个候选模型的均值收益达 **+0.3323**（最高达 +0.8917），正收益占比高达 **86.0%**，最差初始仓位收益达标率 100%（418/418）。
   - **根因定位（一手代码证据）**：验证集中该格子仅包含 1 个合约（`fu2505`，`contract_count == 1`）。源码 [FineFT_two_dimensional_agent_selector.py:1020-1035](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:1020) 在 `contract_count <= 1` 时无法计算跨合约样本标准差，将 `standard_error` 和 `lcb_return_per_step` 均置为 `None`。而在第 1368 行的门控判断中，`pl.all_horizontal([pl.col(c) > min_lcb]).fill_null(False)` 将所有为 `None` 的 LCB 一票否决为 `False`。**导致 349 个高盈利、低回撤的模型遭到 100% 统计误杀，直接砸出“选模空洞”**！
   - **回测连锁破坏**：高层回测中因此触发了 1561 步防御性强制平仓（占比 17.5%），直接斩断趋势持仓并摧毁策略收益率。

3. **发现第二大核心缺陷：集成专家网络的马太效应与两极分化**：
   - `bin_0` 至 `bin_4`、`bin_7` 全周期均值收益为负（-0.081 ~ -0.003），严重拖累训练质量；
   - 收益几乎全由高序号专家提供（`bin_8`: +0.110, `bin_9`: +0.231, `bin_10`: +0.187）；在最终入选的 8 个有效模型中，`bin_9` 和 `bin_10` 垄断了 5 个槽位（62.5%）。集成多专家未实现预期的互补异构分工。

4. **发现第三大工程缺陷：联合评估中换手率监控彻底缺失**：
   - `two_dimensional_joint_metrics.csv` 中 `turnover_sum` 字段全列 7524 行恒等于 `0.0`。
   - 源码第 849 行硬编码 `pl.lit(0.0).alias("turnover_sum")`，使得选模阶段完全脱离换手率与交易摩擦考量，选出的模型在高层容易演变为“换手吞噬收益”。

---

## 2. 数据全景与九宫格覆盖度穿透分析 (3x3 Grid Detailed Audit)

### 2.1 基础元数据与样本量分布
- 文件规模：7,524 行，18 列
- 候选模型规模：38 个 Epoch（35 到 72），每个 Epoch 11 个 Bin（0 到 10），共计 **418 个候选 Candidate**。
- 双评估流：`route_context` 包含 `slope` 与 `volatility`。
- 组合全集：418 candidates × 2 route_contexts × 9 slots = **7,524 评估记录**。

### 2.2 九宫格（3x3）全景评估指标比对表

基于 `two_dimensional_joint_metrics.csv`、`two_dimensional_candidate_rankings.csv` 及 `two_dimensional_selection.csv` 计算整合：

| Slot ID | 市场状态 (波动 x 斜率) | 验证集支持合约数 | 候选盈利比例 (% > 0) | 候选均值收益 (mean) | 候选中位数收益 (median) | 候选 LCB 均值 | 门控合格候选数 (Eligible) | 最终选中模型 | Pair Score | 最终选定类型 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Slot 0** | `label_0 x label_0` (低波阴跌) | 4 | **3.95%** | -0.6550 | -0.6974 | -0.7160 | **1 / 418** | `epoch_48:bin_3` | 0.0522 | `model` |
| **Slot 1** | `label_0 x label_1` (低波横盘) | 5 | **90.19%** | +0.1836 | +0.1783 | +0.1101 | **267 / 418** | `epoch_68:bin_9` | 0.4904 | `model` |
| **Slot 2** | `label_0 x label_2` (低波慢牛) | **1** | **86.00%** | **+0.3323** | +0.3340 | **NaN** | **0 / 418** | **N/A** | **0.0000** | **`empty_model`** |
| **Slot 3** | `label_1 x label_0` (中波急跌) | 3 | **19.62%** | -0.4030 | -0.4284 | -0.4974 | **46 / 418** | `epoch_47:bin_10` | 0.6973 | `model` |
| **Slot 4** | `label_1 x label_1` (中波震荡) | 3 | **58.73%** | +0.0186 | +0.0141 | -0.0209 | **168 / 418** | `epoch_50:bin_4` | 0.3703 | `model` |
| **Slot 5** | `label_1 x label_2` (中波主升) | 3 | **83.25%** | +0.2702 | +0.2618 | +0.2112 | **265 / 418** | `epoch_51:bin_3` | 0.7151 | `model` |
| **Slot 6** | `label_2 x label_0` (高波暴跌) | 7 | **13.76%** | -0.3902 | -0.4035 | -0.9376 | **1 / 418** | `epoch_42:bin_9` | 0.0566 | `model` |
| **Slot 7** | `label_2 x label_1` (高波剧烈震荡) | 2 | **78.47%** | +0.4522 | +0.4636 | +0.3065 | **100 / 418** | `epoch_63:bin_10` | 0.9814 | `model` |
| **Slot 8** | `label_2 x label_2` (高波强牛) | 10 | **76.91%** | +0.5420 | +0.5218 | +0.3619 | **187 / 418** | `epoch_62:bin_10` | 1.8482 | `model` |

---

## 3. 策略训练质量深度诊断 (Core Quality Diagnostics)

### 3.1 诊断一：选模器算法除零漏洞致 Slot 2 选模空洞（Empty Model Trap）

- **数据事实 [已验证]**：
  在 `two_dimensional_selection.csv` 中，Slot 2 的入选模型为空，原因标注为 `no_candidate_passed_all_profitability_and_stability_gates`，最优被拒模型为 `epoch_35:bin_0`，被拒理由为 `lcb`。
- **机制穿透 [代码与公式证伪]**：
  查阅 [FineFT_two_dimensional_agent_selector.py:1020-1035](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:1020)：
  ```python
  .with_columns(
      pl.when(pl.col("contract_count") > 1)
      .then(pl.col("contract_std") / (pl.col("contract_count").cast(pl.Float64).sqrt()))
      .otherwise(None)
      .alias("standard_error")
  )
  .with_columns(
      pl.when(
          pl.col("standard_error").is_not_null() & pl.col("standard_error").is_finite()
      )
      .then(pl.col("mean_return_per_step") - self.config.lcb_z * pl.col("standard_error"))
      .otherwise(None)
      .alias("lcb_return_per_step")
  )
  ```
  而验证集在 30min 尺度下，`label_0 x label_2` 仅有合约 `fu2505` 一个合约（`contract_count == 1`），导致所有 418 个候选模型的 `standard_error` 全部为 `None`，进而 `lcb_return_per_step` 统统为 `None`。
  随后在第 1368 行的门禁判定中：
  ```python
  "lcb": pl.all_horizontal([pl.col(c) > self.config.min_lcb for c in lcb_columns]).fill_null(False)
  ```
  `.fill_null(False)` 直接使 418 个候选模型的 LCB 门禁全部判定为 `False`。
- **危害定量**：
  实际上，Slot 2 拥有 **349 个通过了 Mean Return > 0、Worst Initial Position >= -1.0 以及 PCR >= 0.60 的优质候选**！其中表现最好的候选（如 `epoch_69:bin_5` 单步均值收益达 0.788，`epoch_71:bin_10` 单步均值收益达 0.891）。将其误杀为 `empty_model` 后，高层 VAE 路由到该低波慢牛状态时强制平仓，白白丢掉所有盈利。

---

### 3.2 诊断二：做空能力系统性衰竭（Severe Bear-Market Asymmetry）

- **数据事实 [已验证]**：
  观察所有 `slope_label == label_0`（下跌行情）的三个格子：
  1. `Slot 0 (label_0, label_0)`：平均单步收益 **-0.655**，仅 16/418 收益为正，417/418 被淘汰，**仅 1 个候选合格**（`epoch_48:bin_3`，pair_score 仅 0.0522）；
  2. `Slot 3 (label_1, label_0)`：平均单步收益 **-0.403**，仅 46/418 合格；
  3. `Slot 6 (label_2, label_0)`：平均单步收益 **-0.390**，仅 42/418 收益为正，417/418 被淘汰，**仅 1 个候选合格**（`epoch_42:bin_9`，pair_score 仅 0.0566）。
- **深层归因 [机制推导]**：
  1. **日内 Bar 稀疏与跳空破坏做空**：30min 频度下一天仅 11 根 Bar，下跌行情多由隔夜跳空或开盘急挫构成，低层 Agent 单步持仓时间长，若没有及时的短周期空头止盈，极易在日内反弹中被动挨打。
  2. **逆势换手惩罚过重（Turnover Adverse Ratio = 5.0）**：在 [train_commodity_fu_30.sh](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/train/train_commodity_fu_30.sh:26) 中，逆势换手惩罚系数被设为 5.0（高达 100 bps）。在震荡阴跌中，Agent 一旦开空面临微观反弹，换手平仓被施加极重惩罚，导致 Agent 宁可被动持有或不敢做空，策略学向“多头偏执”或“死扛”。
  3. **反手能力关闭**：默认未开启高效反手机制，在 30min 级别需要花费两个决策周期（平仓后再开空），错失下跌黄金窗口。

---

### 3.3 诊断三：N=11 集成网络的头部极化与尾部退化 (Head Polarization)

统计 418 个模型按 11 个集成专家分箱（`bin_index` 0~10）在所有数据上的全局表现：

| 集成分箱 (`bin_index`) | 全局平均单步收益 (Mean Return) | 中位数收益 (Median) | 正收益占比 (Pct Positive) | 最差初始仓位均值 (Worst Pos) | 收益标准差 (Std) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `bin_0` | **-0.0814** | -0.0450 | 45.9% | -0.2202 | 0.4954 |
| `bin_1` | **-0.0764** | +0.0012 | 50.1% | -0.1896 | 0.5149 |
| `bin_2` | **-0.0522** | +0.0105 | 50.4% | -0.1594 | 0.5439 |
| `bin_3` | +0.0189 | +0.0829 | 57.8% | -0.0941 | 0.5885 |
| `bin_4` | **-0.0168** | +0.0636 | 55.4% | -0.1186 | 0.5358 |
| `bin_5` | +0.0430 | +0.0943 | 58.7% | -0.0625 | 0.5693 |
| `bin_6` | +0.0688 | +0.1533 | 64.6% | -0.0314 | 0.5605 |
| `bin_7` | **-0.0034** | +0.0689 | 57.3% | -0.0884 | 0.5567 |
| `bin_8` | **+0.1100** | +0.1056 | 60.3% | +0.0101 | 0.5410 |
| `bin_9` | **+0.2311** | +0.1637 | **64.9%** | **+0.1227** | 0.6094 |
| `bin_10` | **+0.1871** | +0.1156 | **60.3%** | **+0.0714** | 0.5813 |

- **关键洞察**：
  - 低序号分箱（0、1、2、4、7）普遍陷入亏损泥潭，均值收益均为负；
  - 高序号分箱（8、9、10）表现强劲，`bin_9` 均值收益高达 +0.2311，且在各种初始仓位下表现极稳健（worst_pos 为 +0.1227）；
  - 最终选中的 8 个有效模型中，`bin_9` 和 `bin_10` 拿下了 5 个，`bin_3` 拿下 2 个，`bin_4` 拿下 1 个。
  - 这表明集成网络中的损失分配或多样性正则不足，导致低序号专家严重退化，系统未能达成“11 个专家对应 11 种互补交易姿态”的初衷。

---

### 3.4 诊断四：Epoch 演化中的中期断崖式塌陷 (Mid-Epoch Slump)

分析 38 个 Epoch（35~72）的演进轨迹：

| Epoch 阶段 | 代表 Epoch 范围 | 平均单步收益 (Mean Return) | 正收益占比 (Pct Positive) | 最差初始仓位均值 | 动态特征与机制 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **初期爬坡** | Epoch 35 ~ 42 | -0.032 -> **+0.115** | 54.0% -> **63.6%** | -0.157 -> **+0.023** | 预训练 Warmup 结束，多尺度强化学习快速学习，表现单调攀升 |
| **中期崩塌** | Epoch 50 ~ 55 | **-0.091 -> -0.125** | **45.9% -> 39.8%** | **-0.234 -> -0.253** | 课程学习轮换（`curriculum_block_epochs 5`）与周期性衰减引发剧烈遗忘与动荡 |
| **晚期修复** | Epoch 68 ~ 72 | **+0.086 -> +0.139** | **61.6% -> 63.1%** | **+0.006 -> +0.028** | 学习率降至低位（`lr_min 0.0001`），策略在优质解附近重新稳固收敛 |

- **关键洞察**：
  模型训练并没有单调收敛，在 Epoch 50~55 发生了严重的性能雪崩。但在最终选模中，Slot 4 选了 `epoch_50:bin_4`，Slot 5 选了 `epoch_51:bin_3`，恰好选在中期塌陷期边缘，虽然在特定格子上分数达标，但模型整体抗逆能力存疑。

---

### 3.5 诊断五：换手率监控在联合评估中被彻底丢弃 (Turnover Blindness)

- **数据事实 [已验证]**：
  在 `two_dimensional_joint_metrics.csv` 中，`turnover_sum` 列的所有 7524 行数值全部为 `0.0`。
- **代码事实 [已验证]**：
  查阅 [FineFT_two_dimensional_agent_selector.py:848-854](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:848)：
  ```python
  grouped = (
      detail_df.group_by([...])
      .agg(
          pl.col("reward").cast(pl.Float64).sum().alias("reward_sum"),
          pl.len().alias("step_count"),
      )
      .with_columns(
          pl.lit(0.0).alias("turnover_sum"), # <--- 硬编码 0.0！
          ...
      )
  )
  ```
- **危害**：
  在边际评估（Marginal Metrics）中明明从 `analysis_result.csv` 读取了换手率，但在核心的联合评估中却直接赋 0.0 丢弃。导致选模阶段无法对高换手模型进行折价过滤，最终选出的模型可能在无摩擦假设下回报可观，但在高层接入摩擦后净利被吃空。

---

## 4. 全链路优化方案设计 (Actionable Optimization Roadmap)

根据上述一手事实与归因，优化方案按实施周期与生效层级划分为三个阶段：

### 4.1 第一阶段：选模器算法防御性修复（无需重训，立即可见成效）

#### 4.1.1 修复 Slot 2 LCB 统计除零缺陷，释放被误杀模型
- **代码修改方案**：
  在 [FineFT_two_dimensional_agent_selector.py](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/analysis/pick_agent/FineFT_two_dimensional_agent_selector.py:1020) 中，针对 `contract_count == 1` 提供合理的统计兜底，而不是返回 `None`：
  ```python
  # 改进方案：当 contract_count == 1 时，采用自身 step_std 或回退至边际标准差
  .with_columns(
      pl.when(pl.col("contract_count") > 1)
      .then(pl.col("contract_std") / (pl.col("contract_count").cast(pl.Float64).sqrt()))
      .otherwise(
          # 当仅有 1 个合约时，若样本步数充分（如 >= 100 步），采用保守折扣因子（如 0.1 * mean）或自身方差兜底
          pl.lit(0.0) if self.config.min_joint_contracts == 1 else None
      )
      .alias("standard_error")
  )
  ```
- **实测模拟验证效果**：
  以 `LCB = mean_return` 或采用边际 LCB 进行模拟重选，Slot 2 立即可以从 349 个合格候选者中选出：
  - 候选方案 A（边际 LCB 最高）：`epoch_71:bin_10`（边际 LCB 达 +0.6136，均值收益达 +0.891）
  - 候选方案 B（联合均值最高）：`epoch_69:bin_5`（单步均值收益达 +0.788，worst_pos 达 +0.788）
- **收益**：彻底消除 Slot 2 的 `empty_model` 漏洞，防止在 30min 低波慢牛行情中被迫空仓，预计可为高层回测恢复大量因 1561 步防御平仓错失的毛利。

#### 4.1.2 修复 `turnover_sum` 提取，接入摩擦感知选模
- 在 `_load_joint_detail_rows` 中，从 `trading_action_detail_*.csv` 中真实读取 `动作变化` 或 `交易计数`，还原每一步的实际换手额，将换手损耗纳入 `pair_score` 的扣减公式：
  $$\text{Adjusted Score} = \text{LCB} - \lambda_{\text{turnover}} \times \frac{\text{turnover\_sum}}{\text{step\_count}}$$

---

### 4.2 第二阶段：30min 验证集切片与分位数门限重新校准

- **数据根因**：
  查阅 [dataset/30min/fu/train/regime_thresholds.json](/home/lanceliang/opt/aiwork/FineFT_code_space2026/dataset/30min/fu/train/regime_thresholds.json)，30min 阈值是在全合约训练集上拟合而来。但在验证集 12 个合约中，`label_0 x label_2` 仅覆盖了 1 个合约（`fu2505`，占比仅 6.3%）。相比之下，10min 数据集在同样格子上覆盖了 5 个合约。
- **优化动作**：
  根据 ADR 0009（Shared Cross-Contract Label Thresholds）与 ADR 0013（Unified Segment Regime Calibration），重新校验 30min 维度的收益率与波动率时间截断步长，保证验证集的 9 个格子中，每个格子至少有 3 个以上合约获得支撑，避免单合约样本偏差。

---

### 4.3 第三阶段：低层 RL 训练做空能力与专家多样性重构（训练端重跑建议）

针对 Slot 0、3、6 做空能力溃败和 Bin 0~4 专家退化，在下一轮训练脚本 [train_commodity_fu_30.sh](/home/lanceliang/opt/aiwork/FineFT_code_space2026/FineFT/script/train/train_commodity_fu_30.sh) 中调整核心参数：

1. **下调逆势换手惩罚率**：
   - 将 `--turnover_adverse_ratio 5.0` 下调至 **`2.5 ~ 3.0`**（原 5.0 导致单步 1% 本金惩罚，使 Agent 在 30min 下跌初期的试错开空代价过高，彻底锁死了做空探索）。
2. **平滑 TD 步长与贴现**：
   - 当前 30min 脚本中 `--n_step 8 --gamma 0.992`，对应 4 小时单日日内贴现；切忌增大至 18（跨日跳空会破坏价值网络收敛）。保持 `n_step=8` 或微调至 `6`。
3. **引入下跌行情重采样（Regime-Stratified Sampling）**：
   - 根据 ADR 0012，在经验回放池抽样中对 `slope_label == label_0` 的切片增加 30%~50% 的重采样权重，强化网络对下行单边与破位行情的学习。
4. **增加集成专家多样性正则（Ensemble Diversity Loss）**：
   - 在多头 Q 网络损失函数中加入专家间预测差异的协方差惩罚（Cosine Similarity Penalty），防止所有专家在梯度下降中被同化或放弃低序号分箱。
5. **开启 30min 反手支持（`allow_reverse_position=True`）**：
   - 根据 ADR 0001，在 30min 级别开启反手开仓，避免“先平仓等一根 30min Bar 再开空”产生的严重踏空与摩擦。

---

## 5. 结论对比表：现状 vs 优化后预期

| 维度 | 当前 30min 训练与评估现状 | 实施优化方案后预期 |
| :--- | :--- | :--- |
| **Slot 2 (低波慢牛)** | 判定为 `empty_model`，运行时被迫空仓拒单（1561 步防御平仓） | 算法修复后入选 `epoch_71:bin_10`（均值 0.89），恢复正常多头交易 |
| **Slot 0 & Slot 6 (单边下跌)** | 仅 1 个独苗合格（通过率 0.24%），全网做空能力崩塌 | 降低逆势惩罚+下跌重采样后，通过率预计提升至 15%~25% |
| **集成网络结构** | Bin 0~4 严重退化，高序号 Bin 9~10 垄断策略 | 11 个专家实现多尺度/多风格有效异构，降低单专家故障风险 |
| **换手与摩擦感知** | 评估阶段换手率为 0.0，高层易发生过度换手 | 真实注入换手摩擦，抑制回测中 23 倍的高额本金换手 |
| **整体策略收益率 (TR)** | 从 0.05 跌至 0.0219，毛净差额吃掉 28.5% 毛利 | 补齐槽位漏洞并抑制无效换手后，TR 有望重回 0.05 以上基线 |

