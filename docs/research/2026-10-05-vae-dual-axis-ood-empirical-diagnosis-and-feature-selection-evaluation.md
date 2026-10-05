# VAE 双轴特征分布外 (OOD) 实证诊断与特征选择流程评估报告

- **报告编号**：RES-2026-1005-02
- **研究日期**：2026-10-05
- **研究对象**：FineFT 商品期货（燃料油 FU 10min 级别）VAE 双轴解耦表征架构（`vae_slope` 与 `vae_volatility`），涉及脚本 `FineFT/script/analysis/feature/vae_feature_ood_fu_10.sh`、诊断模块 `FineFT/analysis/feature/vae_feature_ood_analysis.py`、特征选择产物 `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/` 及底层三流特征筛选流水线。
- **核心结论**：
  1. **架构解耦效果卓越**：双轴解耦后，测试集相对训练集的平均负对数似然增量（$\Delta\text{NLL}_{\text{test\_vs\_train}}$）从单体混合架构的 $+26.0 \sim +60.0$ 彻底降至斜率轴 **-0.146** 与波动率轴 **+0.534**，分布外重构塌陷被彻底根除。
  2. **斜率 VAE (14 维) 极度稳健**：全特征 $\Delta\text{NLL} \le 0.015$，方差比 $1.01 \sim 1.05\text{x}$，完全无漂移，特征选择结果最优，应完全冻结。
  3. **波动率 VAE (13 维) 存在单一离群特征**：`historical_volatility_2` 单一特征贡献了测试集 NLL 增量的 **76.52%**（$\Delta\text{NLL} = +0.408$，测试集方差比达到 **1.62x**，均值漂移 $0.21\sigma$）。其根本原因是 $w=2$（20分钟）窗口过窄，存在极端样本噪声与换月聚集性波动跳跃。
  4. **特征替换与正交聚类**：将 `historical_volatility_2` 加入 10min 波动率黑名单后，依据 Ward 层次聚类与 VIF 多重共线性检验规则，最优正交替换特征为 **`roc_16_origin`**。该特征既不存在于 `rl_state_features.npy`，也不存在于 `vae_slope_state_features.npy`，具备极佳的独立增量信息价值。

---

## 1. 第一手证据源 (Primary Sources)

本研究严格基于实际运行诊断输出、实证日志与中间特征文件展开：

1. **实证诊断输出与明细报表**：
   - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/slope/`：
     - `feature_ood_summary.csv`：14 个斜率特征的主汇总表，总 $\Delta\text{NLL}_{\text{test\_vs\_train}} = -0.1464$。
     - `feature_ood_test_vs_train.csv`：测试集与训练集特征重构差异明细。
     - `contracts/`：13 个测试合约（`fu2508` ~ `fu2609`）分合约重构误差诊断。
   - `analysis_result/DiHFT/feature_ood/fu/10min_parallel/volatility/`：
     - `feature_ood_summary.csv`：13 个波动率特征的主汇总表，总 $\Delta\text{NLL}_{\text{test\_vs\_train}} = +0.5338$。
     - `feature_ood_test_vs_train.csv`：测试集与训练集特征重构差异明细。
     - `contracts/`：13 个测试合约分合约重构误差诊断。
   - `log/analysis/feature/DiHFT/fu/10min_parallel/slope/feature_ood.log`
   - `log/analysis/feature/DiHFT/fu/10min_parallel/volatility/feature_ood.log`

2. **特征选择配置与工程产物**：
   - `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/feature_selection_manifest.json`：特征选择全量元数据。
   - `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/vae_volatility_state_features.npy`
   - `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/vae_slope_state_features.npy`
   - `PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/rl_state_features.npy`
   - `data_preprocess/operator_futures/feature_selection/commodity_feature_blacklists.json`

---

## 2. VAE 双轴实证 OOD 诊断深度分析

### 2.1 斜率轴 (Slope VAE, 14 维) 实证表现

在测试集上评估 14 个斜率特征的高斯重构负对数似然：
- **总 $\Delta\text{NLL}_{\text{test\_vs\_train}}$**：$-0.1464$（测试集样本重构似然甚至略优于训练集基准）。
- **最大单特征漂移**：`imax_48_origin`，$\Delta\text{NLL} = +0.0155$，方差比 $1.0096\text{x}$。
- **次大单特征漂移**：`beta_16_std_norm_origin`，$\Delta\text{NLL} = +0.0140$，方差比 $1.0331\text{x}$。
- 其余 12 个特征的 $\Delta\text{NLL}$ 均为负值或小于 0.008。
- **诊断结论**：斜率轴特征流在跨合约跨时间切片上具有极高的平稳性与泛化度，特征选择完全收敛，**保持完全冻结**。

### 2.2 波动率轴 (Volatility VAE, 13 维) 实证表现

在测试集上评估 13 个波动率特征的高斯重构负对数似然：
- **总 $\Delta\text{NLL}_{\text{test\_vs\_train}}$**：$+0.5338$（处于健康容忍门限 $\le 12.0$ 范围之内）。
- **离群异常特征**：`historical_volatility_2`。
  - 单特征 $\Delta\text{NLL}$：$+0.4084$。
  - 占全轴总增长贡献比（`contrib_pct`）：**76.52%**！
  - 测试集与训练集方差比（`var_ratio`）：**1.6224x**（测试集方差暴增 62.2%）。
  - 均值漂移（`mean_shift`）：$0.2094\sigma$。
  - 分合约表现：在 13 个测试合约中，`historical_volatility_2` 在部分合约贡献率超过 100%（例如 `fu2511` 为 281.4%，`fu2512` 为 135.7%，`fu2607` 方差比高达 2.44x）。
- 其余 12 个特征：扣除 `historical_volatility_2` 后，剩余 12 个特征的 $\Delta\text{NLL}$ 总和仅为 $0.1254$，平均每个特征单步漂移 $< 0.010$。

### 2.3 异常特征数学机理分析

为什么 `historical_volatility_2` 会产生严重漂移？
1. **采样窗口过短**：在 10min 采样频率下，$w=2$ 仅对应 20 分钟的时间窗口（仅含 2 根 K 线）。
2. **估计量极端发散**：样本标准差计算公式 $s = \sqrt{\frac{1}{n-1}\sum (r_i - \bar{r})^2}$ 在 $n=2$ 时自由度仅为 1，估计量方差达到极值。
3. **厚尾波动率跳跃敏感**：商品期货开盘跳空、大单撮合或流动性切换会导致 20 分钟历史波动率产生剧烈的脉冲性震荡，造成跨合约特征分布方差剧烈放缩（测试集方差比 $1.62 \sim 2.44\text{x}$）。

---

## 3. 特征选择优化方案与替换推导

### 3.1 特征选择流程是否需要重构？

**评估结论：三流特征选择整体流程无需重构，杜绝过度工程。**
- 原有流程（阶段 1.1 数据清洗 $\to$ 阶段 1.2 跨合约 PSI 漂移门控 $\to$ 阶段 1.3 预测力门控 $\to$ 阶段 1.4 状态分桶方差分析 $\to$ 阶段 1.5 综合评分排序 $\to$ 阶段 3 Ward 层次聚类与 VIF 剪枝）结构严密，已将 OOD 漂移控制在 $+0.53$ 的极低水平。
- 绝不宜为了单个极端离群特征去修改阶段 3 的聚类机制或强行加入量价配额启发式规则，最精简、最稳健的方式是通过黑名单机制（`Stream Blacklist`）在入口处剔除超短周期噪声项。

### 3.2 替换特征严格推导

在黑名单中加入 `"historical_volatility_2"` 后，综合评分池（Composite Score Pool）剩余 26 个候选特征。依据 `pipeline.py` 的严格数学规则推导替换特征：

1. **若依据层次聚类保持 12 个候选簇（总计 13 维特征）**：
   - 原本包含 27 个特征时，距离阈值截断产生 12 个簇，`historical_volatility_2` 单独成簇。
   - 剔除 `historical_volatility_2` 后，在 12 个簇的分裂中，`roc_16_origin` 从原本与 `sumn_24_origin` 相邻的聚类中自然分裂，成为独立的第 12 簇簇首（合并距离 $0.4473$，仅略高于截断阈值 $0.4472$）。
   - 经 `prune_by_vif` 检验，在 `max_vif=8.0` 下多重共线性检测**完全通过**（丢弃数为 0）。

2. **统计学与预测力指标对比**：
   - `roc_16_origin`（160分钟价格变化率）：
     - `mean_psi`: $0.0639$（远优于门限 $0.12$）
     - `max_pair_psi`: $0.2154$（远优于门限 $0.25$）
     - `forward_psi`: $0.0272$（样本外时间前向分布极度稳定）
     - `VolRankIC_Mean`: $0.0383$（优于门限 $0.030$）
     - `VolRankIC_IR`: $0.927$（优于门限 $0.40$）
     - `VolSignConsistency`: $0.8367$（优于门限 $0.75$）

3. **跨特征流正交性验证**：
   - 在 `rl_state_features.npy`（143 维）中：**不存在**（`False`）。
   - 在 `vae_slope_state_features.npy`（14 维）中：**不存在**（`False`）。
   - 结论：`roc_16_origin` 作为新特征替换进入 `vae_volatility_state_features.npy`，不仅完美修补了波动率表征维度，而且在三大模型流之间保持绝对正交，无任何冗余重复。

---

## 4. 实施清单与变更记录

1. **黑名单配置更新**：
   - 文件：`data_preprocess/operator_futures/feature_selection/commodity_feature_blacklists.json`
   - 在 `frequencies["10min"]["vae_volatility"]` 中追加 `"historical_volatility_2"`。
2. **特征数组更新**：
   - 文件：`PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/vae_volatility_state_features.npy`
   - 将原列表中的 `'historical_volatility_2'` 替换为 `'roc_16_origin'`，保持 13 维结构。
   - 同步更新：`dataset/10min/fu/vae_volatility_state_features.npy`。
3. **特征选择元数据清单更新**：
   - 文件：`PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION/10min/fu/train/feature_selection_manifest.json`
   - 更新 `vae_volatility_stream.selected_features`、`filter_results` 及全量 `selected_features`。
4. **单元测试与回归防护**：
   - 文件：`data_preprocess/tests/test_commodity_feature_blacklists_triple_stream.py`，新增断言保护；
   - 运行分析与黑名单测试套件，全部 35 个测试用例通过（35 passed）。
