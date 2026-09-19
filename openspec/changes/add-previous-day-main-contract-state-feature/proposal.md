# Change: 增加前一交易日主力合约状态特征 (add-previous-day-main-contract-state-feature)

## Why

在商品期货多合约连续训练和回测中，强化学习 Agent 需感知当前持仓或交易合约在市场中所处的流动性地位（即是否为主力合约）。此前仅在跨月结构特征（`CROSS_MONTH_FEATURE`）中计算了 `cm_contract_role_main`，但其与多合约期限结构特征强耦合，且参与了 `Scale Save` 的 RobustScaler 缩放，导致三档档位语义可能发生均值偏移/反转。为了保证单合约环境与轻量预处理流水线具备原生的主力身份感知能力，需要将“前一交易日是否为主力合约”解耦沉淀为 `Base_Time_feature` 的原生状态特征，并遵循 ADR-0003 实行未缩放直通（Passthrough）。

## What Changes

1. **扩展 `Base_Time_feature`**：
   - 增加 `prev_day_contract_role_tier` 特征列，使 `BASE_TIME_FEATURE_COLUMNS` 从 11 维扩充为 12 维。
   - `generate_base_time_features` 及对应算子新增对 `main_sub_roles` 或 `main_contract_summary.json` 的查询能力，基于上一交易日（$T-1$ 日）判断当前合约是否为日成交量排第一（持仓量平局修正）的主力合约。
2. **继承 ADR-0003 Passthrough 规范**：
   - `prev_day_contract_role_tier` 严格保持 主力为 `1.0`、次主力为 `0.5`、非主力为 `0.0`的离散语义。
   - 在 `Feature Selection` 中作为 `mandatory_state_features` 强制保留在 `state_features.npy`。
   - 在 `Scale Save` 中作为 `passthrough_state_features` 旁路跳过 RobustScaler 缩放。
3. **冷启动/边界缺省处理**：
   - 当遇到合约生命周期的首个交易日，或数据窗口起始点在 summary 中无法找到前序交易日时，严格确定性填充 `0.0`，不产生 `NaN` 或异常。

## Impact

- 涉及模块：
  - `data_preprocess/operator_futures/commodity/base_time_feature.py`
  - `data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh`
  - 依赖 `BASE_TIME_FEATURE_COLUMNS` 的特征合并、特征选择和缩放脚本。
- 兼容性：
  - 下游模型状态输入维度（`state_features`）增加 1 维。重新运行完整预处理或生成新数据集后，环境和 Agent 自动通过 `state_features.npy` 感知并适配该特征。
  - 现有只跑 `cross_month_feature` 的旧产物不受破坏性影响。

## Problem Statement

商品期货不同合约（如主力月、次主力月、远月冷门合约）的流动性深度、滑点和价差波动特性存在显著差异。强化学习策略如果在不同流动性状态间迁移，需要感知当前自身合约是否为主力合约。

当前系统虽然在 `cross_month_feature.py` 中实现了 `cm_contract_role_main`，但存在三个架构局限：
1. **模块强耦合**：`cross_month_feature` 属于跨交割月期限结构特征，生成需要至少 3 个活跃合约同时存在并按秒级时间戳对齐，开销大；对于只想跑单合约轻量化特征或无需价差特征的场景无法使用。
2. **缩放语义失真**：ADR-0004 规定全部跨月特征必须参与 `Scale Save` 的 RobustScaler 缩放。当训练集中主力合约样本占比超过 50% 时，中位数为 1.0，导致主力合约在标准化后变为 0.0，而非主力合约变为 -1.0，破坏了 0/1 离散指示的物理含义。
3. **特征定位不清晰**：“当前合约在上一日是否为主力”属于合约自身生命周期与市场地位属性，与 `contract_life_remaining_ratio`、`contract_month_sin/cos` 等同构，理应归属于合约时间与元数据编码模块（`Base_Time_feature`）。

## Solution

在 `base_time_feature.py` 中增加对 `prev_day_contract_role_tier` 的计算：
1. 算子接收 `summary_path` 或已解析的 `main_sub_roles: dict[str, dict[str, str]]`。
2. 对当前处理的 `trading_day` 和 `contract`，过滤出所有严格小于 `trading_day` 的历史交易日列表 `prior_days`。
3. 若 `prior_days` 非空，取最近一个历史日 `role_trading_day = prior_days[-1]`，检查 `roles = main_sub_roles[role_trading_day]`，当 `roles.get(contract) == "main"` 时置为 `1.0`，否则置为 `0.0`。
4. 若 `prior_days` 为空（首日冷启动或样本起始边界），置为 `0.0`。
5. 将 `prev_day_contract_role_tier` 纳入 `BASE_TIME_FEATURE_COLUMNS`。由于既有流水线均已配置 `BASE_TIME_FEATURE_COLUMNS` 自动进入 `--mandatory_state_features` 与 `--passthrough_features`，新特征自动享受强制保留与不缩放直通保障。

## User Stories

1. **作为量化研究员/RL 工程师**，我希望模型输入包含明确的 `prev_day_contract_role_tier` 特征，以便策略网络能根据合约主力状态学习区分不同的交易与持仓风格。
2. **作为预处理管道维护者**，我希望新特征计算独立、快速，在单合约维度仅通过查表完成，且在数据边界确定性回退为 `0.0`，杜绝 `NaN`、`Inf` 或运行时异常。
3. **作为系统架构师**，我希望该二值特征严格保持原始 `1.0 / 0.5 / 0.0` 离散值传递给下游，不被 RobustScaler 扭曲。
