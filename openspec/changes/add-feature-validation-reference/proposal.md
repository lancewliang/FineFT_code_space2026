# add-feature-validation-reference

## 背景与目标

当前商品 futures preprocess 已迁移到 Polars 实现，但需要一个独立验证模块确认特征产物不仅字段齐全，而且公式计算结果正确。`docs/data/*.md` 已列出各阶段应产出的因子列，另一个工作区 `/home/lanceliang/opt/aiwork/FineFT_code_space2026_2/data_preprocess/operator_futures` 保留了原始 pandas 实现，可作为更精准的 reference 计算来源。

目标是新增一个独立运行的特征验证模块，通过旧 pandas reference 对中间产物和最终产物进行抽样逐列对账，生成 Markdown/JSON 报告，帮助发现字段缺失、列口径偏差和公式实现错误。

## 用户场景

- 维护者在商品 preprocess 主流程跑完后，单独运行验证脚本，对 `fu` 的指定日期区间生成特征对账报告。
- Polars 实现发生改动后，维护者用旧 pandas reference 重算 docs 中列出的全部因子，快速发现公式偏差。
- 对账失败时，维护者可以从报告中看到具体 stage、列名、timestamp、实际值、reference 值和最大绝对误差。

## 设计方向

新增独立验证入口：

```text
data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh
```

该 shell 脚本只负责参数、环境和日志/报告路径，内部调用 Python CLI：

```text
data_preprocess/operator_futures/feature_validation/validate_features.py
```

验证模块从中间产物开始读取，不从原始 CSV 重新计算。默认覆盖以下阶段：

- `cross_section`
- `merge_concat`
- `time_feature`
- `merge_clean`
- `ic_correlation`
- `scale_save`

旧 pandas 实现复制到 validation-only 目录，例如：

```text
data_preprocess/operator_futures/feature_validation/pandas_reference/
```

第一版建议复制的 reference 模块：

- `cross_section/base_feature_util.py`
- `cross_section/create_feature.py`
- `features_related/feature_util.py`
- `features_related/base_feature.py`
- `time_operator/multi_processing_util.py`
- `time_operator/create_feature_multi_processing.py`
- `time_operator/time_operator_util.py`
- `merge_concat/merge.py`
- `merge_concat/concat.py`
- `merge_all/merge_clean.py`
- `feature_selection/cor_util.py`
- `feature_selection/ic_correlation.py`
- `scale_describe_save/scale_save.py`
- `util.py`

验证器不运行时解析 markdown。根据 `docs/data/*.md` 和旧 pandas 代码生成固定的 `expected_columns.py`、固定 validators 和固定 reference 调用逻辑。

默认对比方式：

- 覆盖 docs 中列出的全部因子列。
- 对行做抽样逐列比对。
- 按 `timestamp` 对齐。
- 数值容忍度为 `abs_diff <= 1e-9`。
- 输出 Markdown 和 JSON 报告。

## 关键决策

- 独立 `.sh` 入口，不接入 `main.sh`，避免影响现有 preprocess 主流程。
- 从中间产物开始验证，不从原始行情 CSV 全链路重算。
- 使用旧 pandas 实现作为 validation-only reference，不污染现有 Polars 生产实现。
- 不做动态 docs markdown 解析；字段清单和验证逻辑固定编码，docs 变更时通过代码 review 同步更新。
- 默认抽样行逐列对比，避免全量逐行验证导致运行时间不可控。
- 报告为主，同时保留 CLI 退出码，便于后续 CI 或脚本自动判断。

## 范围边界

**包含：**

- 新增商品特征验证 `.sh` 入口。
- 新增 `feature_validation` Python 模块。
- 复制旧 pandas reference 所需模块到 validation-only 目录。
- 固化 docs 中列出的 expected columns。
- 对中间产物和最终产物进行抽样逐列对账。
- 生成 Markdown/JSON 验证报告。
- 增加单元测试、reference import/适配测试和集成 smoke test。

**不包含（本次）：**

- 不修改现有 Polars 生产计算逻辑。
- 不把验证器接入 `main.sh`。
- 不从原始 CSV 全链路重算。
- 不做运行时 markdown 解析。
- 不复制模型训练、CatBoost、Lasso、rank IC 或去重策略相关模块，除非后续明确需要。
- 不要求第一版所有 stage 都 pass；若旧 pandas reference 与当前商品特化口径存在差异，报告应明确记录 fail/partial 原因。

## 验收标准

- [ ] 可以通过 `validate_features.sh` 独立运行验证指定商品、频率和日期区间。
- [ ] 验证器使用复制的旧 pandas reference 代码进行重算，不 import 生产 Polars 实现作为 reference。
- [ ] 验证器覆盖 `docs/data/*.md` 中固化的全部因子列清单。
- [ ] 验证器按 `timestamp` 抽样逐列比对，默认 `abs_diff <= 1e-9`。
- [ ] 验证报告包含每个 stage 的 pass/fail/partial/error 状态、缺失列、多余列、未验证列、最大误差和失败样本。
- [ ] CLI 退出码区分全部通过、存在验证失败和参数/输入错误。
- [ ] 不修改 `main.sh`，不改变现有 preprocess 产物路径和生产计算逻辑。
