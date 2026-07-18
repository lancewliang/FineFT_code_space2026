# add-scale-save-nan-validation

## 背景与目标

`data_preprocess/operator_futures/scale_describe_save/scale_save.py` 会读取特征选择后的 feather 文件，缩放 state 特征，并写出 scale-save 数据集。当前流程没有在处理前后显式阻断 NaN，可能让含 NaN 的输入继续进入缩放流程，或让处理后含 NaN 的输出落盘，影响后续训练与诊断。

本变更目标是在 `scale_save.py` 增加两个固定 NaN 检查点：开始处理输入文件后立即检查，处理完成并准备写文件前再次检查。任一检查发现 NaN 时，脚本直接报错退出，且不写出新的输出文件。

## 用户场景

- 预处理人员运行 scale-save 脚本时，如果输入 `df_name.feather` 已含 NaN，希望脚本立即失败，避免继续处理污染数据。
- 预处理人员运行 scale-save 脚本时，如果缩放后的最终 `out` 含 NaN，希望脚本在落盘前失败，避免生成不可用输出。

## 设计方向

采用集中校验函数加两个固定检查点的方案，只增强 `data_preprocess/operator_futures/scale_describe_save/scale_save.py`，不新增 CLI 参数，不引入通用数据质量模块。

在 `main()` 里保留现有处理顺序：解析路径和特征选择，读取输入 `df_name.feather`，立即检查整个输入 `df`；随后按现有逻辑拆分 reward/state、执行 `scale_std` 和 `scale_mean`、拼出最终 `out`；在任何 `write_ipc`、`write_csv`、`np.save` 或 `df_describe.write_csv` 调用前检查整个 `out`。校验通过后，现有输出行为保持不变。

新增一个小的 Polars DataFrame 校验函数，职责是检查所有可表达 NaN 的列，发现 NaN 时抛出 `ValueError`。错误信息需要包含检查阶段、对应路径和含 NaN 的列名，方便定位问题。

## 关键决策

- 检查范围严格限定为主输入 `df_name.feather` 和最终准备写出的 `out`。
- 后置 NaN 检查必须发生在任何输出写入前；失败时不写 `df.feather`、`df.csv`、`state_features.npy`、`df_describe.csv`。
- 不单独检查 `state_features.npy`，也不把 `df_describe.csv` 作为独立 NaN 检查目标。
- 不改变命令行参数、路径规则、特征选择规则或缩放算法。
- 校验失败直接抛出 `ValueError`，让脚本以非 0 状态退出；现有文件读取失败、缺列等错误保持原行为。
- 后续验证运行 Python 脚本或测试时，使用 `conda activate finetf` 后执行。

## 范围边界

**包含：**
- 在读取主输入 feather 后检查整个输入 `df` 是否含 NaN。
- 在最终 `out` 写出前检查整个 `out` 是否含 NaN。
- 校验失败时报错退出，并在错误信息中包含阶段、路径和含 NaN 的列名。
- 为输入 NaN、输出 NaN、正常成功路径添加最小测试覆盖。

**不包含（本次）：**
- 不检查所有中间 DataFrame。
- 不单独检查 `state_features.npy`。
- 不单独检查 `df_describe.csv`。
- 不修改缩放公式、特征选择逻辑、输出文件名或 CLI 参数。
- 不抽取跨脚本通用数据质量模块。

## 验收标准

- [ ] 输入 `df_name.feather` 含 NaN 时，脚本在处理前报错退出，错误信息包含 `input` 阶段、输入路径和含 NaN 的列名。
- [ ] 输入 `df_name.feather` 含 NaN 时，脚本不写出 `df.feather`、`df.csv`、`state_features.npy`、`df_describe.csv`。
- [ ] 输入无 NaN 但最终 `out` 含 NaN 时，脚本在任何输出写入前报错退出，错误信息包含 `output` 阶段、目标输出路径和含 NaN 的列名。
- [ ] 输出阶段 NaN 校验失败时，不写出 `df.feather`、`df.csv`、`state_features.npy`、`df_describe.csv`。
- [ ] 正常输入且最终 `out` 无 NaN 时，脚本成功生成既有四类输出文件。
- [ ] 现有 CLI 参数和成功路径输出格式保持兼容。
