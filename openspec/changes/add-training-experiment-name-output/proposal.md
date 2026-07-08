# add-training-experiment-name-output

## 背景与目标

`FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 当前使用 `--base_path` 和 `--dataset_name` 定位输入数据，同时也用 `dataset_name` 组织模型、TensorBoard 和日志输出。用户在不同时间窗口、频率和参数组合上重复训练时，需要用实验名隔离输出，避免同一数据集下的训练结果互相覆盖或混在同一目录。

本变更目标是为串行 Stage I 训练入口新增实验名参数，只控制输出路径；输入数据目录继续沿用现有 `base_path/dataset_name` 语义。

## 用户场景

- 同一个 `dataset_name` 下，使用不同时间窗口或频率对应的 `base_path` 训练，并希望输出目录通过实验名区分。
- 同一个窗口下，调整 `gamma`、`n_step`、手续费、仓位数量等训练参数，并希望每组参数的模型、TensorBoard 和日志保存到独立目录。
- 旧训练命令不额外传参时，也应有明确默认实验名，避免输出目录缺少实验层级。

## 设计方向

采用最小改动：在 `weight_advantage_pretrain.py` 新增 `--experiment_name`，默认值为 `default`。训练输入仍由 `--base_path` 和 `--dataset_name` 决定，不新增输入数据目录参数。

串行训练结果路径从 `<result_path>/<dataset_name>/weights_advantage_pretrain` 调整为 `<result_path>/<dataset_name>/<experiment_name>/weights_advantage_pretrain`。日志路径从 `log_futures/<dataset_name>/low_level/train/advantage.log` 调整为 `log_futures/<dataset_name>/low_level/train/<experiment_name>/advantage.log`。

## 关键决策

- 只修改串行训练入口 `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 和必要的串行训练 shell 脚本。
- 不修改 `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`。
- 不新增 `dataset_root`、`train_data_path` 或多实验配置文件；输入目录继续由现有 `base_path/dataset_name` 控制。
- `--experiment_name` 默认值为 `default`，所有输出都带实验名层级。

## 范围边界

**包含：**
- 为串行训练脚本新增 `--experiment_name default`。
- 将模型、TensorBoard、qtable diagnostics 等位于 `self.model_path` 下的输出隔离到实验名目录。
- 将训练日志隔离到实验名目录。
- 更新串行商品训练 shell，使其可通过环境变量或默认值传入实验名。
- 增加轻量单元测试和语法验证。

**不包含（本次）：**
- 不修改并行训练脚本。
- 不改变输入数据目录解析逻辑。
- 不实现一次命令批量跑多个实验。
- 不迁移旧结果目录或兼容旧输出目录结构。
- 不运行真实长训练。

## 验收标准

- [ ] `weight_advantage_pretrain.py` 支持 `--experiment_name`，默认值为 `default`。
- [ ] 输入目录仍由 `base_path/dataset_name` 控制，训练数据路径保持 `<base_path>/<dataset_name>/train`。
- [ ] 模型和 TensorBoard 输出路径包含 `<dataset_name>/<experiment_name>/weights_advantage_pretrain`。
- [ ] 日志文件路径包含 `<dataset_name>/low_level/train/<experiment_name>/advantage.log`。
- [ ] `parallel_weight_advantage_pretrain.py` 不被修改。
- [ ] 轻量测试和 `py_compile` 通过。
