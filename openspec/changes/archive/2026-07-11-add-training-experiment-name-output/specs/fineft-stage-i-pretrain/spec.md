## ADDED Requirements

### Requirement: 串行 Stage I 训练 SHALL 使用实验名隔离输出
系统 SHALL 在 `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` 中支持实验名参数，用于隔离串行 Stage I 训练输出，同时保持现有输入数据目录语义不变。

#### Scenario: 默认实验名参数
- **WHEN** 用户解析 `weight_advantage_pretrain.py` CLI 参数且未传入 `--experiment_name`
- **THEN** `experiment_name` SHALL 默认为 `default`
- **AND** 用户 SHALL 能通过 `--experiment_name <name>` 显式指定实验名

#### Scenario: 输入数据目录保持 base_path dataset_name 语义
- **WHEN** `Weighted_Contexts_DQN.__init__` 初始化训练数据路径
- **THEN** 系统 SHALL 继续使用 `os.path.join(args.base_path, args.dataset_name, "train")` 作为 `self.train_data_path`
- **AND** 系统 SHALL 继续从 `os.path.join(args.base_path, args.dataset_name, "state_features.npy")` 读取特征列表
- **AND** 系统 SHALL 继续从 `os.path.join(args.base_path, args.dataset_name, "maintenance_margin_ratio_dict.npy")` 读取保证金比例字典
- **AND** 本变更 SHALL NOT 新增或改用独立输入数据目录参数

#### Scenario: 模型和 TensorBoard 输出包含实验名层级
- **WHEN** 用户使用 `--dataset_name fu --experiment_name window_5min_gamma097` 启动串行训练
- **THEN** `self.model_path` SHALL 等于 `os.path.join(args.result_path, "fu", "window_5min_gamma097", "weights_advantage_pretrain")`
- **AND** `self.log_path` SHALL 位于 `self.model_path` 下的 `log`
- **AND** epoch 模型、TensorBoard 文件和 `qtable_diagnostics` SHALL 继续写入 `self.model_path` 及其子目录

#### Scenario: 训练日志输出包含实验名层级
- **WHEN** `configure_logger` 为 `dataset_name=fu` 和 `experiment_name=window_5min_gamma097` 配置文件日志
- **THEN** 日志文件 SHALL 写入 `log_futures/fu/low_level/train/window_5min_gamma097/advantage.log`
- **AND** 不同 `experiment_name` 的训练日志 SHALL 写入不同目录

#### Scenario: 串行商品训练脚本传递实验名
- **WHEN** 用户运行串行商品训练 shell
- **THEN** shell SHALL 将实验名传递给 `weight_advantage_pretrain.py`
- **AND** shell SHALL 提供默认实验名 `default`
- **AND** shell 的 stdout 重定向路径 SHALL 与实验名日志目录一致

#### Scenario: 并行训练文件不受影响
- **WHEN** 本变更实现完成
- **THEN** 系统 SHALL NOT 修改 `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py`
- **AND** 并行训练输出路径语义 SHALL 保持本变更前的行为

#### Scenario: 轻量验证命令
- **WHEN** 变更实现完成
- **THEN** `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py` SHALL 成功
- **AND** focused tests SHALL 覆盖默认实验名、显式实验名、输入路径不变、模型输出路径和日志路径包含实验名
- **AND** shell 语法检查 SHALL 覆盖被修改的串行训练 shell
