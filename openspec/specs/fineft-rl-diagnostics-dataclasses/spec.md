# fineft-rl-diagnostics-dataclasses Specification

## Purpose
TBD - created by archiving change refactor-rl-diagnostics-dataclasses. Update Purpose after archive.
## Requirements
### Requirement: Loss NaN diagnostics SHALL expose dataclass result objects
系统 SHALL 在 `FineFT/RL/DiHFT/low_level/loss_nan_diagnostics.py` 中使用 dataclass 对象表达 loss NaN 诊断结果，并让调用方通过对象属性访问诊断字段。

#### Scenario: Numeric summaries use attributes and preserve legacy dict shape
- **WHEN** `build_loss_nan_diagnostics(numeric_values, info_values)` 汇总 torch tensor、numpy array、list、tuple 或标量数值
- **THEN** 函数 SHALL 返回 `LossNanDiagnostics`
- **AND** `LossNanDiagnostics.numeric` SHALL map input names to `NumericValueSummary` objects or `None`
- **AND** `NumericValueSummary` SHALL expose `shape`, `dtype`, `finite_count`, `nan_count`, `inf_count`, `first_nonfinite_indices`, `finite_min`, `finite_max`, and `finite_mean` attributes
- **AND** `LossNanDiagnostics.to_dict()` SHALL preserve the legacy `{"numeric": ..., "info_nonfinite": ...}` structure

#### Scenario: Info nonfinite locations use dataclass records
- **WHEN** `build_loss_nan_diagnostics(...)` finds nonfinite values inside nested `info_values`
- **THEN** `LossNanDiagnostics.info_nonfinite` SHALL contain `NonfiniteLocation` objects
- **AND** each `NonfiniteLocation` SHALL expose `path`, `shape`, `dtype`, `nan_count`, `inf_count`, and `first_nonfinite_indices`
- **AND** recursive traversal SHALL keep the existing path format such as `info.q_value` and `info.funding_count_down_hour`
- **AND** traversal SHALL continue respecting the existing `max_items` cap

#### Scenario: Loss NaN logging uses dataclass attributes
- **WHEN** `log_loss_nan_diagnostics(logger, numeric_values, info_values, trainer)` logs a NaN loss
- **THEN** it SHALL read `LossNanDiagnostics` through attributes instead of string-key dict access
- **AND** log messages SHALL retain the existing trainer metadata and nonfinite-value information
- **AND** absence of nonfinite info values SHALL still log the existing "no nonfinite values found in info" message

### Requirement: Pretrain qtable diagnostics SHALL expose dataclass contracts and preserve files
系统 SHALL 在 `FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py` 中使用 dataclass 对象表达 qtable 诊断计划、manifest、CSV row、sample 诊断和 prepare 返回值，同时保持现有落盘 JSON/CSV 格式兼容。

#### Scenario: Sample plan and cache access use sample item objects
- **WHEN** `build_sample_plan(total_df_index_length, position_choices)` builds the qtable diagnostics plan
- **THEN** it SHALL return `list[SamplePlanItem]`
- **AND** each `SamplePlanItem` SHALL expose `df_index` and `initial_action`
- **AND** `select_sample_from_plan(sample_plan)` SHALL return a `SamplePlanItem`
- **AND** `get_sample_action_from_cache(...)` SHALL locate cached actions by `SamplePlanItem` or by equivalent explicit `df_index` and `initial_action`
- **AND** a missing cache entry SHALL still raise `KeyError` containing `df_index` and `initial_action`

#### Scenario: Manifest object writes the current JSON contract
- **WHEN** qtable diagnostics writes `manifest.json`
- **THEN** the manifest SHALL be represented by `QTableDiagnosticsManifest`
- **AND** `QTableDiagnosticsManifest.to_dict()` SHALL include `diagnostic_count`, `total_df_index_length`, `position_choices`, `qtable_kwargs`, and `env_kwargs`
- **AND** JSON normalization SHALL continue converting numpy arrays, numpy scalars, dict keys, lists, and tuples into JSON-compatible values
- **AND** existing manifest match behavior SHALL remain unchanged for valid JSON, unreadable JSON, missing files, and mismatched manifests

#### Scenario: CSV rows and sample diagnostics use dataclass records
- **WHEN** `evaluate_and_export_sample(...)` replays a DP action path
- **THEN** each row SHALL be represented by `DiagnosticCsvRow`
- **AND** `DiagnosticCsvRow.to_dict()` SHALL write the existing CSV columns and values without adding or removing columns
- **AND** `evaluate_and_export_sample(...)` SHALL return `SampleDiagnostic`
- **AND** `SampleDiagnostic` SHALL expose `df_index`, `initial_action`, `episode_reward_sum`, `profitable`, `csv_path`, and `action_list`
- **AND** CSV filename and directory behavior SHALL remain unchanged

#### Scenario: Prepare result uses a dataclass return value
- **WHEN** `prepare_pretrain_qtable_diagnostics(...)` completes from recomputation or existing CSV cache
- **THEN** it SHALL return `PretrainQTableDiagnosticsResult`
- **AND** the result SHALL expose `sample_plan`, `q_table_cache`, `train_df_cache`, `diagnostics`, and `sample_action_cache`
- **AND** `weight_advantage_pretrain.py` and `parallel_weight_advantage_pretrain.py` SHALL access the result through attributes
- **AND** existing CSV cache reuse SHALL still load old CSV files when the manifest matches
- **AND** existing invalid-cache conditions SHALL still trigger recomputation

### Requirement: Parallel rollout training SHALL use dataclass worker and metrics contracts
系统 SHALL 在 `FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py` 中使用 dataclass 对象表达 parallel rollout 任务、worker 消息、worker 结果、metrics 和 round summary。

#### Scenario: Rollout planning and epoch parameters use dataclass objects
- **WHEN** `iter_parallel_rollout_tasks(num_epoch, context_count, position_choices)` yields rollout work
- **THEN** each yielded item SHALL be a `ParallelRolloutTask`
- **AND** `ParallelRolloutTask` SHALL expose `epoch_index`, `context_index`, and `initial_action`
- **WHEN** `compute_epoch_training_params(...)` computes scheduled training parameters
- **THEN** it SHALL return `EpochTrainingParams`
- **AND** the result SHALL expose `epsilon`, `ada`, and `lr`
- **AND** `EpochTrainingParams.to_dict()` SHALL preserve the legacy dict shape for compatibility assertions

#### Scenario: Worker queue payloads and results use top-level dataclasses
- **WHEN** the main process sends reset, explore, or shutdown messages to rollout workers
- **THEN** queue payloads SHALL use module-level dataclass objects that are pickle-compatible
- **AND** `DfRolloutWorkerRunner.reset_task(...)` and `explore_round(...)` SHALL read payload fields through attributes
- **WHEN** a worker completes an exploration round
- **THEN** it SHALL return `WorkerRoundResult`
- **AND** transition entries SHALL be represented by `WorkerTransitionRecord`
- **AND** sorting and buffer writes SHALL continue ordering by `df_index` then `step_index`

#### Scenario: Worker errors remain fail-fast and informative
- **WHEN** a rollout worker raises an exception
- **THEN** it SHALL put a `WorkerErrorMessage` onto the result queue
- **AND** `raise_for_worker_error(...)` SHALL recognize that object and raise `RuntimeError`
- **AND** the error message SHALL include `df_index`, `epoch_index`, `context_index`, `initial_action`, `round_counter`, and traceback text
- **AND** unknown worker messages SHALL still raise `ValueError`

#### Scenario: Rollout metrics and summaries use dataclass objects
- **WHEN** rollout metrics are recorded or summarized
- **THEN** individual metrics SHALL be represented by `RolloutMetrics`
- **AND** aggregate metrics SHALL be represented by `RolloutMetricsSummary`
- **AND** rollout diagnostics SHALL be represented by `RolloutDiagnosticsSummary`
- **AND** parallel round summaries SHALL be represented by `ParallelRoundSummary`
- **AND** `.to_dict()` on these summary objects SHALL preserve legacy dict shapes used by logs or compatibility tests

### Requirement: Dataclass diagnostics refactor SHALL be verified by focused tests
系统 SHALL 用 focused tests 和 lightweight compile checks 验证对象接口、落盘兼容性、worker 边界和 OpenSpec 规格。

#### Scenario: Focused tests cover object interfaces and compatibility
- **WHEN** this change is implemented
- **THEN** `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py` SHALL assert loss NaN diagnostics through dataclass attributes and legacy-compatible `.to_dict()`
- **AND** `FineFT/tests/rl/test_pretrain_qtable_diagnostics.py` SHALL assert qtable diagnostics dataclass returns, attribute access, manifest JSON compatibility, CSV compatibility, cache reuse, and invalid-cache recomputation
- **AND** `FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py` SHALL assert dataclass rollout tasks, epoch params, worker messages/results/errors, metrics, transition ordering, and round summaries

#### Scenario: Verification commands pass
- **WHEN** implementation is complete
- **THEN** `conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py FineFT/tests/rl/test_pretrain_qtable_diagnostics.py FineFT/tests/rl/test_parallel_weight_advantage_pretrain.py -q` SHALL pass
- **AND** `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/loss_nan_diagnostics.py FineFT/RL/DiHFT/low_level/pretrain_qtable_diagnostics.py FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py` SHALL pass
- **AND** `openspec validate refactor-rl-diagnostics-dataclasses --strict` SHALL pass

