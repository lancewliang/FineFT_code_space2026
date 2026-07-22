# fineft-low-level-agent-selection Specification

## Purpose
TBD - created by archiving change select-cross-contract-low-level-agents. Update Purpose after archive.
## Requirements
### Requirement: Low-level picker SHALL consume only the cross-contract label result schema
系统 SHALL 在 `FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py` 中只消费新低层测试 schema，并拒绝旧的合约路径混入 label 的结果。

#### Scenario: Picker accepts pure label records with aligned cross-contract arrays
- **WHEN** picker loads an epoch `analysis_result.npy`
- **THEN** every result record SHALL contain `label`, `initial_action`, `bin_index`, `contract`, `df_path`, `reward_sum`, `df_length`, and `turnover`
- **AND** `label` SHALL match `label_<integer>`
- **AND** `contract`, `df_path`, `reward_sum`, `df_length`, and `turnover` SHALL have equal lengths
- **AND** `df_path` entries SHALL be contract-relative paths containing the contract and label directory

#### Scenario: Picker rejects legacy label path schema
- **WHEN** picker loads an epoch result where `label` contains a path separator, such as `fu2409/label_0`
- **THEN** picker SHALL fail fast
- **AND** the error message SHALL state that `test_agent_index.py` must be rerun to generate the new schema

#### Scenario: Picker validates label coverage against num_label
- **WHEN** picker finishes loading candidate records
- **THEN** the discovered pure label set SHALL exactly match `label_0` through `label_<num_label-1>`
- **AND** a missing or extra label SHALL fail fast before `model.pth` is written

### Requirement: Low-level picker SHALL preserve the current two-stage selection algorithm
系统 SHALL 保留当前 low-level picker 的两阶段选择算法，只将候选数据来源改为跨合约 label slice。

#### Scenario: First stage uses sample-equal normalized reward
- **WHEN** picker transforms one epoch result record
- **THEN** it SHALL compute `normalized_reward` as elementwise `reward_sum / df_length`
- **AND** each validation slice SHALL contribute one sample regardless of its contract
- **AND** `trans_reward_mean` SHALL be `mean(normalized_reward)`
- **AND** `trans_reward_std` SHALL be `std(normalized_reward)`
- **AND** invalid `df_length <= 0` or non-finite reward values SHALL fail fast

#### Scenario: First stage chooses the best bin per label and initial action
- **WHEN** picker evaluates one epoch
- **THEN** for each `(label, initial_action)` it SHALL choose the candidate `bin_index` maximizing `trans_reward_mean - std_preference * trans_reward_std`
- **AND** the selected rows SHALL still be written to `result.csv`
- **AND** all transformed candidate rows SHALL still be written to `result_all.csv`

#### Scenario: Final stage uses the current result_all aggregation logic
- **WHEN** picker selects the final agent for each label
- **THEN** it SHALL use `result_all`, not `result.csv`
- **AND** it SHALL group by `label`, `bin_index`, and `epoch_path`
- **AND** it SHALL average `trans_reward_mean` across different `initial_action` values
- **AND** it SHALL select the group with the maximum average `trans_reward_mean`
- **AND** it SHALL NOT add a second-stage standard-deviation penalty

### Requirement: Low-level picker SHALL output a potential model and selection manifest
系统 SHALL 为每个 label 组装后续路由可用的低层 potential model，并输出可审计的选择 manifest。

#### Scenario: Potential model keeps label order
- **WHEN** picker finishes final selection
- **THEN** it SHALL create `result/DiHFT/potential_model/<dataset>/<experiment>/model.pth`
- **AND** the qnet at index `i` SHALL correspond to `label_i`
- **AND** each selected qnet SHALL be loaded from the selected `epoch_path/trained_model.pkl` and selected `bin_index`

#### Scenario: Selection manifest records final choices
- **WHEN** picker writes the potential model
- **THEN** it SHALL write `selection_manifest.json` under `analysis_result/DiHFT/low_level/<dataset>/<experiment>/`
- **AND** the manifest SHALL include `dataset_name`, `experiment_name`, `selection_method`, and one entry per label
- **AND** each label entry SHALL include `label`, `epoch_path`, `model_path`, `bin_index`, `score`, and `source_rows`
- **AND** the manifest SHALL be written only after every label has a unique final selection

