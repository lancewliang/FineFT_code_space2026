# Select Cross Contract Low Level Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean cross-contract low-level agent selection path where each pure `label_i` maps to one reusable low-level qnet.

**Architecture:** `test_agent_index.py` owns validation slice discovery and emits the new cross-contract result schema. `FineFT_single_agent_with_different_position.py` owns schema validation, preserves the current two-stage selection algorithm, assembles the selected qnets in label order, and writes an audit manifest.

**Tech Stack:** Python, pandas, NumPy, PyTorch, pytest, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/select-cross-contract-low-level-agents/plan-ready.md`
- tasks: `openspec/changes/select-cross-contract-low-level-agents/tasks.md`
- plan: `docs/superpowers/plans/2026-07-21-select-cross-contract-low-level-agents.md`

---

### Task 1: Update low-level test-agent discovery tests and helper fixtures to model `valid/<contract>/<label>/df_*.feather`.

> **trace:** plan-ready.md → `### Task 1: Update low-level test-agent discovery tests and helper fixtures to model `valid/<contract>/<label>/df_*.feather`.` | tasks.md → `- [ ] 1.1 Update low-level test-agent discovery tests and helper fixtures to model `valid/<contract>/<label>/df_*.feather`.`
> **sync:** tasks.md → `- [ ] 1.1 Update low-level test-agent discovery tests and helper fixtures to model `valid/<contract>/<label>/df_*.feather`.` | plan-ready.md → `### Task 1: Update low-level test-agent discovery tests and helper fixtures to model `valid/<contract>/<label>/df_*.feather`.`

**Files:**
- Modify: `FineFT/tests/rl/test_test_agent_index.py`

- [ ] **Step 1: Add a helper that writes commodity valid label slices**

In `FineFT/tests/rl/test_test_agent_index.py`, add this helper below `FakeEnsemble`:

```python
def _write_valid_slice(tmp_path, contract, label, filename="df_0.feather", mark_prices=None):
    if mark_prices is None:
        mark_prices = [100.0]
    valid_dir = tmp_path / "valid" / contract / label
    valid_dir.mkdir(parents=True, exist_ok=True)
    df_path = valid_dir / filename
    pd.DataFrame({"mark_price": mark_prices}).to_feather(df_path)
    return df_path
```

- [ ] **Step 2: Update `_make_test_trader` to use the new nested fixture**

Replace the current fixture setup inside `_make_test_trader(...)` with:

```python
    _write_valid_slice(tmp_path, "fu2507", "label_0")
```

Keep the rest of `_make_test_trader(...)` unchanged.

- [ ] **Step 3: Update aggregate assertions to expect the new schema**

In `test_weighted_trader_passes_order_book_depth_to_base_env`, replace the npy and CSV assertions with:

```python
    result = np.load(npy_path, allow_pickle=True).tolist()
    assert result[0]["label"] == "label_0"
    assert result[0]["contract"] == ["fu2507"]
    assert result[0]["df_path"] == ["fu2507/label_0/df_0.feather"]

    csv_df = pd.read_csv(csv_path)
    assert list(csv_df.columns) == [
        "标签",
        "初始动作",
        "分箱索引",
        "合约",
        "数据文件",
        "奖励总和",
        "数据长度",
        "换手率",
    ]
    assert json.loads(csv_df.loc[0, "合约"]) == ["fu2507"]
    assert json.loads(csv_df.loc[0, "数据文件"]) == ["fu2507/label_0/df_0.feather"]
    assert json.loads(csv_df.loc[0, "奖励总和"]) == [1.0]
    assert json.loads(csv_df.loc[0, "数据长度"]) == [1]
    assert json.loads(csv_df.loc[0, "换手率"]) == [0.0]
```

- [ ] **Step 4: Update the nested contract-label test expectation**

In `test_weighted_trader_handles_nested_contract_label_directories`, replace the setup and assertions with:

```python
    _write_valid_slice(tmp_path, "fu2507", "label_2")
    (tmp_path / "valid" / "processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"mark_price": [100.0]}).to_feather(
        tmp_path / "valid" / "processed" / "valid_processed_fu2507.feather"
    )
    pd.DataFrame({"mark_price": [100.0]}).to_feather(
        tmp_path / "valid" / "fu2507.feather"
    )

    monkeypatch.setattr(tai, "initiate_base_env", lambda **kwargs: FakeEnv())
    monkeypatch.setattr(tai, "map_action_to_position_leverage", lambda *args: (0, 1))

    trader = _make_test_trader(tai, tmp_path)
    trader.test()

    result = np.load(tmp_path / "analysis_result.npy", allow_pickle=True).tolist()
    assert result[0]["label"] == "label_0"
    assert result[0]["contract"] == ["fu2507"]
    assert result[0]["df_path"] == ["fu2507/label_0/df_0.feather"]
    label_2_records = [row for row in result if row["label"] == "label_2"]
    assert label_2_records[0]["contract"] == ["fu2507"]
    assert label_2_records[0]["df_path"] == ["fu2507/label_2/df_0.feather"]
```

- [ ] **Step 5: Run the focused test to verify the RED state**

Run:

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py::test_weighted_trader_handles_nested_contract_label_directories -q
```

Expected: FAIL because `result[0]["contract"]` is missing or `label` still contains a contract path.

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Refactor `test_agent_index.py` validation file discovery and aggregate result construction to emit pure `label`, aligned `contract`, and contract-relative `df_path` arrays.

> **trace:** plan-ready.md → `### Task 2: Refactor `test_agent_index.py` validation file discovery and aggregate result construction to emit pure `label`, aligned `contract`, and contract-relative `df_path` arrays.` | tasks.md → `- [ ] 1.2 Refactor `test_agent_index.py` validation file discovery and aggregate result construction to emit pure `label`, aligned `contract`, and contract-relative `df_path` arrays.`
> **sync:** tasks.md → `- [ ] 1.2 Refactor `test_agent_index.py` validation file discovery and aggregate result construction to emit pure `label`, aligned `contract`, and contract-relative `df_path` arrays.` | plan-ready.md → `### Task 2: Refactor `test_agent_index.py` validation file discovery and aggregate result construction to emit pure `label`, aligned `contract`, and contract-relative `df_path` arrays.`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/test_agent_index.py`

- [ ] **Step 1: Add a label pattern import and constant**

Add `import re` near the existing imports and add this constant near `AGGREGATE_JSON_COLUMNS`:

```python
LABEL_DIR_PATTERN = re.compile(r"^label_\d+$")
```

- [ ] **Step 2: Replace `_iter_valid_feather_files` with strict commodity discovery**

Replace the existing `_iter_valid_feather_files(root_dir)` with:

```python
def _iter_valid_feather_files(root_dir):
    entries = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"valid data path does not exist: {root_dir}")

    for contract in sorted(os.listdir(root_dir)):
        contract_dir = os.path.join(root_dir, contract)
        if contract == "processed" or not os.path.isdir(contract_dir):
            continue
        for label in sorted(os.listdir(contract_dir)):
            label_dir = os.path.join(contract_dir, label)
            if not os.path.isdir(label_dir) or not LABEL_DIR_PATTERN.fullmatch(label):
                continue
            for filename in sorted(os.listdir(label_dir)):
                if filename.startswith("df_") and filename.endswith(".feather"):
                    rel_path = os.path.join(contract, label, filename)
                    entries.append(
                        {
                            "contract": contract,
                            "label": label,
                            "df_path": rel_path,
                            "abs_path": os.path.join(root_dir, rel_path),
                        }
                    )

    if not entries:
        raise FileNotFoundError(
            f"no validation label slices found under {root_dir}; expected "
            "valid/<contract>/label_*/df_*.feather"
        )
    return entries
```

- [ ] **Step 3: Update `weighted_trader.test()` to group by pure label**

In `weighted_trader.test()`, replace:

```python
        df_entries = list(_iter_valid_feather_files(self.valid_data_path))
        label_list = sorted({label for label, _ in df_entries})
```

with:

```python
        df_entries = _iter_valid_feather_files(self.valid_data_path)
        label_list = sorted({entry["label"] for entry in df_entries})
```

Then replace the `df_list` block with:

```python
            label_entries = [
                entry for entry in df_entries if entry["label"] == label
            ]
```

And replace `for df_path in df_list:` with:

```python
                    single_label_initial_action_bin_index_contract_result = []
                    for entry in label_entries:
                        contract = entry["contract"]
                        df_path = entry["df_path"]
```

- [ ] **Step 4: Read feather files from `abs_path` and append contract metadata**

Inside the slice loop, replace the dataframe read with:

```python
                        self.test_df = pd.read_feather(entry["abs_path"])
```

At the end of the slice loop, add the contract append before `df_path` append:

```python
                        single_label_initial_action_bin_index_contract_result.append(
                            contract
                        )
```

Then include `contract` in `_overall_result` before `df_path`:

```python
                            "contract": single_label_initial_action_bin_index_contract_result,
```

- [ ] **Step 5: Run the focused test to verify the GREEN state**

Run:

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py::test_weighted_trader_handles_nested_contract_label_directories -q
```

Expected: PASS.

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Update aggregate CSV serialization and Chinese headers to include `合约` while preserving JSON array cells.

> **trace:** plan-ready.md → `### Task 3: Update aggregate CSV serialization and Chinese headers to include `合约` while preserving JSON array cells.` | tasks.md → `- [ ] 1.3 Update aggregate CSV serialization and Chinese headers to include `合约` while preserving JSON array cells.`
> **sync:** tasks.md → `- [ ] 1.3 Update aggregate CSV serialization and Chinese headers to include `合约` while preserving JSON array cells.` | plan-ready.md → `### Task 3: Update aggregate CSV serialization and Chinese headers to include `合约` while preserving JSON array cells.`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/test_agent_index.py`
- Modify: `FineFT/tests/rl/test_test_agent_index.py`

- [ ] **Step 1: Include `contract` in aggregate JSON columns**

In `FineFT/RL/DiHFT/low_level/test_agent_index.py`, replace:

```python
AGGREGATE_JSON_COLUMNS = ["df_path", "reward_sum", "df_length", "turnover"]
```

with:

```python
AGGREGATE_JSON_COLUMNS = ["contract", "df_path", "reward_sum", "df_length", "turnover"]
```

- [ ] **Step 2: Add the Chinese header label**

In `CSV_HEADER_LABELS`, add:

```python
    "contract": "合约",
```

immediately after:

```python
    "bin_index": "分箱索引",
```

- [ ] **Step 3: Run all low-level test-agent tests**

Run:

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py -q
```

Expected: PASS.

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 4: Add picker schema-validation and sample-equal scoring tests for the new cross-contract result schema and legacy schema rejection.

> **trace:** plan-ready.md → `### Task 4: Add picker schema-validation and sample-equal scoring tests for the new cross-contract result schema and legacy schema rejection.` | tasks.md → `- [ ] 1.4 Add picker schema-validation and sample-equal scoring tests for the new cross-contract result schema and legacy schema rejection.`
> **sync:** tasks.md → `- [ ] 1.4 Add picker schema-validation and sample-equal scoring tests for the new cross-contract result schema and legacy schema rejection.` | plan-ready.md → `### Task 4: Add picker schema-validation and sample-equal scoring tests for the new cross-contract result schema and legacy schema rejection.`

**Files:**
- Modify: `FineFT/tests/analysis/test_pick_agent.py`

- [ ] **Step 1: Add imports for error assertions and JSON manifest checks**

At the top of `FineFT/tests/analysis/test_pick_agent.py`, add:

```python
import json
import pytest
```

- [ ] **Step 2: Add a cross-contract result factory**

Add this helper below `_picker(tmp_path)`:

```python
def _cross_contract_record(label, initial_action, bin_index, rewards, lengths):
    contracts = [f"fu25{i:02d}" for i in range(len(rewards))]
    return {
        "label": label,
        "initial_action": initial_action,
        "bin_index": bin_index,
        "contract": contracts,
        "df_path": [
            f"{contract}/{label}/df_{index}.feather"
            for index, contract in enumerate(contracts)
        ],
        "reward_sum": rewards,
        "df_length": lengths,
        "turnover": [0.0 for _ in rewards],
    }
```

- [ ] **Step 3: Add the sample-equal transform test**

Add this test:

```python
def test_transform_single_epoch_result_uses_sample_equal_cross_contract_rewards(tmp_path):
    p = _picker(tmp_path)
    result = [
        _cross_contract_record(
            "label_0",
            initial_action=0,
            bin_index=1,
            rewards=[10.0, 6.0],
            lengths=[5, 3],
        )
    ]

    transformed = p.transform_single_epoch_result(result, "epoch_1")

    assert transformed[0]["normalized_reward"].tolist() == [2.0, 2.0]
    assert transformed[0]["trans_reward_mean"] == 2.0
    assert transformed[0]["trans_reward_std"] == 0.0
```

- [ ] **Step 4: Add legacy schema rejection and label coverage tests**

Add these tests:

```python
def test_transform_single_epoch_result_rejects_legacy_contract_label_schema(tmp_path):
    p = _picker(tmp_path)
    legacy = [
        {
            "label": "fu2409/label_0",
            "initial_action": 0,
            "bin_index": 1,
            "df_path": ["df_0.feather"],
            "reward_sum": [1.0],
            "df_length": [1],
            "turnover": [0.0],
        }
    ]

    with pytest.raises(ValueError, match="rerun test_agent_index.py"):
        p.transform_single_epoch_result(legacy, "epoch_1")


def test_picker_rejects_label_set_mismatch(tmp_path):
    p = _picker(tmp_path)
    result_all = pd.DataFrame(
        [
            {
                "label": "label_0",
                "bin_index": 1,
                "epoch_path": "epoch_1",
                "trans_reward_mean": 0.2,
            }
        ]
    )

    with pytest.raises(ValueError, match="label_1"):
        p.pick_best_agent_regarding_dynamics_bin_index_path(result_all)
```

- [ ] **Step 5: Add final aggregation and manifest tests**

Add these tests:

```python
def test_final_selection_keeps_current_result_all_initial_action_mean(tmp_path):
    p = _picker(tmp_path)
    rows = []
    for initial_action, score in [(0, 1.0), (1, 3.0)]:
        rows.append(
            {
                "label": "label_0",
                "bin_index": 0,
                "epoch_path": "epoch_1",
                "initial_action": initial_action,
                "trans_reward_mean": score,
            }
        )
    for initial_action, score in [(0, 1.9), (1, 1.9)]:
        rows.append(
            {
                "label": "label_0",
                "bin_index": 1,
                "epoch_path": "epoch_2",
                "initial_action": initial_action,
                "trans_reward_mean": score,
            }
        )
    for label_index in range(1, 5):
        rows.append(
            {
                "label": f"label_{label_index}",
                "bin_index": label_index,
                "epoch_path": f"epoch_{label_index}",
                "initial_action": 0,
                "trans_reward_mean": 0.1,
            }
        )

    best = p.pick_best_agent_regarding_dynamics_bin_index_path(pd.DataFrame(rows))

    label_0 = best[best["label"] == "label_0"].iloc[0]
    assert label_0["bin_index"] == 0
    assert label_0["epoch_path"] == "epoch_1"
    assert label_0["reward_max"] == 2.0


def test_write_selection_manifest_records_label_choices(tmp_path):
    p = _picker(tmp_path)
    best = pd.DataFrame(
        [
            {
                "label": f"label_{index}",
                "epoch_path": f"epoch_{index}",
                "bin_index": index,
                "reward_max": float(index),
                "source_rows": index + 1,
            }
            for index in range(5)
        ]
    )

    manifest_path = p.write_selection_manifest(best)

    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["dataset_name"] == "fu"
    assert manifest["experiment_name"] == "10min_nstep6_costw5"
    assert manifest["selection_method"] == "sample_equal_current_picker_logic"
    assert manifest["labels"][0]["label"] == "label_0"
    assert manifest["labels"][0]["model_path"] == "epoch_0/trained_model.pkl"
```

- [ ] **Step 6: Run picker tests to verify the RED state**

Run:

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/analysis/test_pick_agent.py -q
```

Expected: FAIL because picker does not yet validate schema, enforce label coverage, include `source_rows`, or provide `write_selection_manifest`.

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 5: Refactor `FineFT_single_agent_with_different_position.py` to validate new schema, preserve current first-stage and final-stage selection logic, and fail fast on invalid labels or metrics.

> **trace:** plan-ready.md → `### Task 5: Refactor `FineFT_single_agent_with_different_position.py` to validate new schema, preserve current first-stage and final-stage selection logic, and fail fast on invalid labels or metrics.` | tasks.md → `- [ ] 1.5 Refactor `FineFT_single_agent_with_different_position.py` to validate new schema, preserve current first-stage and final-stage selection logic, and fail fast on invalid labels or metrics.`
> **sync:** tasks.md → `- [ ] 1.5 Refactor `FineFT_single_agent_with_different_position.py` to validate new schema, preserve current first-stage and final-stage selection logic, and fail fast on invalid labels or metrics.` | plan-ready.md → `### Task 5: Refactor `FineFT_single_agent_with_different_position.py` to validate new schema, preserve current first-stage and final-stage selection logic, and fail fast on invalid labels or metrics.`

**Files:**
- Modify: `FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`

- [ ] **Step 1: Add imports and validation constants**

Add near the top:

```python
import re
```

Add after parser arguments:

```python
LABEL_PATTERN = re.compile(r"^label_(\d+)$")
ARRAY_FIELDS = ["contract", "df_path", "reward_sum", "df_length", "turnover"]
REQUIRED_RESULT_FIELDS = ["label", "initial_action", "bin_index"] + ARRAY_FIELDS
```

- [ ] **Step 2: Add record validation helpers inside `picker`**

Add these methods to `class picker` before `transform_single_epoch_result`:

```python
    def _result_output_dir(self):
        return os.path.join(self.save_path, self.dataset_name, self.experiment_name)

    def _label_sort_key(self, label):
        match = LABEL_PATTERN.fullmatch(str(label))
        if not match:
            raise ValueError(f"invalid label {label}; expected label_<integer>")
        return int(match.group(1))

    def _expected_label_set(self):
        return set(self.label_list)

    def _validate_result_record(self, single_result):
        missing_fields = [
            field for field in REQUIRED_RESULT_FIELDS if field not in single_result
        ]
        if missing_fields:
            raise ValueError(
                f"analysis_result record missing fields {missing_fields}; "
                "rerun test_agent_index.py to generate the new schema"
            )
        label = single_result["label"]
        if "/" in str(label) or "\\" in str(label):
            raise ValueError(
                f"legacy label schema {label}; rerun test_agent_index.py "
                "to generate the new schema"
            )
        self._label_sort_key(label)
        lengths = {field: len(single_result[field]) for field in ARRAY_FIELDS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"aligned array fields have mismatched lengths: {lengths}")
        if lengths["reward_sum"] == 0:
            raise ValueError(f"record for {label} has no validation samples")
        reward_sum = np.asarray(single_result["reward_sum"], dtype=float)
        df_length = np.asarray(single_result["df_length"], dtype=float)
        if np.any(df_length <= 0):
            raise ValueError(f"record for {label} has non-positive df_length")
        if not np.all(np.isfinite(reward_sum)):
            raise ValueError(f"record for {label} has non-finite reward_sum")
        if len(set(single_result["df_path"])) != len(single_result["df_path"]):
            raise ValueError(f"record for {label} has duplicate df_path values")

    def _validate_label_coverage(self, labels):
        actual = set(labels)
        expected = self._expected_label_set()
        if actual != expected:
            missing = sorted(expected - actual, key=self._label_sort_key)
            extra = sorted(actual - expected, key=self._label_sort_key)
            raise ValueError(f"label coverage mismatch; missing={missing}, extra={extra}")
```

- [ ] **Step 3: Validate and copy records before computing metrics**

In `transform_single_epoch_result`, replace the first lines inside the loop with:

```python
            single_result = dict(single_result)
            self._validate_result_record(single_result)
            reward_sum = np.asarray(single_result["reward_sum"], dtype=float)
            df_length = np.asarray(single_result["df_length"], dtype=float)
            single_result["normalized_reward"] = reward_sum / df_length
```

Keep the existing `trans_reward_mean`, `trans_reward_std`, `mean_turnover`, and `epoch_path` assignments.

- [ ] **Step 4: Preserve final-stage logic while adding coverage and source row counts**

In `pick_best_agent_regarding_dynamics_bin_index_path`, add at the beginning:

```python
        self._validate_label_coverage(result_all["label"].unique())
```

Replace the grouped calculation block with:

```python
            reward_mean_info = (
                selected_df.groupby(["label", "bin_index", "epoch_path"])[
                    "trans_reward_mean"
                ]
                .agg(["mean", "count"])
                .dropna()
            )
            if reward_mean_info.empty:
                continue
            selected_information_based_reward_sum = reward_mean_info["mean"].idxmax()
            label = selected_information_based_reward_sum[0]
            bin_index = selected_information_based_reward_sum[1]
            epoch_path = selected_information_based_reward_sum[2]
            reward_max = reward_mean_info.loc[
                selected_information_based_reward_sum, "mean"
            ]
            source_rows = int(
                reward_mean_info.loc[selected_information_based_reward_sum, "count"]
            )
```

Add `source_rows_list = []` next to the other output lists, append `source_rows`, and include `"source_rows": source_rows_list` in `best_agent_info`.

- [ ] **Step 5: Run picker tests for validation and selection**

Run:

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/analysis/test_pick_agent.py::test_transform_single_epoch_result_uses_sample_equal_cross_contract_rewards FineFT/tests/analysis/test_pick_agent.py::test_transform_single_epoch_result_rejects_legacy_contract_label_schema FineFT/tests/analysis/test_pick_agent.py::test_picker_rejects_label_set_mismatch FineFT/tests/analysis/test_pick_agent.py::test_final_selection_keeps_current_result_all_initial_action_mean -q
```

Expected: PASS for the validation and selection tests, FAIL only if manifest is still missing.

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 6: Add `selection_manifest.json` output and enforce label-order model assembly for `model.pth`.

> **trace:** plan-ready.md → `### Task 6: Add `selection_manifest.json` output and enforce label-order model assembly for `model.pth`.` | tasks.md → `- [ ] 1.6 Add `selection_manifest.json` output and enforce label-order model assembly for `model.pth`.`
> **sync:** tasks.md → `- [ ] 1.6 Add `selection_manifest.json` output and enforce label-order model assembly for `model.pth`.` | plan-ready.md → `### Task 6: Add `selection_manifest.json` output and enforce label-order model assembly for `model.pth`.`

**Files:**
- Modify: `FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`
- Modify: `FineFT/tests/analysis/test_pick_agent.py`

- [ ] **Step 1: Add JSON import**

At the top of `FineFT_single_agent_with_different_position.py`, add:

```python
import json
```

- [ ] **Step 2: Add final selection validation and sorting helpers**

Add these methods inside `class picker` before `create_potential_result`:

```python
    def _ordered_best_agent_df(self, best_agent_df):
        labels = best_agent_df["label"].tolist()
        self._validate_label_coverage(labels)
        if best_agent_df["label"].duplicated().any():
            raise ValueError("each label must have exactly one selected agent")
        ordered_df = best_agent_df.copy()
        ordered_df["_label_index"] = ordered_df["label"].apply(self._label_sort_key)
        ordered_df = ordered_df.sort_values("_label_index").drop(columns="_label_index")
        return ordered_df

    def write_selection_manifest(self, best_agent_df):
        ordered_df = self._ordered_best_agent_df(best_agent_df)
        output_dir = self._result_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        labels = []
        for row in ordered_df.to_dict("records"):
            model_path = os.path.join(row["epoch_path"], "trained_model.pkl")
            labels.append(
                {
                    "label": row["label"],
                    "epoch_path": row["epoch_path"],
                    "model_path": model_path,
                    "bin_index": int(row["bin_index"]),
                    "score": float(row["reward_max"]),
                    "source_rows": int(row.get("source_rows", 0)),
                }
            )
        manifest = {
            "dataset_name": self.dataset_name,
            "experiment_name": self.experiment_name,
            "selection_method": "sample_equal_current_picker_logic",
            "labels": labels,
        }
        manifest_path = os.path.join(output_dir, "selection_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
        return manifest_path
```

- [ ] **Step 3: Sort labels before model assembly and write manifest after save**

At the start of `create_potential_result`, add:

```python
        best_agent_df = self._ordered_best_agent_df(best_agent_df)
```

After `torch.save(...)`, add:

```python
        self.write_selection_manifest(best_agent_df)
```

- [ ] **Step 4: Run picker tests**

Run:

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/analysis/test_pick_agent.py -q
```

Expected: PASS.

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 7: Update commodity low-level test and picker shell scripts so `fu` runs pass the required base path, experiment name, position choices, and label count.

> **trace:** plan-ready.md → `### Task 7: Update commodity low-level test and picker shell scripts so `fu` runs pass the required base path, experiment name, position choices, and label count.` | tasks.md → `- [ ] 1.7 Update commodity low-level test and picker shell scripts so `fu` runs pass the required base path, experiment name, position choices, and label count.`
> **sync:** tasks.md → `- [ ] 1.7 Update commodity low-level test and picker shell scripts so `fu` runs pass the required base path, experiment name, position choices, and label count.` | plan-ready.md → `### Task 7: Update commodity low-level test and picker shell scripts so `fu` runs pass the required base path, experiment name, position choices, and label count.`

**Files:**
- Modify: `FineFT/script/test/DiHFT/low_level/test_util_fu.sh`
- Modify: `FineFT/script/analysis/pick_agent/low_level_fu.sh`

- [ ] **Step 1: Parameterize `test_util_fu.sh` active run**

At the bottom of `FineFT/script/test/DiHFT/low_level/test_util_fu.sh`, replace the active `run_ddqn_context fu ...` call with:

```bash
DATASET_NAME=${DATASET_NAME:-fu}
MAX_HOLDING_NUMBER=${MAX_HOLDING_NUMBER:-1}
EPOCH_START=${EPOCH_START:-1}
EPOCH_END=${EPOCH_END:-60}
BASE_PATH=${BASE_PATH:-dataset/10min}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_nstep6_costw5}

run_ddqn_context "${DATASET_NAME}" "${MAX_HOLDING_NUMBER}" "${EPOCH_START}" "${EPOCH_END}" "${BASE_PATH}" "${EXPERIMENT_NAME}"
```

- [ ] **Step 2: Pass ensemble count to `test_agent_index.py`**

Inside `run_ddqn_context`, add:

```bash
    local ensemble_number=${ENSEMBLE_NUMBER:-5}
```

Then add this argument to the `python FineFT/RL/DiHFT/low_level/test_agent_index.py` command:

```bash
            --N "${ensemble_number}" \
```

- [ ] **Step 3: Parameterize `low_level_fu.sh`**

Replace the hard-coded command in `FineFT/script/analysis/pick_agent/low_level_fu.sh` with:

```bash
DATASET_NAME=${DATASET_NAME:-fu}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-10min_nstep6_costw5}
BASE_PATH=${BASE_PATH:-dataset/10min}
POSITION_CHOICES=${POSITION_CHOICES:-3}
NUM_LABEL=${NUM_LABEL:-5}

nohup python FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py \
    --dataset_name "${DATASET_NAME}" --experiment_name "${EXPERIMENT_NAME}" \
    --base_path "${BASE_PATH}" --position_choices "${POSITION_CHOICES}" --num_label "${NUM_LABEL}" \
    >"log/analysis/pick_agent/DiHFT/${DATASET_NAME}/${EXPERIMENT_NAME}.log" 2>&1 &
```

- [ ] **Step 4: Run shell syntax checks**

Run:

```bash
bash -n FineFT/script/test/DiHFT/low_level/test_util_fu.sh FineFT/script/analysis/pick_agent/low_level_fu.sh
```

Expected: PASS.

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 8: Run `conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/analysis/test_pick_agent.py -q`.

> **trace:** plan-ready.md → `### Task 8: Run `conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/analysis/test_pick_agent.py -q`.` | tasks.md → `- [ ] 2.1 Run `conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/analysis/test_pick_agent.py -q`.`
> **sync:** tasks.md → `- [ ] 2.1 Run `conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/analysis/test_pick_agent.py -q`.` | plan-ready.md → `### Task 8: Run `conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/analysis/test_pick_agent.py -q`.`

**Files:**
- Verify: `FineFT/tests/rl/test_test_agent_index.py`
- Verify: `FineFT/tests/analysis/test_pick_agent.py`

- [ ] **Step 1: Run focused pytest**

Run:

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && PYTHONPATH=FineFT pytest FineFT/tests/rl/test_test_agent_index.py FineFT/tests/analysis/test_pick_agent.py -q
```

Expected: PASS with all tests in both files passing.

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 9: Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`.

> **trace:** plan-ready.md → `### Task 9: Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`.` | tasks.md → `- [ ] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`.`
> **sync:** tasks.md → `- [ ] 2.2 Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`.` | plan-ready.md → `### Task 9: Run `conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`.`

**Files:**
- Verify: `FineFT/RL/DiHFT/low_level/test_agent_index.py`
- Verify: `FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py`

- [ ] **Step 1: Run py_compile**

Run:

```bash
source /home/lanceliang/miniconda3/etc/profile.d/conda.sh && conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/test_agent_index.py FineFT/analysis/pick_agent/FineFT_single_agent_with_different_position.py
```

Expected: command exits with status 0 and prints no traceback.

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 10: Run `openspec validate select-cross-contract-low-level-agents --strict`.

> **trace:** plan-ready.md → `### Task 10: Run `openspec validate select-cross-contract-low-level-agents --strict`.` | tasks.md → `- [ ] 2.3 Run `openspec validate select-cross-contract-low-level-agents --strict`.`
> **sync:** tasks.md → `- [ ] 2.3 Run `openspec validate select-cross-contract-low-level-agents --strict`.` | plan-ready.md → `### Task 10: Run `openspec validate select-cross-contract-low-level-agents --strict`.`

**Files:**
- Verify: `openspec/changes/select-cross-contract-low-level-agents/`

- [ ] **Step 1: Run strict OpenSpec validation**

Run:

```bash
openspec validate select-cross-contract-low-level-agents --strict
```

Expected: `Change 'select-cross-contract-low-level-agents' is valid`.

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

