# Training Experiment Name Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `experiment_name` output namespace to serial Stage I training so repeated runs for the same dataset can save models, TensorBoard logs, qtable diagnostics, and file logs separately.

**Architecture:** Keep input data resolution unchanged through `base_path/dataset_name`. Add small path helpers and a CLI argument in `weight_advantage_pretrain.py`, then route model output and file logger paths through `<dataset_name>/<experiment_name>`. Update only the serial commodity training scripts to pass the experiment name and write stdout to the same experiment-specific log directory.

**Tech Stack:** Python 3.10, argparse, logging, PyTorch TensorBoard writer, pytest, bash.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-training-experiment-name-output/plan-ready.md`
- tasks: `openspec/changes/add-training-experiment-name-output/tasks.md`
- plan: `docs/superpowers/plans/2026-07-08-add-training-experiment-name-output.md`

---

### Task 1: Experiment-name output isolation

> **trace:** plan-ready.md → `### Task 1: Experiment-name output isolation` | tasks.md → `- [ ] 1.0 Complete experiment-name output isolation for serial Stage I training.`
> **sync:** tasks.md → `- [ ] 1.0 Complete experiment-name output isolation for serial Stage I training.` | plan-ready.md → `### Task 1: Experiment-name output isolation`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
- Modify: `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`
- Modify: `FineFT/script/train/train_commodity_fu.sh`
- Modify: `FineFT/script/train/train_commodity_al.sh`

- [x] **Step 1: Add failing tests for experiment-name paths and CLI defaults**

Append these tests to `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`:

```python
def test_parser_defaults_experiment_name_to_default():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    args = wap.parser.parse_args([])

    assert args.experiment_name == "default"


def test_parser_accepts_explicit_experiment_name():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    args = wap.parser.parse_args(["--experiment_name", "5min_gamma097"])

    assert args.experiment_name == "5min_gamma097"


def test_build_serial_model_path_includes_experiment_name():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    assert wap.build_serial_model_path(
        result_path="result/DiHFT/low_level",
        dataset_name="fu",
        experiment_name="5min_gamma097",
    ) == "result/DiHFT/low_level/fu/5min_gamma097/weights_advantage_pretrain"


def test_build_train_data_paths_keep_base_path_dataset_name_semantics():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    paths = wap.build_training_data_paths(base_path="dataset_5min", dataset_name="fu")

    assert paths == {
        "train_data_path": "dataset_5min/fu/train",
        "state_features_path": "dataset_5min/fu/state_features.npy",
        "maintenance_margin_ratio_path": "dataset_5min/fu/maintenance_margin_ratio_dict.npy",
    }


def test_build_file_log_path_includes_experiment_name():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    assert wap.build_train_log_path(
        dataset_name="fu",
        experiment_name="5min_gamma097",
    ) == "log_futures/fu/low_level/train/5min_gamma097/advantage.log"
```

- [x] **Step 2: Run tests and verify they fail**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py -q
```

Expected: FAIL with missing `experiment_name` parser attribute or missing helper functions `build_serial_model_path`, `build_training_data_paths`, and `build_train_log_path`.

- [x] **Step 3: Add path helpers and CLI argument**

In `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`, replace `configure_logger(dataset_name)` and add helper functions above it:

```python
def build_train_log_path(dataset_name, experiment_name):
    return os.path.join(
        "log_futures",
        dataset_name,
        "low_level",
        "train",
        experiment_name,
        "advantage.log",
    )


def build_serial_model_path(result_path, dataset_name, experiment_name):
    return os.path.join(
        result_path,
        dataset_name,
        experiment_name,
        "weights_advantage_pretrain",
    )


def build_training_data_paths(base_path, dataset_name):
    dataset_root = os.path.join(base_path, dataset_name)
    return {
        "train_data_path": os.path.join(dataset_root, "train"),
        "state_features_path": os.path.join(dataset_root, "state_features.npy"),
        "maintenance_margin_ratio_path": os.path.join(
            dataset_root,
            "maintenance_margin_ratio_dict.npy",
        ),
    }


def configure_logger(dataset_name, experiment_name):
    log_path = build_train_log_path(dataset_name, experiment_name)
    log_dir = os.path.dirname(log_path)
    os.makedirs(log_dir, exist_ok=True)
    abs_log_path = os.path.abspath(log_path)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == abs_log_path:
            return abs_log_path

    file_handler = logging.FileHandler(abs_log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return abs_log_path
```

Add the parser argument after `--dataset_name`:

```python
parser.add_argument(
    "--experiment_name",
    type=str,
    default="default",
    help="experiment name used to namespace serial training outputs",
)
```

- [x] **Step 4: Wire helpers into `Weighted_Contexts_DQN.__init__` and `__main__`**

In `Weighted_Contexts_DQN.__init__`, replace the current model path and data path blocks with:

```python
self.experiment_name = args.experiment_name
self.model_path = build_serial_model_path(
    args.result_path,
    args.dataset_name,
    args.experiment_name,
)
```

Replace the current training data path block:

```python
self.base_path = args.base_path
self.dataset_name = args.dataset_name
training_data_paths = build_training_data_paths(self.base_path, self.dataset_name)
self.train_data_path = training_data_paths["train_data_path"]
self.total_df_index_length = len(os.listdir(self.train_data_path)) - 1
self.tech_indicator_list = np.load(training_data_paths["state_features_path"])
self.maintenance_margin_ratio_dict = np.load(
    training_data_paths["maintenance_margin_ratio_path"],
    allow_pickle=True,
).item()
```

At the bottom of the file, replace:

```python
configure_logger(args.dataset_name)
```

with:

```python
configure_logger(args.dataset_name, args.experiment_name)
```

- [x] **Step 5: Run focused tests and verify Python changes pass**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py -q
```

Expected: PASS.

- [x] **Step 6: Update serial commodity training shell scripts**

In both `FineFT/script/train/train_commodity_fu.sh` and `FineFT/script/train/train_commodity_al.sh`, add after `cd "$ROOTPATH"`:

```bash
EXPERIMENT_NAME=${EXPERIMENT_NAME:-default}
```

For `FineFT/script/train/train_commodity_fu.sh`, replace the `mkdir` line with:

```bash
mkdir -p "log_futures/fu/low_level/train/${EXPERIMENT_NAME}"
```

Add this argument to the Python command after `--dataset_name fu`:

```bash
--experiment_name "${EXPERIMENT_NAME}"
```

Replace stdout redirection with:

```bash
>"log_futures/fu/low_level/train/${EXPERIMENT_NAME}/advantage.log"
```

For `FineFT/script/train/train_commodity_al.sh`, replace the `mkdir` line with:

```bash
mkdir -p "log_futures/al/low_level/train/${EXPERIMENT_NAME}"
```

Add this argument to the Python command after `--dataset_name al`:

```bash
--experiment_name "${EXPERIMENT_NAME}"
```

Replace stdout redirection with:

```bash
>"log_futures/al/low_level/train/${EXPERIMENT_NAME}/advantage.log"
```

- [x] **Step 7: Verify shell syntax**

Run:

```bash
bash -n FineFT/script/train/train_commodity_fu.sh FineFT/script/train/train_commodity_al.sh
```

Expected: exits 0 with no output.

- [x] **Step 8: Run py_compile and OpenSpec validation**

Run:

```bash
conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py
openspec validate add-training-experiment-name-output --strict
```

Expected: both commands exit 0; OpenSpec prints `Change 'add-training-experiment-name-output' is valid`.

- [x] **Step 9: Confirm no parallel training file changes**

Run:

```bash
git diff -- FineFT/RL/DiHFT/low_level/parallel_weight_advantage_pretrain.py
```

Expected: no output.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
