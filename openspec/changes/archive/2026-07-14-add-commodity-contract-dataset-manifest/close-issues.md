# Close Issues

## 2026-07-14

### CRITICAL: final code review found merge blockers

User requested final code review during `/sddflow close`.

Reviewer assessment:

```text
Ready to merge? No
```

Critical findings:

1. `FineFT/script/data/commodity_data_handler_fu.sh` has the dataset generation and per-contract valid slicing stages commented out. It currently only runs VAE creation, so it does not satisfy the advertised FU commodity data handler workflow.
2. `slice_model.py` can still abort the shell for very short valid contracts because it does not skip-and-record insufficient-data contracts before `label_util` filtering/labeling. With shell `set -e`, one short contract can stop the full commodity handler.

Important findings:

1. The shell-script test searches raw text, so commands inside comments still satisfy the assertions.
2. The rerun flow can process stale `valid/df_*.feather` files because scripts scan the directory rather than manifest-selected outputs, and dataset generation does not clean old files.

Close is stopped until these findings are fixed in `/sddflow build`.

### RESOLVED: short valid contracts are skipped and recorded

Build fix applied after review:

- `slice_model.py` now checks for inputs too short for the Butterworth filter before constructing `label_util.Worker`.
- When a contract is too short, it removes that contract's stale label output directory, keeps the processed file, and writes `valid/slice_manifest.json` under `skipped_contracts`.
- A later successful run removes that contract from `skipped_contracts` and rebuilds the label summary from non-skipped contracts.

Regression test:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest FineFT/tests/datahandler/test_slice_model.py::test_run_skips_and_records_contract_with_insufficient_rows -q
```

Result:

```text
1 passed
```

Focused verification:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest FineFT/tests/datahandler/test_slice_model.py FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_vae_data_creation.py FineFT/tests/rl/test_weight_advantage_pretrain_logging.py -q
```

Result:

```text
36 passed, 14 warnings
```

Fresh FU script verification:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && ./FineFT/script/data/commodity_data_handler_fu.sh
```

Result: exit code 0. The regenerated `dataset/10min/fu/valid/slice_manifest.json` contains 12 contract records, labels `label_0` through `label_4`, and no skipped contracts for the current data.

### IMPORTANT: reruns can still consume stale managed outputs

Final re-review on 2026-07-14 found no remaining Critical issues, but found one Important issue that blocks archive:

- `commodity_contract_dataset.py` creates the dataset root but does not clean previous managed outputs before writing a new manifest.
- Commodity shell scripts scan every existing `valid/df_*.feather`, so stale valid files can still be sliced even if they are no longer selected by the new manifest.
- `vae_data_creation.py` reuses an existing `VAE_data` directory, so stale `VAE_data/<contract>/label_*.npy` or `VAE_data/test/test_<contract>.npy` can remain after reruns.

Reviewer assessment:

```text
Ready to merge? With fixes
```

Recommended fix: make reruns idempotent by cleaning managed outputs such as `train/slice`, `valid/processed`, `valid/<contract>`, `valid/slice_manifest.json`, obsolete `train/df_*.feather`, `valid/df_*.feather`, `test/df_*.feather`, and `VAE_data`, or drive shell slicing from the new manifest instead of globbing all existing files.

User decision: archive directly without expanding this change's scope. This issue remains a known follow-up risk rather than a blocker for this archive.

### RESOLVED: `commodity_data_handler_fu.sh` VAE data creation

Command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && ./FineFT/script/data/commodity_data_handler_fu.sh
```

Result: exit code 0.

Output:

```text
start to create VAE data
```

Confirmed generated outputs use the multi-contract layout:

```text
dataset/10min/fu/VAE_data/fu2409/label_0.npy
dataset/10min/fu/VAE_data/fu2505/label_0.npy
dataset/10min/fu/VAE_data/test/test_fu2601.npy
```

Sample shapes from generated arrays show 472 state feature columns:

```text
dataset/10min/fu/VAE_data/fu2409/label_0.npy (418, 472)
dataset/10min/fu/VAE_data/fu2505/label_0.npy (1502, 472)
dataset/10min/fu/VAE_data/test/test_fu2601.npy (4555, 472)
```

### PASS: focused implementation tests

Command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest FineFT/tests/datahandler/test_vae_data_creation.py FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_slice_model.py FineFT/tests/rl/test_weight_advantage_pretrain_logging.py -q
```

Result:

```text
35 passed, 14 warnings in 3.99s
```

### WARNING: full test suite still cannot be collected in the current environment

Command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest FineFT/tests -q
```

Result: exit code 2.

Failure:

```text
FileNotFoundError: [Errno 2] No such file or directory: '/data2/mlqin/HFT4Ind2/dataset/BNBUSDT/train/df_0.feather'
```

The failure still occurs while collecting `FineFT/tests/env/test_env.py`, which reads an external dataset path at import time. Close cannot claim full test-suite verification until this environment dependency is handled or the project defines a close-safe test command.

User selected option 2 on 2026-07-14: use the change-focused close-safe test command for this change. The external `/data2/...` environment test is treated as a known non-blocking environment dependency for this close.

Close-safe command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest FineFT/tests/datahandler/test_vae_data_creation.py FineFT/tests/datahandler/test_commodity_contract_dataset.py FineFT/tests/datahandler/test_slice_model.py FineFT/tests/rl/test_weight_advantage_pretrain_logging.py -q
```

Fresh result:

```text
35 passed, 14 warnings in 3.74s
```

Fresh data-generation result:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && ./FineFT/script/data/commodity_data_handler_fu.sh
```

```text
start to create VAE data
```

### PASS: OpenSpec strict validation

Command:

```bash
openspec validate add-commodity-contract-dataset-manifest --strict
```

Result:

```text
Change 'add-commodity-contract-dataset-manifest' is valid
```

## 2026-07-13

### CRITICAL: `commodity_data_handler_fu.sh` fails during VAE data creation

Command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && ./FineFT/script/data/commodity_data_handler_fu.sh
```

Result: exit code 1.

Failure:

```text
KeyError: "... not in index"
```

Observed cause from local inspection:

- `dataset/10min/fu/state_features.npy` contains 474 feature names.
- `dataset/10min/fu/valid/df_fu2409.feather` contains 115 columns and is already missing 385 state feature columns.
- `dataset/10min/fu/valid/fu2409/label_0/df_0.feather` contains 119 columns and is missing the same 385 state feature columns.

This blocks close because the generated commodity valid label files cannot be converted into VAE arrays using the configured `state_features.npy`.

### CRITICAL: full test suite cannot be collected in the current environment

Command:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate finetf && pytest FineFT/tests -q
```

Result: exit code 2.

Failure:

```text
FileNotFoundError: [Errno 2] No such file or directory: '/data2/mlqin/HFT4Ind2/dataset/BNBUSDT/train/df_0.feather'
```

The failure occurs while collecting `FineFT/tests/env/test_env.py`, which reads an external dataset path at import time. Close cannot claim full test-suite verification until this environment dependency is handled or the project defines a close-safe test command.

### PASS: OpenSpec strict validation

Command:

```bash
openspec validate add-commodity-contract-dataset-manifest --strict
```

Result:

```text
Change 'add-commodity-contract-dataset-manifest' is valid
```
