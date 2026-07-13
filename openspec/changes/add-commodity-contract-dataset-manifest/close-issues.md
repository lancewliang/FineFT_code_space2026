# Close Issues

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
