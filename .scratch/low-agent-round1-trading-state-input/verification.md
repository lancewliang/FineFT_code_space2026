# 历史短程验证记录（代码已回退，当前未验收）

状态：未实施、未验收。2026-11-01 用户确认本功能全部代码改动已回退；当前源码仍为四维交易观测。本次仅维护文档，未运行训练或 pytest。

以下命令、测试文件名与通过/失败计数仅保留为回退前历史记录；部分测试文件和接口当前已不存在。它们不能证明当前 HEAD 通过，也不能直接当作当前可复现操作说明。重新实施后须从当前源码重新执行验证，并记录新命令、代码版本、设备及结果。

当前待实施规格使用 `--exit_price_scale_floor 5`；下方历史记录中的 `--tick_size` 是回退前旧命名，不是新实施接口或当前可执行命令。

## 回退前记录

The historical checks below were reported as run from `FineFT/` with the repository instruction:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate finetf
```

The experiment scale was passed explicitly as `--tick_size 5` (yuan/tonne). This is the experiment scale floor and is recorded separately from the exchange minimum price increment of 1 yuan/tonne. The shared metadata records the 18-change/19-observation causal window, age saturation at 256, tanh divisor 5, and holding-duration normalization of 180 steps.

The deterministic contract/model/replay check used synthetic float32 trajectories and seeded replay instances; the parameter-update assertion uses a real Qnet, SGD optimizer, finite loss, and a changed trading-memory input. The lifecycle check uses 25 same-contract rows and records the actual close price at observation step 19. The parallel test group exercises compact IPC, empty and tail micro-batches, FIFO/index recycling, CPU replay and the available CUDA path.

Passed short-run groups:

```text
tests/common/test_trading_observation.py
tests/model/test_low_level_trading_observation.py
tests/rl/test_trading_info_replay_contract.py
tests/env/test_causal_price_scale_preparation.py
tests/env/test_trading_observation_lifecycle_v2.py
    included in the combined command below

tests/rl/test_parallel_diverse_cpu_exploration.py
tests/rl/test_compact_trajectory_and_micro_batched_ingestion.py
tests/rl/test_gpu_resident_buffer_and_pipeline.py
    26 passed

tests/rl/test_test_agent_index.py
tests/rl/test_vae_routing_action_persistence.py
tests/rl/test_expert_hysteresis_routing.py
tests/rl/test_async_greedy_evaluation.py
tests/analysis/test_low_level_agent_ood_analysis.py
    included in the combined command below
```

Combined migrated-contract command: `121 passed, 3 warnings`.

The full-suite command was `python -m pytest -q`. Collection first hit an existing unrelated `Picker` import error; excluding that one test file, after the final contract fixes, produced `652 passed, 80 failed, 32 errors`. The remaining failures are legacy parser/short-fixture and unrelated picker/shell/high-level experiment coverage that has not yet been migrated to the required explicit tick size and causal warmup. No formal historical retraining was started.
