# add-diverse-rollout-latest-logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Print the latest diverse-training metrics for each `df_index + rollout_index` at every epoch boundary before the existing epoch summary log.

**Architecture:** Keep the change inside `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`. Add two small module-level helpers: one records latest diverse rollout metrics into a train-level dict, and one logs that dict in sorted order with `盈利`/`亏损` labels. `Weighted_Contexts_DQN.train()` owns the dict lifecycle and calls the helpers only from the diverse-training path.

**Tech Stack:** Python, PyTorch training loop, standard `logging`, pytest, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-diverse-rollout-latest-logging/plan-ready.md`
- tasks: `openspec/changes/add-diverse-rollout-latest-logging/tasks.md`
- plan: `docs/superpowers/plans/2026-07-02-add-diverse-rollout-latest-logging.md`

---

### Task 1: Latest metrics helpers

> **trace:** plan-ready.md → `### Task 1: Latest metrics helpers` | tasks.md → `- [ ] 1.0 Complete latest metrics helper implementation.`
> **sync:** tasks.md → `- [ ] 1.0 Complete latest metrics helper implementation.` | plan-ready.md → `### Task 1: Latest metrics helpers`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
- Test: `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`

- [x] **Step 1: Write failing tests for recording, overwriting, sorted logging, labels, and empty cache**

Append these tests to `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`:

```python
def test_record_diverse_rollout_latest_metric_overwrites_existing_key():
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    metrics_by_df = {}

    wap.record_diverse_rollout_latest_metric(
        metrics_by_df,
        df_index=2,
        rollout_index=1,
        reward_sum=100.0,
        final_balance=101000.0,
        return_rate=0.01,
    )
    wap.record_diverse_rollout_latest_metric(
        metrics_by_df,
        df_index=2,
        rollout_index=1,
        reward_sum=-50.0,
        final_balance=99500.0,
        return_rate=-0.005,
    )

    assert metrics_by_df == {
        2: {
            1: {
                "reward_sum": -50.0,
                "final_balance": 99500.0,
                "return_rate": -0.005,
            }
        }
    }


def test_log_diverse_rollout_latest_metrics_sorts_and_labels_profit_loss(caplog):
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    metrics_by_df = {
        2: {
            1: {
                "reward_sum": 20.0,
                "final_balance": 100100.0,
                "return_rate": 0.001,
            }
        },
        1: {
            3: {
                "reward_sum": 0.0,
                "final_balance": 100000.0,
                "return_rate": 0.0,
            },
            0: {
                "reward_sum": -30.0,
                "final_balance": 99900.0,
                "return_rate": -0.001,
            },
        },
    }

    with caplog.at_level(logging.INFO, logger=logger.name):
        wap.log_diverse_rollout_latest_metrics(7, metrics_by_df)

    messages = [
        record.message
        for record in caplog.records
        if "多样化训练最新明细" in record.message
    ]
    assert messages == [
        "第 7 轮 epoch 训练完成 | 多样化训练最新明细 | df_index=1 | rollout_index=0 | 累计奖励=-30.0000 | 最终余额=99900.0000 | 收益率=-0.001000 | 亏损",
        "第 7 轮 epoch 训练完成 | 多样化训练最新明细 | df_index=1 | rollout_index=3 | 累计奖励=0.0000 | 最终余额=100000.0000 | 收益率=0.000000 | 亏损",
        "第 7 轮 epoch 训练完成 | 多样化训练最新明细 | df_index=2 | rollout_index=1 | 累计奖励=20.0000 | 最终余额=100100.0000 | 收益率=0.001000 | 盈利",
    ]


def test_log_diverse_rollout_latest_metrics_skips_empty_cache(caplog):
    from RL.DiHFT.low_level import weight_advantage_pretrain as wap

    with caplog.at_level(logging.INFO, logger=logger.name):
        wap.log_diverse_rollout_latest_metrics(3, {})

    assert "多样化训练最新明细" not in caplog.text
```

- [x] **Step 2: Run helper tests and verify they fail**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_record_diverse_rollout_latest_metric_overwrites_existing_key FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_log_diverse_rollout_latest_metrics_sorts_and_labels_profit_loss FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_log_diverse_rollout_latest_metrics_skips_empty_cache -q
```

Expected: FAIL with `AttributeError` for `record_diverse_rollout_latest_metric` or `log_diverse_rollout_latest_metrics`, because the helpers do not exist yet.

- [x] **Step 3: Add latest-metrics helpers**

In `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`, add these functions after `summarize_rollout_metrics(...)` and before `summarize_rollout_diagnostics(...)`:

```python
def record_diverse_rollout_latest_metric(
    metrics_by_df,
    df_index,
    rollout_index,
    reward_sum,
    final_balance,
    return_rate,
):
    df_metrics = metrics_by_df.setdefault(int(df_index), {})
    df_metrics[int(rollout_index)] = {
        "reward_sum": float(reward_sum),
        "final_balance": float(final_balance),
        "return_rate": float(return_rate),
    }


def log_diverse_rollout_latest_metrics(epoch_index, metrics_by_df):
    for df_index in sorted(metrics_by_df):
        for rollout_index in sorted(metrics_by_df[df_index]):
            metrics = metrics_by_df[df_index][rollout_index]
            profit_label = "盈利" if metrics["return_rate"] > 0 else "亏损"
            logger.info(
                "第 %d 轮 epoch 训练完成 | 多样化训练最新明细 | "
                "df_index=%d | rollout_index=%d | 累计奖励=%.4f | "
                "最终余额=%.4f | 收益率=%.6f | %s",
                epoch_index,
                df_index,
                rollout_index,
                metrics["reward_sum"],
                metrics["final_balance"],
                metrics["return_rate"],
                profit_label,
            )
```

- [x] **Step 4: Run helper tests and verify they pass**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_record_diverse_rollout_latest_metric_overwrites_existing_key FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_log_diverse_rollout_latest_metrics_sorts_and_labels_profit_loss FineFT/tests/rl/test_weight_advantage_pretrain_logging.py::test_log_diverse_rollout_latest_metrics_skips_empty_cache -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Training loop integration

> **trace:** plan-ready.md → `### Task 2: Training loop integration` | tasks.md → `- [ ] 2.0 Complete training loop integration.`
> **sync:** tasks.md → `- [ ] 2.0 Complete training loop integration.` | plan-ready.md → `### Task 2: Training loop integration`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
- Test: `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`

- [x] **Step 1: Initialize the train-level cache before the sample loop**

In `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`, inside `Weighted_Contexts_DQN.train()`, add the cache immediately after `epoch_number = 4`:

```python
epoch_number = 4
diverse_rollout_latest_metrics_by_df = {}
group_number = self.N
```

- [x] **Step 2: Record latest diverse rollout metrics after each diverse rollout**

In the `else:` branch for multi-style diverse training, replace the repeated return-rate expression after `required_money = self.initial_wallet_balance` with a named value and record it. The block should read:

```python
final_balance = env.unrealized_pnl + env.wallet_balance
required_money = self.initial_wallet_balance
diverse_return_rate = final_balance / (required_money + 1e-12) - 1
self.writer.add_scalar(
    tag="return_rate_train_{}".format(index),
    scalar_value=diverse_return_rate,
    global_step=sample,
    walltime=None,
)

self.writer.add_scalar(
    tag="reward_sum_train_{}".format(index),
    scalar_value=episode_reward_sum,
    global_step=sample,
    walltime=None,
)
record_diverse_rollout_latest_metric(
    diverse_rollout_latest_metrics_by_df,
    df_index,
    index,
    episode_reward_sum,
    final_balance,
    diverse_return_rate,
)
logger.info(
    "多样化回合结束 | 上下文索引=%d | 累计奖励=%.4f | 最终余额=%.4f | 收益率=%.6f",
    index,
    episode_reward_sum,
    final_balance,
    diverse_return_rate,
)
```

Keep the existing `sample_rollout_metrics.append(...)` block unchanged so epoch mean behavior does not change:

```python
sample_rollout_metrics.append(
    {
        "return_rate": final_balance / (required_money + 1e-12),
        "final_balance": final_balance,
        "reward_sum": episode_reward_sum,
    }
)
```

- [x] **Step 3: Log latest diverse rollout metrics at the epoch boundary before the existing epoch summary**

Inside `if len(epoch_reward_sum_train_list) == epoch_number:`, after `epoch_path` creation and `torch.save(...)`, but before the existing epoch summary `logger.info("第 %d 轮 epoch 训练完成 | 平均收益率=...")`, add:

```python
log_diverse_rollout_latest_metrics(
    epoch_index,
    diverse_rollout_latest_metrics_by_df,
)
```

The final order in the epoch block should be:

```python
torch.save(
    self.eval_net.state_dict(),
    os.path.join(epoch_path, "trained_model.pkl"),
)
log_diverse_rollout_latest_metrics(
    epoch_index,
    diverse_rollout_latest_metrics_by_df,
)
logger.info(
    "第 %d 轮 epoch 训练完成 | 平均收益率=%.6f | 平均最终余额=%.4f | 平均累计奖励=%.4f | 模型已保存至=%s",
    epoch_index,
    mean_return_rate_train,
    mean_final_balance_train,
    mean_reward_sum_train,
    epoch_path,
)
```

- [x] **Step 4: Run focused logging tests**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py -q
```

Expected: PASS.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 3: Verification

> **trace:** plan-ready.md → `### Task 3: Verification` | tasks.md → `- [ ] 3.0 Complete verification.`
> **sync:** tasks.md → `- [ ] 3.0 Complete verification.` | plan-ready.md → `### Task 3: Verification`

**Files:**
- Verify: `FineFT/tests/rl/test_weight_advantage_pretrain_logging.py`
- Verify: `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
- Verify: `openspec/changes/add-diverse-rollout-latest-logging/specs/fineft-stage-i-training-logging/spec.md`

- [x] **Step 1: Run focused pytest**

Run:

```bash
conda activate finetf && pytest FineFT/tests/rl/test_weight_advantage_pretrain_logging.py -q
```

Expected: PASS for the full logging-focused test file.

- [x] **Step 2: Run Python syntax check**

Run:

```bash
conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py
```

Expected: exit code 0 and no output.

- [x] **Step 3: Run OpenSpec strict validation**

Run:

```bash
openspec validate add-diverse-rollout-latest-logging --strict
```

Expected:

```text
Change 'add-diverse-rollout-latest-logging' is valid
```

- [x] **Step 4: Inspect the relevant diff**

Run:

```bash
git diff -- FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py FineFT/tests/rl/test_weight_advantage_pretrain_logging.py openspec/changes/add-diverse-rollout-latest-logging docs/superpowers/plans/2026-07-02-add-diverse-rollout-latest-logging.md
```

Expected: diff only shows the latest-metrics helpers, focused logging tests, training-loop cache/log calls, and this change's spec/plan documents. It should not show changes to sample planning, reward calculation, pretrain/full-df warmup behavior, TensorBoard epoch scalar names, or model save paths.

- [x] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
