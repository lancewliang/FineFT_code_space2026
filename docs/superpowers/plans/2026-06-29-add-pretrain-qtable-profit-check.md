# Pretrain Qtable Profit Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Precompute the actual qtables used by `weight_advantage_pretrain.py` before the sample loop and print per-sample DP-path profitability diagnostics.

**Architecture:** Keep the change surgical inside `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`. Add focused helper methods on `Weighted_Contexts_DQN` for sample-plan generation, df loading, initial-state setup, qtable caching, environment construction, and DP-path reward evaluation; then make `train()` consume the plan/cache instead of sampling qtables inside the pretrain branch.

**Tech Stack:** Python 3.10, pandas, NumPy, existing FineFT demo environment utilities, existing qtable utilities, OpenSpec.

**Traceability (sddflow):**
- plan-ready: `openspec/changes/add-pretrain-qtable-profit-check/plan-ready.md`
- tasks: `openspec/changes/add-pretrain-qtable-profit-check/tasks.md`
- plan: `docs/superpowers/plans/2026-06-29-add-pretrain-qtable-profit-check.md`

---

## File Structure

- `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`: modify only this training script. Add small helper methods to `Weighted_Contexts_DQN`, build qtable diagnostics before `for sample in range(self.num_sample)`, and reuse the cached qtable in the pretrain branch.
- `openspec/changes/add-pretrain-qtable-profit-check/tasks.md`: already created; build stage will mark task checkboxes when complete.
- `openspec/changes/add-pretrain-qtable-profit-check/plan-ready.md`: already created; build stage will mark task checkboxes when complete.

### Task 1: Sample plan and qtable cache implementation

> **trace:** plan-ready.md -> `### Task 1: Sample plan and qtable cache implementation` | tasks.md -> `- [ ] 1.0 Complete sample plan and qtable cache implementation.`
> **sync:** tasks.md -> `- [ ] 1.0 Complete sample plan and qtable cache implementation.` | plan-ready.md -> `### Task 1: Sample plan and qtable cache implementation`

**Files:**
- Modify: `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`

- [ ] **Step 1: Add helper methods to `Weighted_Contexts_DQN`**

Insert these methods after `act_multi_styles_pretrain()` and before `act_multi_styles()`:

```python
    def _build_sample_plan(self):
        sample_plan = []
        for _ in range(self.num_sample):
            df_index = random.choices(range(self.total_df_index_length), k=1)[0]
            initial_action = random.choices(range(self.position_choices), k=1)[0]
            sample_plan.append((df_index, initial_action))
        return sample_plan

    def _load_train_df_by_index(self, df_index):
        df_path = os.path.join(
            self.train_data_path, "df_{}.feather".format(df_index)
        )
        return pd.read_feather(df_path)

    def _set_initial_state_from_action(self, train_df, initial_action):
        self.initial_position, self.initial_leverage = (
            map_action_to_position_leverage(
                initial_action, self.leverage_choices, self.position_list
            )
        )
        current_markprice = train_df["mark_price"].values[0]
        self.initial_margin = np.abs(
            self.initial_position * current_markprice / self.initial_leverage
        )
        self.initial_state = (
            self.initial_wallet_balance,
            self.initial_margin,
            self.initial_unrealized_pnL,
            self.initial_position,
            self.initial_leverage,
        )

    def _create_demo_env_for_df(self, train_df):
        return initiate_demo_env(
            df=train_df,
            feature_list=self.tech_indicator_list,
            max_holding_number=self.max_holding_number,
            order_book_depth=self.order_book_depth,
            position_choices=self.position_choices,
            leverage_choice=self.leverage_choices,
            long_estimated_rate=self.long_estimated_rate,
            short_estimated_rate=self.short_estimated_rate,
            commission_rate=self.transcation_cost,
            maintenance_margin_ratio_dict=self.maintenance_margin_ratio_dict,
            early_stop=self.early_stop,
            initial_state=self.initial_state,
            gamma=self.gamma,
            max_punishment=1e10,
        )

    def _create_q_table_for_df(self, train_df):
        return create_optimal_q_table_from_df(
            df=train_df,
            max_holding_number=self.max_holding_number,
            order_book_depth=self.order_book_depth,
            position_choices=self.position_choices,
            leverage_choice=self.leverage_choices,
            long_estimated_rate=self.long_estimated_rate,
            short_estimated_rate=self.short_estimated_rate,
            commission_rate=self.transcation_cost,
            max_punishment=1e10,
            gamma=1,
        )

    def _evaluate_dp_action_path_reward(
        self, train_df, q_table, initial_action, sample_index, df_index
    ):
        self._set_initial_state_from_action(train_df, initial_action)
        env = self._create_demo_env_for_df(train_df)
        perfection_action_list = get_dp_action_from_qtable(q_table, initial_action)
        s, info = env.reset()
        episode_reward_sum = 0

        for action in perfection_action_list:
            s, reward, done, info = env.step(action)
            episode_reward_sum += reward
            if done:
                break

        profitable = episode_reward_sum > 0
        logger.info(
            "qtable诊断 | sample=%d | df_index=%d | initial_action=%d | episode_reward_sum=%.4f | profitable=%s",
            sample_index + 1,
            df_index,
            initial_action,
            episode_reward_sum,
            profitable,
        )
        print(
            "qtable诊断: sample={} df_index={} initial_action={} episode_reward_sum={:.4f} profitable={}".format(
                sample_index + 1,
                df_index,
                initial_action,
                episode_reward_sum,
                profitable,
            )
        )
        return episode_reward_sum, profitable

    def _prepare_q_table_cache_and_diagnostics(self, sample_plan):
        q_table_cache = {}
        train_df_cache = {}
        for sample_index, (df_index, initial_action) in enumerate(sample_plan):
            if df_index not in train_df_cache:
                train_df_cache[df_index] = self._load_train_df_by_index(df_index)
            if df_index not in q_table_cache:
                q_table_cache[df_index] = self._create_q_table_for_df(
                    train_df_cache[df_index]
                )
            self._evaluate_dp_action_path_reward(
                train_df_cache[df_index],
                q_table_cache[df_index],
                initial_action,
                sample_index,
                df_index,
            )
        return q_table_cache, train_df_cache
```

- [ ] **Step 2: Build sample plan and qtable cache before the sample loop**

In `train()`, replace this block:

```python
        step_counter_pretrain = 0
        step_counter_diverse = 0
        for sample in range(self.num_sample):
```

with:

```python
        step_counter_pretrain = 0
        step_counter_diverse = 0
        sample_plan = self._build_sample_plan()
        q_table_cache, train_df_cache = self._prepare_q_table_cache_and_diagnostics(
            sample_plan
        )
        for sample in range(self.num_sample):
```

- [ ] **Step 3: Make the training loop consume `sample_plan`**

Inside the `for sample in range(self.num_sample):` loop, replace:

```python
            df_index = random.choices(range(self.total_df_index_length), k=1)[0]
            initial_action = random.choices(range(self.position_choices), k=1)[0]
```

with:

```python
            df_index, initial_action = sample_plan[sample]
```

- [ ] **Step 4: Reuse cached dfs and shared initial-state/env helpers in the loop**

Replace the current df loading, initial state setup, and `initiate_demo_env(...)` block:

```python
            self.train_df = pd.read_feather(
                os.path.join(self.train_data_path, "df_{}.feather".format(df_index))
            )
            self.initial_position, self.initial_leverage = (
                map_action_to_position_leverage(
                    initial_action, self.leverage_choices, self.position_list
                )
            )
            print(
                "初始仓位={}, 初始杠杆={}".format(
                    self.initial_position, self.initial_leverage
                )
            )
            current_markprice = self.train_df["mark_price"].values[0]
            self.initial_margin = np.abs(
                self.initial_position * current_markprice / self.initial_leverage
            )
            self.initial_state = (
                self.initial_wallet_balance,
                self.initial_margin,
                self.initial_unrealized_pnL,
                self.initial_position,
                self.initial_leverage,
            )
            env = initiate_demo_env(
                df=self.train_df,
                feature_list=self.tech_indicator_list,
                max_holding_number=self.max_holding_number,
                order_book_depth=self.order_book_depth,
                position_choices=self.position_choices,
                leverage_choice=self.leverage_choices,
                long_estimated_rate=self.long_estimated_rate,
                short_estimated_rate=self.short_estimated_rate,
                commission_rate=self.transcation_cost,
                maintenance_margin_ratio_dict=self.maintenance_margin_ratio_dict,
                early_stop=self.early_stop,
                initial_state=self.initial_state,
                gamma=self.gamma,
                max_punishment=1e10,
            )
```

with:

```python
            self.train_df = train_df_cache[df_index]
            self._set_initial_state_from_action(self.train_df, initial_action)
            print(
                "初始仓位={}, 初始杠杆={}".format(
                    self.initial_position, self.initial_leverage
                )
            )
            env = self._create_demo_env_for_df(self.train_df)
```

- [ ] **Step 5: Reuse cached qtable in the pretrain branch**

Inside `if pretrain:`, replace:

```python
                q_table = create_optimal_q_table_from_df(
                    df=self.train_df,
                    max_holding_number=self.max_holding_number,
                    order_book_depth=self.order_book_depth,
                    position_choices=self.position_choices,
                    leverage_choice=self.leverage_choices,
                    long_estimated_rate=self.long_estimated_rate,
                    short_estimated_rate=self.short_estimated_rate,
                    commission_rate=self.transcation_cost,
                    max_punishment=1e10,
                    gamma=1,
                )
```

with:

```python
                q_table = q_table_cache[df_index]
```

- [ ] **Step 6: Run a focused syntax check for the changed script**

Run:

```bash
conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py
```

Expected: command exits with status 0 and prints no Python syntax errors.

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）

### Task 2: Verification

> **trace:** plan-ready.md -> `### Task 2: Verification` | tasks.md -> `- [ ] 2.0 Complete verification.`
> **sync:** tasks.md -> `- [ ] 2.0 Complete verification.` | plan-ready.md -> `### Task 2: Verification`

**Files:**
- Verify: `FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py`
- Verify: `openspec/changes/add-pretrain-qtable-profit-check/specs/fineft-stage-i-pretrain/spec.md`
- Verify: `openspec/changes/add-pretrain-qtable-profit-check/tasks.md`

- [ ] **Step 1: Run Python syntax verification in the requested conda environment**

Run:

```bash
conda activate finetf && python -m py_compile FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py
```

Expected: command exits with status 0 and produces no `SyntaxError`.

- [ ] **Step 2: Run OpenSpec strict validation**

Run:

```bash
openspec validate add-pretrain-qtable-profit-check --strict
```

Expected:

```text
Change 'add-pretrain-qtable-profit-check' is valid
```

- [ ] **Step 3: Check whether local training data is available**

Run:

```bash
test -d dataset && find dataset -path '*/train/df_0.feather' -print -quit
```

Expected if data is available: prints a path ending in `/train/df_0.feather`.

Expected if data is unavailable: prints nothing. Record that the smoke run was skipped because the local dataset is absent.

- [ ] **Step 4: If data is available, run a small smoke command**

Use the dataset name from the printed path. For a path like `dataset/BTCUSDT/train/df_0.feather`, run:

```bash
conda activate finetf && python FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py --base_path dataset --dataset_name BTCUSDT --num_sample 2 --pretrain_epoch 1
```

Expected: before `===== 第 1/2 轮采样 =====`, output includes two qtable diagnostics. Each diagnostic line includes `qtable诊断: sample=`, `df_index=`, `initial_action=`, `episode_reward_sum=`, and `profitable=`.

The first diagnostic line starts with `qtable诊断: sample=1`; the second starts with `qtable诊断: sample=2`.

- [ ] **Step 5: Confirm qtable calculation is not duplicated in the pretrain branch**

Run:

```bash
rg -n "create_optimal_q_table_from_df|q_table_cache\\[df_index\\]|random\\.choices\\(range\\(self\\.total_df_index_length\\)|random\\.choices\\(range\\(self\\.position_choices\\)" FineFT/RL/DiHFT/low_level/weight_advantage_pretrain.py
```

Expected: output includes the existing import, the two `random.choices(...)` calls inside `_build_sample_plan()`, one `return create_optimal_q_table_from_df(` inside `_create_q_table_for_df()`, and one `q_table = q_table_cache[df_index]` inside the pretrain branch. The two `random.choices(...)` matches should be inside `_build_sample_plan()`, not inside the sample loop.

- [ ] **Task complete**（本 Task 全部 Step 为 `[x]` 后勾选；与 plan-ready **任务完成**、tasks.md 对应行同步）
