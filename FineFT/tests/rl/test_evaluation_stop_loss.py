"""Exercise the stop through the actual environment and evaluation entry points."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from env.env_initiate.base_initiate import initiate_base_env
from model.low_level import ensemble_Qnet
from RL.DiHFT.low_level.evaluation_stop_loss import (
    EvaluationStopLoss,
    executed_position,
)


def market(direction=1):
    prices = [100.0, 100.0 - 4 * direction] + [100.0 - 2 * direction] * 6
    times = pd.date_range("2026-01-01", periods=8, freq="10min")
    return pd.DataFrame(
        {
            "timestamp": times,
            "funding_rate": 0.0,
            "funding_timestamp": times + pd.Timedelta(hours=1),
            "mark_price": prices,
            "bid1_price": np.array(prices) - 0.01,
            "ask1_price": np.array(prices) + 0.01,
            "bid1_size": [10, 10, 0.4, 0.4, 10, 10, 10, 10],
            "ask1_size": [10, 10, 0.4, 0.4, 10, 10, 10, 10],
            "limit_down_single_sided_ratio": [0, int(direction > 0), 0, 0, 0, 0, 0, 0],
            "limit_up_single_sided_ratio": [0, int(direction < 0), 0, 0, 0, 0, 0, 0],
            "feature1": 0.0,
        }
    )


@pytest.mark.parametrize("direction", [1, -1])
def test_real_environment_blocked_partial_exit_and_reentry(direction):
    env = initiate_base_env(
        market(direction),
        ["feature1"],
        order_book_depth=1,
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        allow_partial_positions=True,
        initial_state=(10000, 100, 0, direction, 1),
        maintenance_margin_ratio_dict={"1000000000000": [0.1, 0.0]},
        commission_rate=0,
        long_estimated_rate=0,
        short_estimated_rate=0,
    )
    env.reset()
    guard = EvaluationStopLoss()
    requests, positions = [], []
    for step in range(6):
        stop = guard.before_action(env)
        env.step(1 if stop else (2 if direction > 0 else 0))
        position = executed_position(env)
        guard.after_execution(position)
        requests.append(stop)
        positions.append(position)
    assert requests == [False, True, True, True, True, False]
    assert positions[1] == direction
    assert abs(positions[2]) == pytest.approx(0.6)
    assert abs(positions[3]) == pytest.approx(0.2)
    assert positions[4] == 0
    assert positions[5] == direction
    assert not guard.pending


def test_threshold_boundary_disabled_invalid_and_pending():
    env = SimpleNamespace(position=2.0, current_markprice=100.0, unrealized_pnl=-6.0)
    assert not EvaluationStopLoss(0).before_action(env)
    stop = EvaluationStopLoss()
    assert stop.before_action(env)
    stop.after_execution(1)
    env.position, env.unrealized_pnl = 1, 1
    assert stop.before_action(env)
    assert stop.pending
    for value in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            EvaluationStopLoss(value)


def test_training_stop_loss_threshold_configuration():
    from RL.DiHFT.low_level.parallel_weight_advantage_pretrain import parser

    assert parser.parse_args([]).stop_loss_return_threshold == 0.03
    assert (
        parser.parse_args(
            ["--stop_loss_return_threshold", "0"]
        ).stop_loss_return_threshold
        == 0
    )
    assert (
        parser.parse_args(
            ["--stop_loss_return_threshold", "0.015"]
        ).stop_loss_return_threshold
        == 0.015
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--stop_loss_return_threshold", "-1"])


@pytest.mark.parametrize("threshold", [None, "0", "0.015"])
def test_training_shell_forwards_stop_loss(tmp_path, threshold):
    script = (
        Path(__file__).resolve().parents[2] / "script/train/train_commodity_fu_10.sh"
    )
    binaries = tmp_path / "bin"
    binaries.mkdir()
    conda_root = tmp_path / "conda"
    profile = conda_root / "etc/profile.d/conda.sh"
    profile.parent.mkdir(parents=True)
    profile.write_text("conda() { return 0; }\n")
    for name, body in {
        "python": '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$CAPTURE_PATH"\n',
        "conda": f"#!/bin/sh\nprintf '%s\\n' '{conda_root}'\n",
    }.items():
        path = binaries / name
        path.write_text(body)
        path.chmod(0o700)
    capture = tmp_path / "arguments"
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.defpath,
        "ROOTPATH": str(tmp_path),
        "CAPTURE_PATH": str(capture),
    }
    env.pop("STOP_LOSS_RETURN_THRESHOLD", None)
    if threshold is not None:
        env["STOP_LOSS_RETURN_THRESHOLD"] = threshold
    subprocess.run(
        ["bash", str(script)], env=env, check=True, capture_output=True, text=True
    )
    argv = capture.read_text().splitlines()
    assert argv[argv.index("--stop_loss_return_threshold") + 1] == (threshold or "0.03")


@pytest.mark.parametrize(
    "entry,legacy_layout",
    [("index", False), ("average", False), ("average", True), ("batch", False)],
)
@pytest.mark.parametrize("end_before_exit", [False, True])
def test_entry_uses_stop_without_extra_records(
    tmp_path, monkeypatch, entry, legacy_layout, end_before_exit
):
    from RL.DiHFT.low_level import evaluate_sub_agents as batch
    from RL.DiHFT.low_level import test_agent_average as average
    from RL.DiHFT.low_level import test_agent_index as index

    dataset = tmp_path / "data/fu"
    dataset.mkdir(parents=True)
    np.save(dataset / "rl_state_features.npy", ["feature1"])
    np.save(
        dataset / "maintenance_margin_ratio_dict.npy", {"1000000000000": [0.1, 0.0]}
    )
    if entry == "index":
        data = dataset / "valid/slope/fu0001/label_0/df_0.feather"
        epoch = tmp_path / "result/fu/test/weights_advantage_pretrain/epoch_1"
    elif entry == "average":
        data = dataset / (
            "valid/label_0/df_0.feather"
            if legacy_layout
            else "valid/volatility/fu0001/label_0/df_0.feather"
        )
        epoch = tmp_path / "result/fu/weights_advantage/epoch_1"
    else:
        data = dataset / "valid/label_0/df_0.feather"
        epoch = tmp_path / "result/fu/weights_advantage/epoch_1"
    data.parent.mkdir(parents=True)
    frame = market()
    (frame.iloc[:3] if end_before_exit else frame).reset_index(drop=True).to_feather(
        data
    )
    epoch.mkdir(parents=True)
    torch.save(ensemble_Qnet(1, 3, 8, 2, 1).state_dict(), epoch / "trained_model.pkl")
    if entry == "batch":
        trader = batch.weighted_trader(
            logg_file_path="",
            data_file_path=str(data),
            model_path=str(epoch / "trained_model.pkl"),
            tech_indicator_list_path=str(dataset / "rl_state_features.npy"),
            maintenance_margin_ratio_dict_path=str(
                dataset / "maintenance_margin_ratio_dict.npy"
            ),
            transcation_cost=0,
            max_holding_number=1,
            position_choices=3,
            N=1,
            hidden_nodes=8,
            order_book_depth=1,
        )
    else:
        module = index if entry == "index" else average
        args = [
            "--base_path",
            str(tmp_path / "data"),
            "--dataset_name",
            "fu",
            "--result_path",
            str(tmp_path / "result"),
            "--N",
            "1",
            "--hidden_nodes",
            "8",
            "--position_choices",
            "3",
            "--epoch_num",
            "1",
            "--leverage_choices",
            "1",
            "--initial_leverage",
            "1",
            "--max_holding_number",
            "1",
            "--order_book_depth",
            "1",
            "--long_estimated_rate",
            "0",
            "--transcation_cost",
            "0",
        ]
        if entry == "index":
            args += [
                "--label_type",
                "slope",
                "--experiment_name",
                "test",
                "--action_persistence",
                "100",
                "--no_enable_limit_reward",
            ]
        else:
            args += ["--model_path", str(epoch.parent)]
        parsed = module.parser.parse_args(args)
        if entry == "average":
            parsed.leverage_choices = [1]
        trader = module.weighted_trader(parsed)
    trader.initial_action_list = [2]
    module = {"index": index, "average": average, "batch": batch}[entry]
    original_factory = module.initiate_base_env
    actions = []
    positions = []

    def capture_env(*args, **kwargs):
        env = original_factory(*args, **kwargs)
        original_step = env.step

        def capture_step(action):
            result = original_step(action)
            actions.append(action)
            positions.append(executed_position(env))
            return result

        env.step = capture_step
        return env

    monkeypatch.setattr(module, "initiate_base_env", capture_env)
    trader.act_test = lambda *args: 2
    result = trader.test()
    assert actions[:2] == [2, 1]
    if end_before_exit:
        assert len(actions) == 2
        assert positions[-1] == 1
    else:
        assert actions[:6] == [2, 1, 1, 1, 1, 2]
        assert positions[4] == 0
        assert positions[5] == 1
    if entry == "batch":
        assert not any(key.startswith("stop_loss_") for key in result[0])
    else:
        name = (
            "slope/analysis_result.npy"
            if entry == "index"
            else "average_analysis_result.npy"
        )
        result_path = epoch / name
        record = np.load(result_path, allow_pickle=True)[0]
        assert not any(key.startswith("stop_loss_") for key in record)
        if entry == "index":
            detail_path = next((epoch / "slope").glob("*detail*.csv"))
            detail = pd.read_csv(detail_path)
            assert "action_source" not in detail.columns
            assert "action_decision_reason" not in detail.columns
            assert not any(column.startswith("stop_loss_") for column in detail.columns)
        assert not (epoch / "evaluations").exists()
        assert not list(epoch.rglob("evaluation_config.json"))
    if not end_before_exit:
        trader.stop_loss_return_threshold = 0.0
        actions.clear()
        trader.test()
        assert actions and all(action == 2 for action in actions)
        if entry != "batch":
            assert result_path.exists()
            assert not (epoch / "evaluations").exists()
    if entry == "average":
        trader.writer.close()


@pytest.mark.parametrize("threshold", [None, "0", "0.015"])
def test_shell_forwards_threshold(tmp_path, threshold):
    root = Path(__file__).resolve().parents[2]
    script = root / "script/test/DiHFT/low_level/test_util_fu_10.sh"
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, body in {
        "python": '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$CAPTURE_PATH"\n',
        "conda": "#!/bin/sh\nexit 0\n",
    }.items():
        path = binaries / name
        path.write_text(body)
        path.chmod(0o700)
    capture = tmp_path / "arguments"
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.defpath,
        "ROOTPATH": str(tmp_path),
        "EPOCH_START": "1",
        "EPOCH_END": "1",
        "LABEL_TYPE": "slope",
        "CAPTURE_PATH": str(capture),
    }
    env.pop("STOP_LOSS_RETURN_THRESHOLD", None)
    if threshold is not None:
        env["STOP_LOSS_RETURN_THRESHOLD"] = threshold
    subprocess.run(
        ["bash", str(script)], env=env, check=True, capture_output=True, text=True
    )
    argv = capture.read_text().splitlines()
    assert argv[argv.index("--stop_loss_return_threshold") + 1] == (threshold or "0.03")


@pytest.mark.parametrize("blocked", [False, True])
@pytest.mark.parametrize("threshold", [0.0, 0.03])
def test_training_exploration_preempts_persistence_on_loss(blocked, threshold):
    from RL.DiHFT.low_level import parallel_diverse_train as training

    frame = market().copy()
    if not blocked:
        frame["limit_down_single_sided_ratio"] = 0
        frame["bid1_size"] = 10.0
    env = initiate_base_env(
        frame,
        ["feature1"],
        order_book_depth=1,
        max_holding_number=1,
        position_choices=3,
        leverage_choice=[1],
        allow_partial_positions=True,
        initial_state=(10000, 100, 0, 1, 1),
        maintenance_margin_ratio_dict={"1000000000000": [0.1, 0.0]},
        commission_rate=0,
        long_estimated_rate=0,
        short_estimated_rate=0,
    )
    actions = []
    original_step = env.step

    def capture_step(action):
        actions.append(action)
        return original_step(action)

    env.step = capture_step
    runner = training.DfRolloutWorkerRunner.__new__(training.DfRolloutWorkerRunner)
    runner._get_initial_state_and_env = lambda *args: (env, (10000, 100, 0, 1, 1))
    runner.model = SimpleNamespace(eval=lambda: None)
    runner._act = lambda *args: (2, 0.0)
    runner.flat_action = 1
    runner.action_persistence = 2
    runner.shared_market_data = None
    runner.initial_wallet_balance = 10000
    runner.gamma = 0.99
    runner.n_step = 1
    runner.action_dim = 3
    runner.stop_loss_return_threshold = threshold
    task = training.ExploreTask(
        df_index=0,
        epoch_index=0,
        context_index=0,
        initial_action=2,
        round_counter=0,
        epsilon=1.0,
    )
    runner.run_task(task)
    assert actions[:2] == ([2, 1] if threshold else [2, 2])
    if blocked and threshold:
        assert actions[:6] == [2, 1, 1, 1, 1, 2]


@pytest.mark.parametrize("use_shared", [False, True])
@pytest.mark.parametrize("empty_book", [False, True])
@pytest.mark.parametrize("round_mode", [False, True])
@pytest.mark.parametrize("threshold", [0.0, 0.03])
def test_training_runner_real_demo_env_partial_stop_exit(
    use_shared, empty_book, round_mode, threshold
):
    from RL.DiHFT.low_level import parallel_diverse_train as training
    from RL.DiHFT.low_level.shared_data_manager import SharedMarketDataPack

    frame = market()
    if empty_book:
        frame.loc[3, ["bid1_size", "ask1_size"]] = 0.0
    env_kwargs = {
        "feature_list": ["feature1"],
        "max_holding_number": 1,
        "order_book_depth": 1,
        "position_choices": 3,
        "leverage_choices": [1],
        "long_estimated_rate": 0.0,
        "short_estimated_rate": 0.0,
        "commission_rate": 0.0,
        "early_stop": 0,
        "maintenance_margin_ratio_dict": {"1000000000000": [0.1, 0.0]},
        "gamma": 0.99,
    }
    runner = training.DfRolloutWorkerRunner.__new__(training.DfRolloutWorkerRunner)
    runner.env_kwargs = env_kwargs
    runner.shared_market_data = (
        SharedMarketDataPack.from_dataframes({0: frame}, env_kwargs)
        if use_shared
        else None
    )
    runner.train_df_by_df = {0: frame}
    runner.cached_envs = {}
    runner.leverage_choices = [1]
    runner.position_list = [-1.0, 0.0, 1.0]
    runner.initial_wallet_balance = 10000
    runner.initial_unrealized_pnL = 0.0
    runner.stop_loss_return_threshold = threshold
    runner.model = SimpleNamespace(eval=lambda: None)
    runner._act = lambda *args: (2, 0.0)
    runner.flat_action = 1
    runner.action_persistence = 2
    runner.gamma = 0.99
    runner.n_step = 1
    runner.action_dim = 3
    if round_mode:
        runner.episodes = {}
        runner.reset_task(
            training.ResetWorkerTask(
                df_index=0, epoch_index=0, context_index=0, initial_action=2
            )
        )
        result = runner.explore_round(
            training.ExploreWorkerRound(
                df_index=0,
                epoch_index=0,
                context_index=0,
                initial_action=2,
                round_counter=0,
                state_dict={},
                epsilon=1,
            )
        )
        actions = [record.transition[2] for record in result.transitions]
        masks = [
            record.transition[5]["avaliable_action"] for record in result.transitions
        ]
        priorities = [record.td_error for record in result.transitions]
    else:
        result = runner.run_task(
            training.ExploreTask(
                df_index=0,
                epoch_index=0,
                context_index=0,
                initial_action=2,
                round_counter=0,
                epsilon=1,
            )
        )
        actions = result.compact_trajectory.actions.tolist()
        masks = result.compact_trajectory.next_avaliable_actions
        priorities = result.compact_trajectory.td_errors
    assert result.done
    if threshold:
        assert actions[:6] == [2, 1, 1, 1, 1, 2]
        for mask in masks[:4]:
            np.testing.assert_array_equal(mask, [0, 1, 0])
        assert max(priorities[1:5]) < 100
    else:
        assert actions[:6] == [2] * 6
    assert runner.cached_envs[0].allow_partial_positions == bool(threshold)
