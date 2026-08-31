import sys
import types
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

if "optuna" not in sys.modules:
    sys.modules["optuna"] = MagicMock()

from RL.DiHFT.high_level import vae_routing_util as vru
from RL.DiHFT.high_level import vae_routing_optuna as vro


def test_load_two_dimensional_selection_manifest_validates_slot_layout(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "axes": {
                    "volatility": ["label_0", "label_1", "label_2"],
                    "slope": ["label_0", "label_1", "label_2"],
                },
                "slot_count": 9,
                "slot_index_formula": "volatility_index * num_labels + slope_index",
                "slots": [
                    {"slot_id": slot_id, "kind": "empty_model"}
                    for slot_id in range(9)
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = vru.load_two_dimensional_selection_manifest(manifest_path)

    assert manifest["slot_count"] == 9
    assert manifest["axes"]["volatility"] == [
        "label_0",
        "label_1",
        "label_2",
    ]


def test_two_dimensional_routing_maps_axis_labels_to_slot():
    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.num_labels = 3
    routing.slot_count = 9
    routing.rule_base_threshold = 0.2
    routing.axis_thresholds = {"volatility": 0.2, "slope": 0.2}
    routing.selection_manifest = {
        "slots": [
            {"slot_id": slot_id, "kind": "model"} for slot_id in range(9)
        ]
    }
    routing.action = 4
    routing.macro_action_history = []
    routing.calculate_axis_window_result = lambda axis: {
        "volatility": [0.1, 0.8, 0.2],
        "slope": [0.1, 0.2, 0.9],
    }[axis]
    routing.agent_act = lambda state, info: 7

    action = routing.get_action({}, np.array([0.0]))

    assert action == 7
    assert routing.selected_agent_index == 5
    assert routing.macro_action_history == [5]


def test_two_dimensional_empty_slot_uses_defensive_action(monkeypatch):
    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.num_labels = 3
    routing.slot_count = 9
    routing.rule_base_threshold = 0.2
    routing.axis_thresholds = {"volatility": 0.2, "slope": 0.2}
    routing.selection_manifest = {
        "slots": [
            {
                "slot_id": slot_id,
                "kind": "empty_model" if slot_id == 0 else "model",
            }
            for slot_id in range(9)
        ]
    }
    routing.zero_position_action = 4
    routing.leverage_choices = [5]
    routing.position_list = [-1, 0, 1]
    routing.action = 6
    routing.macro_action_history = []
    routing.calculate_axis_window_result = lambda axis: [0.9, 0.1, 0.1]
    monkeypatch.setattr(vru, "rule_based_close", lambda *args: 4)

    def fail_if_model_is_called(state, info):
        raise AssertionError("empty slot must not call the qnet")

    routing.agent_act = fail_if_model_is_called

    action = routing.get_action({}, np.array([0.0]))

    assert action == 4
    assert routing.macro_action_history == [9]


def test_two_dimensional_routing_applies_axis_specific_thresholds(monkeypatch):
    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.num_labels = 3
    routing.slot_count = 9
    routing.rule_base_threshold = 0.2
    routing.axis_thresholds = {"volatility": 0.8, "slope": 0.2}
    routing.selection_manifest = {
        "slots": [
            {"slot_id": slot_id, "kind": "model"} for slot_id in range(9)
        ]
    }
    routing.zero_position_action = 4
    routing.leverage_choices = [5]
    routing.position_list = [-1, 0, 1]
    routing.action = 6
    routing.macro_action_history = []
    routing.calculate_axis_window_result = lambda axis: {
        "volatility": [0.7, 0.1, 0.1],
        "slope": [0.1, 0.9, 0.1],
    }[axis]
    monkeypatch.setattr(vru, "rule_based_close", lambda *args: 4)
    routing.agent_act = lambda state, info: 7

    action = routing.get_action({}, np.array([0.0]))

    assert action == 4
    assert routing.macro_action_history == [9]


def test_vae_routing_util_parser_allow_reverse_position():
    args_default = vru.parser.parse_args([])
    assert args_default.allow_reverse_position is False

    args_flag = vru.parser.parse_args(["--allow_reverse_position"])
    assert args_flag.allow_reverse_position is True


def test_vae_routing_util_parser_experiment_name():
    args_default = vru.parser.parse_args([])
    assert args_default.experiment_name == "default"

    args_exp = vru.parser.parse_args(["--experiment_name", "exp123"])
    assert args_exp.experiment_name == "exp123"


def test_vae_routing_optuna_parser_allow_reverse_position():
    args_default = vro.parser_all.parse_args([])
    assert args_default.allow_reverse_position is False

    args_flag = vro.parser_all.parse_args(["--allow_reverse_position"])
    assert args_flag.allow_reverse_position is True


def test_vae_routing_optuna_parser_experiment_name():
    args_default = vro.parser_all.parse_args([])
    assert args_default.experiment_name == "default"

    args_exp = vro.parser_all.parse_args(["--experiment_name", "exp123"])
    assert args_exp.experiment_name == "exp123"


def test_vae_routing_util_parser_order_book_depth():
    args_default = vru.parser.parse_args([])
    assert args_default.order_book_depth == 25

    args_depth = vru.parser.parse_args(["--order_book_depth", "5"])
    assert args_depth.order_book_depth == 5


def test_vae_routing_optuna_parser_order_book_depth():
    args_default = vro.parser_all.parse_args([])
    assert args_default.order_book_depth == 25

    args_depth = vro.parser_all.parse_args(["--order_book_depth", "5"])
    assert args_depth.order_book_depth == 5


def test_vae_routing_optuna_parser_configuration():
    args = vro.parser_all.parse_args(
        [
            "--selection_manifest",
            "manifest.json",
            "--n_trials",
            "7",
            "--n_jobs",
            "2",
        ]
    )

    assert args.selection_manifest == "manifest.json"
    assert args.n_trials == 7
    assert args.n_jobs == 2


def test_prepare_base_args_loads_two_dimensional_model_from_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"artifacts": {"model_assembly": "assembled/model.pth"}}),
        encoding="utf-8",
    )
    args_1 = types.SimpleNamespace(
        dataset_name="BTCUSDT",
        experiment_name="old",
        max_holding_number=8,
        order_book_depth=25,
        allow_reverse_position=False,
    )
    args_2 = types.SimpleNamespace(
        dataset_name="fu",
        experiment_name="30min_multi",
        max_holding_number=2,
        order_book_depth=5,
        allow_reverse_position=True,
        selection_manifest=str(manifest_path),
    )

    prepared = vro.prepare_base_args(args_1, args_2)

    assert prepared.selection_manifest == str(manifest_path)
    assert prepared.allow_reverse_position is True


def test_prepare_base_args_does_not_mutate_original_args(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"artifacts": {"model_assembly": "model.pth"}}),
        encoding="utf-8",
    )
    args_1 = types.SimpleNamespace(
        dataset_name="old_dataset",
        experiment_name="old_experiment",
        max_holding_number=8,
        order_book_depth=25,
        allow_reverse_position=False,
    )
    args_2 = types.SimpleNamespace(
        dataset_name="new_dataset",
        experiment_name="new_experiment",
        max_holding_number=2,
        order_book_depth=5,
        allow_reverse_position=False,
        selection_manifest=str(manifest_path),
    )

    prepared = vro.prepare_base_args(args_1, args_2)

    assert prepared.dataset_name == "new_dataset"
    assert prepared.experiment_name == "new_experiment"
    assert args_1.dataset_name == "old_dataset"
    assert args_1.experiment_name == "old_experiment"


def test_suggest_trial_parameters_uses_independent_axis_parameters():
    class FakeTrial:
        def __init__(self):
            self.names = []

        def suggest_int(self, name, low, high):
            self.names.append(name)
            return {"slope_window_length": 60, "volatility_window_length": 90}[name]

        def suggest_float(self, name, low, high, log=False):
            self.names.append(name)
            return {
                "slope_gamma": 0.93,
                "volatility_gamma": 0.97,
                "slope_rule_base_threshold": 0.25,
                "volatility_rule_base_threshold": 0.45,
            }[name]

    trial = FakeTrial()
    trial_args = types.SimpleNamespace()
    search_args = types.SimpleNamespace(
        window_length_min=50,
        window_length_max=150,
        gamma_min=0.92,
        gamma_max=0.98,
        rule_base_threshold_min=0.2,
        rule_base_threshold_max=0.5,
    )

    vro.suggest_trial_parameters(trial, trial_args, search_args)

    assert trial.names == [
        "slope_window_length",
        "volatility_window_length",
        "slope_gamma",
        "volatility_gamma",
        "slope_rule_base_threshold",
        "volatility_rule_base_threshold",
    ]
    assert trial_args.window_length == 90
    assert trial_args.gamma == 0.93
    assert trial_args.rule_base_threshold == 0.25


def test_vae_routing_optuna_tune_propagates_allow_reverse_position_and_experiment_name():
    args_1 = types.SimpleNamespace(dataset_name="BTCUSDT", max_holding_number=8)
    args_2 = types.SimpleNamespace(
        dataset_name="ETHUSDT",
        max_holding_number=10,
        allow_reverse_position=True,
        experiment_name="exp123",
    )
    args_1.dataset_name = args_2.dataset_name
    args_1.max_holding_number = args_2.max_holding_number
    args_1.allow_reverse_position = getattr(
        args_2, "allow_reverse_position", False
    ) or getattr(args_1, "allow_reverse_position", False)
    args_1.experiment_name = getattr(
        args_2, "experiment_name", "default"
    ) or getattr(args_1, "experiment_name", "default")

    assert args_1.allow_reverse_position is True
    assert args_1.experiment_name == "exp123"


def test_vae_risk_aware_routing_init_stores_allow_reverse_position(monkeypatch):
    monkeypatch.setattr("os.makedirs", lambda *a, **kw: None)
    monkeypatch.setattr("os.path.exists", lambda *a, **kw: True)
    monkeypatch.setattr("numpy.load", lambda *a, **kw: MagicMock(item=lambda: {}))

    args = vru.parser.parse_args(
        ["--allow_reverse_position", "--experiment_name", "exp123"]
    )
    assert args.allow_reverse_position is True
    assert args.experiment_name == "exp123"


def test_vae_routing_test_uses_contract_level_valid_features(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset" / "fu"
    valid_root = dataset_root / "valid"
    valid_root.mkdir(parents=True)
    pd.DataFrame({"contract_reward": [2.0], "required_money": [10.0]}).to_feather(
        valid_root / "fu2508.feather"
    )
    pd.DataFrame({"contract_reward": [6.0], "required_money": [20.0]}).to_feather(
        valid_root / "fu2509.feather"
    )

    created_envs = []

    class FakeEnv:
        def __init__(self, df):
            self.reward = float(df["contract_reward"].iloc[0])
            self.required_money = float(df["required_money"].iloc[0])
            self.margine_balance_history = [100.0, 100.0 + self.reward]
            self.micro_action_history = []
            self.initial_margin_history = [self.required_money]
            self.wallet_balance_history = [100.0]
            self.unrealized_pnl_history = [0.0]
            self.maintain_marigine_history = [0.0]
            self.new_position_required_money_history = [0.0]

        def reset(self):
            return np.array([0.0]), {"previous_action": 0}

        def step(self, action):
            self.micro_action_history.append(action)
            return np.array([1.0]), self.reward, True, {"previous_action": action}

    def fake_initiate_base_env(**kwargs):
        env = FakeEnv(kwargs["df"])
        created_envs.append(env)
        return env

    monkeypatch.setattr(vru, "initiate_base_env", fake_initiate_base_env)
    monkeypatch.setattr(
        vru,
        "calculate_required_money",
        lambda initial_margin, *_: float(initial_margin[0]),
    )

    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.base_path = str(tmp_path / "dataset")
    routing.dataset_name = "fu"
    routing.test_data_path = str(dataset_root / "valid.feather")
    routing.valid_data_path = str(valid_root)
    routing.test_path = str(tmp_path / "result")
    routing.tech_indicator_list = []
    routing.max_holding_number = 8
    routing.position_choices = 9
    routing.leverage_choices = [5]
    routing.long_estimated_rate = 0.0
    routing.short_estimated_rate = 0.0
    routing.transcation_cost = 0.0
    routing.maintenance_margin_ratio_dict = {}
    routing.early_stop = 0
    routing.initial_state = (100.0, 0.0, 0.0, 0.0, 5.0)
    routing.initial_wallet_balance = 100.0
    routing.allow_reverse_position = False
    routing.num_labels = 3
    routing.window_length = 3
    routing.axis_window_lengths = {"slope": 3, "volatility": 3}
    routing.zero_position_action = 4
    routing.action = 99
    routing.macro_action_history = [99]
    routing.quantiles = {"slope": ["stale"], "volatility": ["stale"]}
    routing.initial_rollout = types.MethodType(
        lambda self, env, s, info: (env, s, 0.0, False, info),
        routing,
    )
    routing_start_states = []

    def fake_get_action(self, info, s):
        routing_start_states.append(
            (
                self.action,
                len(self.macro_action_history),
                [
                    len(quantiles)
                    for quantiles in self.quantiles["slope"]
                ],
            )
        )
        self.action = 7
        self.macro_action_history.append(7)
        return 1

    routing.get_action = types.MethodType(fake_get_action, routing)
    routing.get_quantiles = types.MethodType(lambda self, s: None, routing)

    return_rate = routing.test()

    assert return_rate == pytest.approx(0.04)
    assert len(created_envs) == 2
    result = pd.read_csv(tmp_path / "result" / "contract_results.csv")
    assert result["contract"].tolist() == ["fu2508", "fu2509"]
    assert result["return_rate"].tolist() == pytest.approx([0.2, 0.3])
    assert routing_start_states == [(4, 0, [0, 0, 0]), (4, 0, [0, 0, 0])]
    assert (
        tmp_path / "result" / "contracts" / "fu2508" / "reward_history.npy"
    ).exists()
    assert (
        tmp_path / "result" / "contracts" / "fu2509" / "reward_history.npy"
    ).exists()


def test_vae_routing_test_passes_order_book_depth_to_base_env(tmp_path, monkeypatch):
    valid_path = tmp_path / "dataset" / "fu" / "valid.feather"
    valid_path.parent.mkdir(parents=True)
    pd.DataFrame({"contract_reward": [2.0], "required_money": [10.0]}).to_feather(
        valid_path
    )
    captured_kwargs = {}

    class FakeEnv:
        margine_balance_history = [100.0, 102.0]
        micro_action_history = []
        initial_margin_history = [10.0]
        wallet_balance_history = [100.0]
        unrealized_pnl_history = [0.0]
        maintain_marigine_history = [0.0]
        new_position_required_money_history = [0.0]

        def reset(self):
            return np.array([0.0]), {"previous_action": 0}

        def step(self, action):
            return np.array([1.0]), 2.0, True, {"previous_action": action}

    def fake_initiate_base_env(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeEnv()

    monkeypatch.setattr(vru, "initiate_base_env", fake_initiate_base_env)
    monkeypatch.setattr(vru, "calculate_required_money", lambda *args: 10.0)

    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.base_path = str(tmp_path / "dataset")
    routing.dataset_name = "fu"
    routing.test_data_path = str(valid_path)
    routing.valid_data_path = str(tmp_path / "dataset" / "fu" / "valid")
    routing.test_path = str(tmp_path / "result")
    routing.tech_indicator_list = []
    routing.max_holding_number = 8
    routing.position_choices = 9
    routing.leverage_choices = [5]
    routing.long_estimated_rate = 0.0
    routing.short_estimated_rate = 0.0
    routing.transcation_cost = 0.0
    routing.maintenance_margin_ratio_dict = {}
    routing.early_stop = 0
    routing.initial_state = (100.0, 0.0, 0.0, 0.0, 5.0)
    routing.initial_wallet_balance = 100.0
    routing.allow_reverse_position = False
    routing.order_book_depth = 5
    routing.num_labels = 3
    routing.window_length = 3
    routing.axis_window_lengths = {"slope": 3, "volatility": 3}
    routing.zero_position_action = 4
    routing.initial_rollout = types.MethodType(
        lambda self, env, s, info: (env, s, 0.0, False, info),
        routing,
    )
    routing.get_action = types.MethodType(lambda self, info, s: 1, routing)
    routing.get_quantiles = types.MethodType(lambda self, s: None, routing)

    routing.test()

    assert captured_kwargs["order_book_depth"] == 5
