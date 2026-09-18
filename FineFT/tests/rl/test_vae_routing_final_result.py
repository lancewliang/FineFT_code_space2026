import json
import os
import types
import numpy as np
import pandas as pd
import pytest

from RL.DiHFT.high_level import vae_routing_util as vru


def test_resolve_parameters_from_optuna_csv_via_trial_id(tmp_path):
    optuna_csv = tmp_path / "optuna_results.csv"
    pd.DataFrame(
        [
            {
                "number": 77,
                "params_slope_window_length": 117,
                "params_volatility_window_length": 129,
                "params_slope_gamma": 0.9705,
                "params_volatility_gamma": 0.9564,
                "params_slope_rule_base_threshold": 0.2188,
                "params_volatility_rule_base_threshold": 0.2385,
            }
        ]
    ).to_csv(optuna_csv, index=False)

    para_file = tmp_path / "high_level_agent_para.txt"
    para_file.write_text(
        "gamma_0.9705_window_129_threshold_0.2188_trial_77\n", encoding="utf-8"
    )

    args = types.SimpleNamespace(
        para_file=str(para_file),
        optuna_csv=str(optuna_csv),
        slope_window_length=None,
        volatility_window_length=None,
        slope_gamma=None,
        volatility_gamma=None,
        slope_rule_base_threshold=None,
        volatility_rule_base_threshold=None,
        window_length=64,
        gamma=0.9,
        rule_base_threshold=0.2,
    )

    resolved = vru.resolve_routing_parameters(args)

    assert resolved.slope_window_length == 117
    assert resolved.volatility_window_length == 129
    assert resolved.slope_gamma == pytest.approx(0.9705)
    assert resolved.volatility_gamma == pytest.approx(0.9564)
    assert resolved.slope_rule_base_threshold == pytest.approx(0.2188)
    assert resolved.volatility_rule_base_threshold == pytest.approx(0.2385)
    assert resolved.window_length == 129
    assert resolved.gamma == pytest.approx(0.9705)
    assert resolved.rule_base_threshold == pytest.approx(0.2188)


def test_resolve_parameters_from_formatted_string_without_csv(tmp_path):
    para_file = tmp_path / "high_level_agent_para.txt"
    para_file.write_text(
        "ws_62_wv_118_gs_0.9762_gv_0.9362_ts_0.4372_tv_0.2051\n", encoding="utf-8"
    )

    args = types.SimpleNamespace(
        para_file=str(para_file),
        optuna_csv=None,
        slope_window_length=None,
        volatility_window_length=None,
        slope_gamma=None,
        volatility_gamma=None,
        slope_rule_base_threshold=None,
        volatility_rule_base_threshold=None,
        window_length=64,
        gamma=0.9,
        rule_base_threshold=0.2,
    )

    resolved = vru.resolve_routing_parameters(args)

    assert resolved.slope_window_length == 62
    assert resolved.volatility_window_length == 118
    assert resolved.slope_gamma == pytest.approx(0.9762)
    assert resolved.volatility_gamma == pytest.approx(0.9362)
    assert resolved.slope_rule_base_threshold == pytest.approx(0.4372)
    assert resolved.volatility_rule_base_threshold == pytest.approx(0.2051)
    assert resolved.window_length == 118
    assert resolved.gamma == pytest.approx(0.9762)
    assert resolved.rule_base_threshold == pytest.approx(0.2051)


def test_eval_stage_test_loads_contracts_from_test_directory(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset" / "fu"
    test_dir = dataset_root / "test"
    test_dir.mkdir(parents=True)
    pd.DataFrame({"mark_price": [100.0, 102.0], "contract_reward": [2.0, 2.0], "required_money": [10.0, 10.0]}).to_feather(
        test_dir / "fu2508.feather"
    )
    pd.DataFrame({"mark_price": [200.0, 204.0], "contract_reward": [4.0, 4.0], "required_money": [20.0, 20.0]}).to_feather(
        test_dir / "fu2509.feather"
    )

    created_envs = []

    class FakeEnv:
        def __init__(self, df):
            self.reward = float(df["contract_reward"].iloc[0])
            self.required_money = float(df["required_money"].iloc[0])
            self.position = 0.0
            self.leverage = 5
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

    monkeypatch.setattr(vru, "initiate_base_env", lambda **kwargs: FakeEnv(kwargs["df"]))
    monkeypatch.setattr(
        vru,
        "calculate_required_money",
        lambda initial_margin, *_: float(initial_margin[0]),
    )

    routing = vru.vae_risk_aware_routing.__new__(vru.vae_risk_aware_routing)
    routing.base_path = str(tmp_path / "dataset")
    routing.dataset_name = "fu"
    routing.eval_stage = "test"
    routing.eval_stage_dir = str(test_dir)
    routing.single_data_path = str(dataset_root / "test.feather")
    routing.test_path = str(tmp_path / "result" / "final_result")
    routing.tech_indicator_list = []
    routing.max_holding_number = 2
    routing.position_choices = 5
    routing.leverage_choices = [5]
    routing.long_estimated_rate = 0.0
    routing.short_estimated_rate = 0.0
    routing.transcation_cost = 0.0004
    routing.maintenance_margin_ratio_dict = {}
    routing.early_stop = 0
    routing.initial_state = (10000.0, 0.0, 0.0, 0.0, 5.0)
    routing.initial_wallet_balance = 10000.0
    routing.allow_reverse_position = True
    routing.order_book_depth = 5
    routing.num_labels = 3
    routing.slot_count = 9
    routing.window_length = 64
    routing.axis_window_lengths = {"slope": 64, "volatility": 64}
    routing.axis_gammas = {"slope": 0.9, "volatility": 0.9}
    routing.axis_thresholds = {"slope": 0.2, "volatility": 0.2}
    routing.zero_position_action = 4
    routing.action = 4
    routing.macro_action_history = []
    routing.quantiles = {"slope": [[] for _ in range(3)], "volatility": [[] for _ in range(3)]}
    routing.initial_rollout = types.MethodType(
        lambda self, env, s, info: (env, s, 0.0, False, info),
        routing,
    )
    routing.get_action = types.MethodType(lambda self, info, s, current_position, current_leverage: 1, routing)
    routing.get_quantiles = types.MethodType(lambda self, s: None, routing)

    return_rate = routing.test()

    assert return_rate == pytest.approx(0.0003)
    contract_csv = tmp_path / "result" / "final_result" / "contract_results.csv"
    assert contract_csv.exists()
    df_res = pd.read_csv(contract_csv)
    assert df_res["contract"].tolist() == ["fu2508", "fu2509"]
    assert (tmp_path / "result" / "final_result" / "contracts" / "fu2508" / "macro_action.npy").exists()
    assert (tmp_path / "result" / "final_result" / "contracts" / "fu2508" / "macro_action_history.npy").exists()
    assert (tmp_path / "result" / "final_result" / "macro_action.npy").exists()
    assert (tmp_path / "result" / "final_result" / "trading_info.npy").exists()


def test_vae_routing_final_result_macro_action_entrypoint(tmp_path, monkeypatch):
    from RL.DiHFT.high_level import vae_routing_final_result_macro_action as vrf

    dataset_root = tmp_path / "dataset" / "30min" / "fu"
    test_dir = dataset_root / "test"
    test_dir.mkdir(parents=True)
    pd.DataFrame({"mark_price": [100.0, 102.0], "contract_reward": [2.0, 2.0], "required_money": [10.0, 10.0]}).to_feather(
        test_dir / "fu2508.feather"
    )

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
                "artifacts": {"model_assembly": str(tmp_path / "model.pth")},
            }
        ),
        encoding="utf-8",
    )

    para_file = tmp_path / "high_level_agent_para.txt"
    para_file.write_text("ws_50_wv_60_gs_0.95_gv_0.96_ts_0.25_tv_0.30\n", encoding="utf-8")

    class FakeEnv:
        def __init__(self, df):
            self.reward = float(df["contract_reward"].iloc[0])
            self.required_money = float(df["required_money"].iloc[0])
            self.position = 0.0
            self.leverage = 5
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

    monkeypatch.setattr(vru, "initiate_base_env", lambda **kwargs: FakeEnv(kwargs["df"]))
    monkeypatch.setattr(
        vru,
        "calculate_required_money",
        lambda initial_margin, *_: float(initial_margin[0]),
    )
    monkeypatch.setattr(
        vrf.vae_risk_aware_routing,
        "__init__",
        lambda self, args: setattr(self, "args", args) or setattr(self, "test_path", str(tmp_path / "final_result")) or None,
    )
    monkeypatch.setattr(
        vrf.vae_risk_aware_routing,
        "test",
        lambda self: 0.123456,
    )

    test_args = [
        "prog",
        "--base_path",
        str(tmp_path / "dataset" / "30min"),
        "--dataset_name",
        "fu",
        "--experiment_name",
        "30min_multi",
        "--selection_manifest",
        str(manifest_path),
        "--para_file",
        str(para_file),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    args = vrf.parse_and_prepare_args()

    assert args.eval_stage == "test"
    assert args.slope_window_length == 50
    assert args.volatility_window_length == 60
    assert args.slope_gamma == pytest.approx(0.95)
    assert args.volatility_gamma == pytest.approx(0.96)
    assert args.slope_rule_base_threshold == pytest.approx(0.25)
    assert args.volatility_rule_base_threshold == pytest.approx(0.30)
    assert args.selection_manifest == str(manifest_path)


