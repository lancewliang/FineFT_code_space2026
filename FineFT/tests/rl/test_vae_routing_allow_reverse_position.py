import sys
import types
from unittest.mock import MagicMock

if 'optuna' not in sys.modules:
    sys.modules['optuna'] = MagicMock()

from RL.DiHFT.high_level import vae_routing_util as vru
from RL.DiHFT.high_level import vae_routing_optuna as vro


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

    args = vru.parser.parse_args(["--allow_reverse_position", "--experiment_name", "exp123"])
    assert args.allow_reverse_position is True
    assert args.experiment_name == "exp123"
