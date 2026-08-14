import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


FINEFT_ROOT = Path(__file__).resolve().parents[2]
if str(FINEFT_ROOT) not in sys.path:
    sys.path.insert(0, str(FINEFT_ROOT))


SEVEN_LABEL_SEMANTICS = (
    "label_0:limit_down,label_1:strong_down,label_2:weak_down,"
    "label_3:sideways,label_4:weak_up,label_5:strong_up,label_6:limit_up"
)


def test_cli_maps_all_discovered_labels_and_builds_default_capacities():
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    args = guarded.parser.parse_args(
        ["--label_action_semantics", SEVEN_LABEL_SEMANTICS]
    )
    semantics = guarded.resolve_label_action_semantics(
        [f"label_{index}" for index in range(7)],
        args.label_action_semantics,
    )
    capacities = {
        label: guarded.label_action_capacity(
            semantic,
            window_size=args.label_action_window_size,
            weak_ratio=args.weak_label_opposed_ratio,
            strong_ratio=args.strong_label_opposed_ratio,
        )
        for label, semantic in semantics.items()
    }

    assert semantics == {
        "label_0": "limit_down",
        "label_1": "strong_down",
        "label_2": "weak_down",
        "label_3": "sideways",
        "label_4": "weak_up",
        "label_5": "strong_up",
        "label_6": "limit_up",
    }
    assert capacities == {
        "label_0": 0,
        "label_1": 2,
        "label_2": 4,
        "label_3": 10,
        "label_4": 4,
        "label_5": 2,
        "label_6": 0,
    }


@pytest.mark.parametrize(
    ("semantic", "expected_capacity"),
    [("weak_up", 2), ("strong_down", 2)],
)
def test_adjustable_quota_capacity_uses_floor(semantic, expected_capacity):
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    assert (
        guarded.label_action_capacity(
            semantic,
            window_size=10,
            weak_ratio=0.25,
            strong_ratio=0.25,
        )
        == expected_capacity
    )


@pytest.mark.parametrize(
    ("mapping_text", "error_pattern"),
    [
        ("label_0:weak_down", "missing.*label_1"),
        (
            "label_0:weak_down,label_0:strong_down,label_1:weak_up",
            "duplicate.*label_0",
        ),
        (
            "label_0:weak_down,label_1:weak_up,label_2:sideways",
            "unknown.*label_2",
        ),
        ("label_0:down,label_1:weak_up", "unsupported.*down"),
    ],
)
def test_label_semantics_mapping_fails_fast_with_diagnostic_items(
    mapping_text, error_pattern
):
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    with pytest.raises(ValueError, match=error_pattern):
        guarded.resolve_label_action_semantics(
            ["label_0", "label_1"], mapping_text
        )


@pytest.mark.parametrize(
    "cli_args",
    [
        ["--label_action_window_size", "0"],
        ["--weak_label_opposed_ratio", "-0.01"],
        ["--weak_label_opposed_ratio", "1.01"],
        ["--strong_label_opposed_ratio", "-0.01"],
        ["--strong_label_opposed_ratio", "1.01"],
        ["--opposed_holding_stop_loss_ratio", "-0.01"],
    ],
)
def test_guard_cli_rejects_invalid_quota_parameters(cli_args):
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    args = guarded.parser.parse_args(
        ["--label_action_semantics", "label_0:sideways", *cli_args]
    )
    with pytest.raises(ValueError):
        guarded.validate_guard_cli_args(args)


class _FakeNet:
    def eval(self) -> None:
        pass


class _GuardTrajectoryEnv:
    def __init__(
        self,
        step_count: int,
        *,
        execute_targets: bool,
        initial_position: float = 0.0,
        opening_price: float = 0.0,
        current_mark_price: float = 100.0,
        current_action: int = 2,
    ) -> None:
        self.step_count = step_count
        self.execute_targets = execute_targets
        self.step_index = 0
        self.position = initial_position
        self.leverage = 1
        self.wallet_balance = 100_000.0
        self.unrealized_pnl = 0.0
        self.current_markprice = current_mark_price
        self.current_holding_opening_price = opening_price
        self.current_holding_average_price = opening_price
        self.current_action = current_action
        self.received_actions = []
        self.initial_margin_history = []
        self.wallet_balance_history = []
        self.unrealized_pnl_history = []
        self.maintain_marigine_history = []
        self.new_position_required_money_history = []

    def _info(self) -> dict[str, object]:
        return {
            "previous_action": self.current_action,
            "avaliable_action": [1, 1, 1, 1, 1],
            "funding_count_down_hour": 0,
            "funding_count_down_minute": 0,
            "trading_info": [0.0, 0.0, 0.0, 0.0],
            "commission_fee_step": 0.0,
            "realized_pnl_step": 0.0,
            "slippage_step": 0.0,
            "cumulative_commission_fee": 0.0,
            "cumulative_realized_pnl": 0.0,
            "cumulative_slippage": 0.0,
            "current_holding_opening_price": self.current_holding_opening_price,
            "current_holding_average_price": self.current_holding_average_price,
        }

    def reset(self) -> tuple[list[float], dict[str, object]]:
        return [0.0], self._info()

    def step(
        self, action: int
    ) -> tuple[list[float], float, bool, dict[str, object]]:
        self.received_actions.append(action)
        if self.execute_targets:
            self.position = {0: -2.0, 1: -1.0, 2: 0.0, 3: 1.0, 4: 2.0}[action]
        self.step_index += 1
        done = self.step_index == self.step_count
        return [0.0], 0.0, done, self._info()


def _write_guard_slice(
    tmp_path: Path,
    mark_prices: list[float] | np.ndarray,
    filename: str = "df_0.feather",
    label: str = "label_0",
) -> None:
    label_dir = tmp_path / "valid" / "fu2507" / label
    label_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01", periods=len(mark_prices), freq="min"
            ),
            "close": mark_prices,
            "volume": np.ones(len(mark_prices)),
            "mark_price": mark_prices,
        }
    ).to_feather(label_dir / filename)


def _write_real_env_guard_slice(tmp_path: Path) -> None:
    mark_prices = np.array([100.0, 97.0, 97.0])
    row_count = len(mark_prices)
    data = {
        "timestamp": pd.date_range("2026-01-01", periods=row_count, freq="min"),
        "close": mark_prices,
        "volume": np.ones(row_count),
        "mark_price": mark_prices,
        "funding_rate": np.zeros(row_count),
        "funding_timestamp": pd.date_range(
            "2026-01-01", periods=row_count, freq="min"
        ),
        "is_limit_down": [False, True, True],
        "is_limit_up": [False, False, False],
        "limit_up_single_sided_ratio": np.zeros(row_count),
        "limit_down_single_sided_ratio": [0.0, 1.0, 1.0],
        "limit_up_ask_depth_ratio_5": np.zeros(row_count),
        "limit_down_bid_depth_ratio_5": np.zeros(row_count),
        "UpperLimitPrice": np.full(row_count, 110.0),
        "LowerLimitPrice": np.array([90.0, 97.0, 97.0]),
    }
    for level in range(1, 26):
        data[f"ask{level}_price"] = mark_prices + level
        data[f"ask{level}_size"] = np.full(row_count, 10.0)
        data[f"bid{level}_price"] = mark_prices - level
        data[f"bid{level}_size"] = np.full(row_count, 10.0)
    label_dir = tmp_path / "valid" / "fu2507" / "label_0"
    label_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_feather(label_dir / "df_0.feather")


def _make_guarded_trader(
    guarded: ModuleType,
    tmp_path: Path,
    *,
    semantic: str,
    proposed_actions: list[int],
    window_size: int = 3,
    weak_ratio: float = 0.34,
    strong_ratio: float = 0.34,
    initial_action: int = 2,
) -> object:
    trader = guarded.weighted_trader.__new__(guarded.weighted_trader)
    trader.eval_net = _FakeNet()
    trader.valid_data_path = str(tmp_path / "valid")
    trader.initial_action_list = [initial_action]
    trader.N = 1
    trader.leverage_choices = [1]
    trader.position_list = [-2.0, -1.0, 0.0, 1.0, 2.0]
    trader.initial_wallet_balance = 100_000.0
    trader.initial_unrealized_pnL = 0.0
    trader.max_holding_number = 2
    trader.position_choices = 5
    trader.order_book_depth = 5
    trader.long_estimated_rate = 0.0
    trader.short_estimated_rate = 0.0
    trader.transcation_cost = 0.0
    trader.maintenance_margin_ratio_dict = {}
    trader.tech_indicator_list = []
    trader.epoch_path = str(tmp_path)
    trader.epoch_num = 1
    trader.save_trading_detail_csv = True
    trader.allow_reverse_position = True
    trader.label_action_semantics = {"label_0": semantic}
    trader.label_action_window_size = window_size
    trader.weak_label_opposed_ratio = weak_ratio
    trader.strong_label_opposed_ratio = strong_ratio
    trader.opposed_holding_stop_loss_ratio = 0.03
    action_iterator = iter(proposed_actions)
    trader.observed_masks = []

    def act_test(
        state: object, info: dict[str, object], bin_index: int
    ) -> int:
        trader.observed_masks.append(list(info["avaliable_action"]))
        return next(action_iterator)

    trader.act_test = act_test
    return trader


def test_complete_trajectory_applies_rolling_quota_after_model_and_recovers(
    monkeypatch, tmp_path
):
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    _write_guard_slice(tmp_path, [100.0, 100.0, 100.0, 100.0])
    created_envs = []

    def make_env(**kwargs):
        env = _GuardTrajectoryEnv(4, execute_targets=False)
        created_envs.append(env)
        return env

    monkeypatch.setattr(guarded, "initiate_base_env", make_env)
    trader = _make_guarded_trader(
        guarded,
        tmp_path,
        semantic="weak_down",
        proposed_actions=[3, 3, 3, 3],
    )

    trader.test()

    detail = pd.read_csv(tmp_path / "trading_action_detail_epoch_1.csv")
    assert trader.observed_masks == [[1, 1, 1, 1, 1]] * 4
    assert detail["proposed_action"].tolist() == [3, 3, 3, 3]
    assert detail["动作"].tolist() == [3, 2, 2, 3]
    assert detail["guard_decision"].tolist() == [
        "allowed",
        "quota_hold",
        "quota_hold",
        "allowed",
    ]
    assert detail["opposed_action_count"].tolist() == [1, 1, 1, 1]
    assert detail["opposed_action_capacity"].tolist() == [1, 1, 1, 1]
    assert detail["执行后仓位"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert created_envs[0].received_actions == [3, 2, 2, 3]

    result = np.load(tmp_path / "analysis_result.npy", allow_pickle=True).tolist()
    assert result[0]["turnover"] == [0.75]
    assert result[0]["label_action_semantic"] == "weak_down"
    assert result[0]["label_action_window_size"] == 3
    assert result[0]["weak_label_opposed_ratio"] == 0.34
    assert result[0]["strong_label_opposed_ratio"] == 0.34
    assert result[0]["opposed_holding_stop_loss_ratio"] == 0.03


def test_window_size_one_does_not_carry_the_previous_step(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    _write_guard_slice(tmp_path, [100.0, 100.0])
    created_envs = []

    def make_env(**kwargs):
        env = _GuardTrajectoryEnv(2, execute_targets=True)
        created_envs.append(env)
        return env

    monkeypatch.setattr(guarded, "initiate_base_env", make_env)
    trader = _make_guarded_trader(
        guarded,
        tmp_path,
        semantic="strong_down",
        proposed_actions=[3, 4],
        window_size=1,
        strong_ratio=1.0,
    )

    trader.test()

    assert created_envs[0].received_actions == [3, 4]


@pytest.mark.parametrize(
    ("semantic", "step_count", "expected_actions"),
    [
        ("weak_down", 5, [3, 3, 3, 3, 2]),
        ("strong_down", 3, [3, 3, 2]),
        ("limit_down", 1, [2]),
    ],
)
def test_complete_trajectory_uses_default_weak_strong_and_limit_capacities(
    monkeypatch, tmp_path, semantic, step_count, expected_actions
):
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    _write_guard_slice(tmp_path, [100.0] * step_count)
    created_envs = []

    def make_env(**kwargs):
        env = _GuardTrajectoryEnv(step_count, execute_targets=False)
        created_envs.append(env)
        return env

    monkeypatch.setattr(guarded, "initiate_base_env", make_env)
    trader = _make_guarded_trader(
        guarded,
        tmp_path,
        semantic=semantic,
        proposed_actions=[3] * step_count,
        window_size=10,
        weak_ratio=0.40,
        strong_ratio=0.20,
    )

    trader.test()

    assert created_envs[0].received_actions == expected_actions


def test_opposed_open_add_and_reduce_count_but_hold_and_close_do_not(
    monkeypatch, tmp_path
):
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    _write_guard_slice(tmp_path, [100.0] * 5)
    monkeypatch.setattr(
        guarded,
        "initiate_base_env",
        lambda **kwargs: _GuardTrajectoryEnv(5, execute_targets=True),
    )
    trader = _make_guarded_trader(
        guarded,
        tmp_path,
        semantic="weak_down",
        proposed_actions=[3, 4, 3, 3, 2],
        window_size=10,
        weak_ratio=0.30,
    )

    trader.test()

    detail = pd.read_csv(tmp_path / "trading_action_detail_epoch_1.csv")
    assert detail["动作"].tolist() == [3, 4, 3, 3, 2]
    assert detail["opposed_action_count"].tolist() == [1, 2, 3, 3, 3]
    assert detail["guard_decision"].tolist() == ["allowed"] * 5


def test_sideways_trajectory_never_defines_an_opposed_action(monkeypatch, tmp_path):
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    _write_guard_slice(tmp_path, [100.0] * 4)
    created_envs = []

    def make_env(**kwargs):
        env = _GuardTrajectoryEnv(4, execute_targets=True)
        created_envs.append(env)
        return env

    monkeypatch.setattr(guarded, "initiate_base_env", make_env)
    trader = _make_guarded_trader(
        guarded,
        tmp_path,
        semantic="sideways",
        proposed_actions=[0, 4, 1, 3],
    )

    trader.test()

    assert created_envs[0].received_actions == [0, 4, 1, 3]
    detail = pd.read_csv(tmp_path / "trading_action_detail_epoch_1.csv")
    assert detail["opposed_action_count"].tolist() == [0, 0, 0, 0]


def test_nested_trajectory_boundaries_each_start_with_a_fresh_quota(
    monkeypatch, tmp_path
):
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    _write_guard_slice(tmp_path, [100.0, 100.0], filename="df_0.feather")
    _write_guard_slice(tmp_path, [100.0, 100.0], filename="df_1.feather")
    _write_guard_slice(
        tmp_path,
        [100.0, 100.0],
        filename="df_0.feather",
        label="label_1",
    )
    _write_guard_slice(
        tmp_path,
        [100.0, 100.0],
        filename="df_1.feather",
        label="label_1",
    )
    created_envs = []

    def make_env(**kwargs):
        env = _GuardTrajectoryEnv(2, execute_targets=False)
        created_envs.append(env)
        return env

    monkeypatch.setattr(guarded, "initiate_base_env", make_env)
    trader = _make_guarded_trader(
        guarded,
        tmp_path,
        semantic="weak_down",
        proposed_actions=[3, 3] * 16,
    )
    trader.initial_action_list = [1, 2]
    trader.N = 2
    trader.label_action_semantics["label_1"] = "weak_down"

    trader.test()

    assert len(created_envs) == 16
    assert all(env.received_actions == [3, 2] for env in created_envs)


def test_quota_hold_uses_environment_action_for_non_grid_actual_position(
    monkeypatch, tmp_path
):
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    _write_guard_slice(tmp_path, [100.0])
    created_envs = []

    def make_env(**kwargs):
        env = _GuardTrajectoryEnv(
            1,
            execute_targets=False,
            initial_position=0.5,
            current_action=3,
        )
        created_envs.append(env)
        return env

    monkeypatch.setattr(guarded, "initiate_base_env", make_env)
    trader = _make_guarded_trader(
        guarded,
        tmp_path,
        semantic="weak_down",
        proposed_actions=[4],
        weak_ratio=0.0,
        initial_action=3,
    )

    trader.test()

    assert created_envs[0].received_actions == [3]
    detail = pd.read_csv(tmp_path / "trading_action_detail_epoch_1.csv")
    assert detail.loc[0, "guard_decision"] == "quota_hold"
    assert detail.loc[0, "执行后仓位"] == 0.5


@pytest.mark.parametrize(
    (
        "semantic",
        "initial_action",
        "initial_position",
        "proposed_action",
        "current_mark_price",
    ),
    [
        ("weak_down", 3, 1.0, 4, 97.0),
        ("weak_up", 1, -1.0, 0, 103.0),
    ],
)
def test_quota_block_at_exact_adverse_threshold_attempts_symmetric_close(
    monkeypatch,
    tmp_path,
    semantic,
    initial_action,
    initial_position,
    proposed_action,
    current_mark_price,
):
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    _write_guard_slice(tmp_path, [current_mark_price])
    created_envs = []

    def make_env(**kwargs):
        env = _GuardTrajectoryEnv(
            1,
            execute_targets=False,
            initial_position=initial_position,
            opening_price=100.0,
            current_mark_price=current_mark_price,
        )
        created_envs.append(env)
        return env

    monkeypatch.setattr(guarded, "initiate_base_env", make_env)
    trader = _make_guarded_trader(
        guarded,
        tmp_path,
        semantic=semantic,
        proposed_actions=[proposed_action],
        weak_ratio=0.0,
        initial_action=initial_action,
    )

    trader.test()

    detail = pd.read_csv(tmp_path / "trading_action_detail_epoch_1.csv")
    assert created_envs[0].received_actions == [2]
    assert detail.loc[0, "guard_decision"] == "stop_loss_close"
    assert detail.loc[0, "动作"] == 2
    assert detail.loc[0, "执行后仓位"] == initial_position
    assert detail.loc[0, "current_holding_opening_price"] == 100.0
    assert detail.loc[0, "current_holding_average_price"] == 100.0


def test_adverse_move_does_not_stop_loss_when_candidate_fits_quota(
    monkeypatch, tmp_path
):
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    _write_guard_slice(tmp_path, [97.0])
    created_envs = []

    def make_env(**kwargs):
        env = _GuardTrajectoryEnv(
            1,
            execute_targets=False,
            initial_position=1.0,
            opening_price=100.0,
            current_mark_price=97.0,
        )
        created_envs.append(env)
        return env

    monkeypatch.setattr(guarded, "initiate_base_env", make_env)
    trader = _make_guarded_trader(
        guarded,
        tmp_path,
        semantic="weak_down",
        proposed_actions=[4],
        window_size=10,
        weak_ratio=0.10,
        initial_action=3,
    )

    trader.test()

    assert created_envs[0].received_actions == [4]
    detail = pd.read_csv(tmp_path / "trading_action_detail_epoch_1.csv")
    assert detail.loc[0, "guard_decision"] == "allowed"


def test_invalid_discovered_label_mapping_fails_before_model_construction(
    monkeypatch, tmp_path
):
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    _write_guard_slice(tmp_path, [100.0])
    args = guarded.parser.parse_args(
        [
            "--base_path",
            str(tmp_path),
            "--dataset_name",
            "",
            "--label_action_semantics",
            "label_1:sideways",
        ]
    )
    monkeypatch.setattr(
        guarded,
        "ensemble_Qnet",
        lambda **kwargs: pytest.fail("model must not be constructed"),
    )

    with pytest.raises(ValueError, match="missing.*label_0.*unknown.*label_1"):
        guarded.weighted_trader(args)


def test_real_env_limit_down_can_prevent_guard_stop_loss_close(tmp_path):
    from RL.DiHFT.low_level import test_agent_index_with_guard as guarded

    _write_real_env_guard_slice(tmp_path)
    trader = _make_guarded_trader(
        guarded,
        tmp_path,
        semantic="weak_down",
        proposed_actions=[3, 4],
        weak_ratio=0.0,
        initial_action=3,
    )
    trader.max_holding_number = 4
    trader.position_list = [-4.0, -2.0, 0.0, 2.0, 4.0]
    trader.maintenance_margin_ratio_dict = {"50000": [0.004, 0.0]}

    trader.test()

    detail = pd.read_csv(tmp_path / "trading_action_detail_epoch_1.csv")
    assert detail["guard_decision"].tolist() == ["allowed", "stop_loss_close"]
    assert detail["动作"].tolist() == [3, 2]
    assert detail["执行后仓位"].tolist() == [2.0, 2.0]
