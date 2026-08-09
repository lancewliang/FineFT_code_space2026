from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


from RL.DiHFT.low_level.test_agents_indexs import (
    STEP_DETAIL_COLUMNS,
    build_position_levels,
    build_step_detail_row,
    discover_model_epochs,
    EvaluationConfig,
    IsolatedAgentEvaluator,
    build_window_rows,
    expand_window_rows,
    validate_required_market_columns,
)


def test_build_position_levels_returns_ordered_signed_levels():
    assert build_position_levels(8.0, 5) == [-8.0, -4.0, 0.0, 4.0, 8.0]


def test_discover_model_epochs_only_accepts_direct_epoch_models(tmp_path):
    (tmp_path / "epoch_2").mkdir()
    (tmp_path / "epoch_2" / "trained_model.pkl").touch()
    (tmp_path / "epoch_1").mkdir()
    (tmp_path / "epoch_1" / "other.bin").touch()
    (tmp_path / "nested" / "epoch_3").mkdir(parents=True)
    (tmp_path / "nested" / "epoch_3" / "trained_model.pkl").touch()

    assert discover_model_epochs(tmp_path) == [(2, tmp_path / "epoch_2")]


def test_required_market_columns_fail_without_raw_volume():
    with pytest.raises(ValueError, match="volume"):
        validate_required_market_columns(pd.DataFrame({"mark_price": [1.0]}))


def test_step_detail_row_uses_stable_english_schema():
    row = build_step_detail_row(
        epoch=2,
        label="label_1",
        contract="fu2505",
        df_path="fu2505/label_1/df_0.feather",
        initial_action=0,
        bin_index=1,
        timestep=0,
        market_row={
            "timestamp": "2025-01-01 09:00:00",
            "close": 100.0,
            "volume": 12.0,
            "mark_price": 100.0,
        },
        action=0,
        target_position=0.0,
        target_leverage=1.0,
        position_before=0.0,
        leverage_before=1.0,
        position_after=0.0,
        leverage_after=1.0,
        step_reward=0.0,
        info={
            "realized_pnl_step": 0.0,
            "cumulative_realized_pnl": 0.0,
            "commission_fee_step": 0.0,
            "cumulative_commission_fee": 0.0,
            "slippage_step": 0.0,
            "cumulative_slippage": 0.0,
        },
        wallet_balance=1000.0,
        unrealized_pnl=0.0,
        action_change_step=0,
        trade_count_step=0,
        cumulative_action_change_count=0,
        cumulative_trade_count=0,
    )

    assert list(row) == STEP_DETAIL_COLUMNS
    assert row["volume"] == 12.0
    assert row["epoch"] == 2


class _FakeQNet:
    def __call__(self, **kwargs):
        return torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)


class _FakeNetwork:
    qnet_list = [_FakeQNet()]


class _FakeEnvironment:
    def __init__(self, **kwargs):
        self.position = 0.0
        self.leverage = 1.0
        self.wallet_balance = 1000.0
        self.unrealized_pnl = 0.0
        self._step = 0

    def reset(self):
        return [0.0], {
            "previous_action": 0,
            "avaliable_action": [1, 1, 1],
            "funding_count_down_hour": 0,
            "funding_count_down_minute": 0,
            "trading_info": [0.0, 0.0, 0.0, 0.0],
        }

    def step(self, action):
        self._step += 1
        self.position = 1.0 if action == 2 else 0.0
        self.unrealized_pnl = float(self._step)
        return [0.0], 1.0, self._step >= 2, {
            "previous_action": action,
            "avaliable_action": [1, 1, 1],
            "funding_count_down_hour": 0,
            "funding_count_down_minute": 0,
            "trading_info": [0.0, 0.0, 0.0, 0.0],
            "realized_pnl_step": 0.0,
            "cumulative_realized_pnl": 0.0,
            "commission_fee_step": 0.0,
            "cumulative_commission_fee": 0.0,
            "slippage_step": 0.0,
            "cumulative_slippage": 0.0,
        }


def test_isolated_evaluator_writes_one_epoch_detail_for_all_initial_actions(tmp_path):
    model_root = tmp_path / "models"
    (model_root / "epoch_1").mkdir(parents=True)
    (model_root / "epoch_1" / "trained_model.pkl").touch()
    valid_root = tmp_path / "valid" / "fu2505" / "label_1"
    valid_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "contract": ["fu2505", "fu2505"],
            "timestamp": ["t0", "t1"],
            "close": [100.0, 101.0],
            "volume": [10.0, 11.0],
            "mark_price": [100.0, 101.0],
        }
    ).to_feather(valid_root / "df_0.feather")
    state_features = tmp_path / "state_features.npy"
    np.save(state_features, np.array(["mark_price"], dtype=object))
    margin = tmp_path / "maintenance_margin_ratio_dict.npy"
    np.save(margin, np.array({}, dtype=object))
    output_dir = tmp_path / "analysis"

    config = EvaluationConfig(
        model_root=model_root,
        valid_root=tmp_path / "valid",
        output_dir=output_dir,
        state_features_path=state_features,
        maintenance_margin_path=margin,
        max_holding_number=1.0,
        position_choices=3,
        ensemble_number=1,
    )
    evaluator = IsolatedAgentEvaluator(
        config,
        environment_factory=_FakeEnvironment,
        model_loader=lambda _: _FakeNetwork(),
    )

    paths = evaluator.run()

    assert paths == [output_dir / "step_detail" / "agent_pattern_step_detail_epoch_1.csv"]
    detail = pd.read_csv(paths[0])
    assert len(detail) == 6
    assert detail.columns.tolist() == STEP_DETAIL_COLUMNS
    assert sorted(detail["initial_action"].unique().tolist()) == [0, 1, 2]
    assert (output_dir / "agent_pattern_coverage_report.csv").exists()
    assert (output_dir / "agent_pattern_window_table.csv").exists()
    assert (output_dir / "agent_pattern_expanded_table.csv").exists()
    assert (output_dir / "agent_pattern_classifier_diagnostics.csv").exists()
    for stem in (
        "agent_pattern_kline_scenario_summary",
        "agent_pattern_kline_triple_summary",
        "agent_pattern_strategy_scenario_summary",
        "agent_pattern_strategy_triple_summary",
        "agent_pattern_cross_scenario_summary",
        "agent_pattern_cross_triple_summary",
    ):
        assert (output_dir / f"{stem}.csv").exists()
    assert (output_dir / "analysis_manifest.json").exists()


def _detail_rows_for_window(initial_action=0):
    rows = []
    values = [(1.0, 2.0, 0.1), (3.0, 5.0, 0.2)] + [(0.0, 5.0, 0.0)] * 18
    for timestep, (realized, unrealized, fee) in enumerate(values):
        rows.append(
            {
                "epoch": 1,
                "label": "label_1",
                "contract": "fu2505",
                "df_path": "fu2505/label_1/df_0.feather",
                "initial_action": initial_action,
                "bin_index": 0,
                "timestep": timestep,
                "timestamp": f"t{timestep}",
                "mark_price": 100.0,
                "realized_pnl_step": realized,
                "unrealized_pnl": unrealized,
                "commission_fee_step": fee,
                "slippage_step": 0.05,
                "position_before": 0.0,
                "position_after": 0.0,
                "volume": 1.0,
            }
        )
    return rows


def test_window_rows_conserve_pnl_and_distinguish_initial_action():
    first = build_window_rows(_detail_rows_for_window(0))[0]
    second = build_window_rows(_detail_rows_for_window(1))[0]

    assert first["gross_pnl"] == pytest.approx(9.0)
    assert first["net_pnl"] == pytest.approx(8.7)
    assert first["window_id"] != second["window_id"]
    assert first["kline_patterns"] == '["未分类"]'
    assert first["strategy_patterns"] == '["策略未分类"]'


def test_limit_trajectory_produces_one_kx1_window_and_expands_once():
    rows = _detail_rows_for_window(0)
    for row in rows:
        row["label"] = "label_6"
    windows = build_window_rows(rows)
    expanded = expand_window_rows(windows)

    assert len(windows) == 1
    assert windows[0]["kline_patterns"] == '["KX1"]'
    assert len(expanded) == 1
    assert expanded[0]["kline_pattern"] == "KX1"


def test_ordinary_trajectory_keeps_absolute_timesteps_across_windows():
    rows = _detail_rows_for_window(0) + [dict(row) for row in _detail_rows_for_window(0)]
    for index, row in enumerate(rows):
        row["timestep"] = index
    windows = build_window_rows(rows)
    assert [(row["start_timestep"], row["end_timestep"]) for row in windows] == [(0, 19), (20, 39)]
