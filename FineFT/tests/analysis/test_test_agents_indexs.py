import pandas as pd
import pytest


from RL.DiHFT.low_level.aggregate_agents_indexs import (
    DETAIL_CSV_HEADER_TO_FIELD,
    STEP_DETAIL_COLUMNS,
    EvaluationConfig,
    aggregate_detail_csvs,
    build_position_levels,
    build_window_rows,
    discover_model_epochs,
    expand_window_rows,
    read_detail_csv,
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


def _write_test_agent_detail(detail_path):
    rows = []
    for initial_action in range(3):
        for timestep in range(2):
            row = {
                column: 0.0
                for column in STEP_DETAIL_COLUMNS
                if column not in {"epoch", "contract"}
            }
            row.update(
                {
                    "label": "label_1",
                    "df_path": "fu2505/label_1/df_0.feather",
                    "initial_action": initial_action,
                    "bin_index": 0,
                    "timestep": timestep,
                    "timestamp": f"t{timestep}",
                    "close": 100.0 + timestep,
                    "volume": 10.0 + timestep,
                    "mark_price": 100.0 + timestep,
                    "target_leverage": 1.0,
                    "leverage_before": 1.0,
                    "leverage_after": 1.0,
                    "wallet_balance": 1000.0,
                    "cash_balance": 1000.0,
                    "total_value": 1000.0,
                }
            )
            rows.append(row)

    english_to_chinese = {
        field: header.removesuffix(".1")
        for header, field in DETAIL_CSV_HEADER_TO_FIELD.items()
    }
    frame = pd.DataFrame(
        rows,
        columns=[
            column
            for column in STEP_DETAIL_COLUMNS
            if column not in {"epoch", "contract"}
        ],
    ).rename(columns=english_to_chinese)
    frame.to_csv(detail_path, index=False)


def test_reads_test_agent_chinese_detail_schema(tmp_path):
    detail_path = tmp_path / "trading_action_detail_epoch_1.csv"
    _write_test_agent_detail(detail_path)

    detail = read_detail_csv(detail_path, epoch=1)

    assert detail.columns[: len(STEP_DETAIL_COLUMNS)].tolist() == STEP_DETAIL_COLUMNS
    assert set(detail["epoch"]) == {1}
    assert set(detail["contract"]) == {"fu2505"}
    assert detail.loc[0, "timestamp"] == "t0"
    assert detail.loc[0, "cash_balance"] == 1000.0


def test_aggregates_test_agent_detail_csv(tmp_path):
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
    state_features.touch()
    margin = tmp_path / "maintenance_margin_ratio_dict.npy"
    margin.touch()
    output_dir = tmp_path / "analysis"
    detail_path = model_root / "epoch_1" / "trading_action_detail_epoch_1.csv"
    _write_test_agent_detail(detail_path)

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
    paths = aggregate_detail_csvs(config, epoch_start=1, epoch_end=1)

    assert paths == [detail_path]
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
