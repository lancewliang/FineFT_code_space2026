from pathlib import Path

import numpy as np
import polars as pl
import pytest

from RL.DiHFT.low_level.test_agents_type_index2 import (
    LIFECYCLE_COLUMNS,
    build_analysis_result,
    extract_trade_lifecycles,
    generate_epoch_artifacts,
)


def _detail_frame(
    positions: list[tuple[float, float]],
    *,
    rewards: list[float],
    label: str = "label_2",
    df_path: str | None = None,
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "label": label,
                "df_path": df_path or f"fu2505/{label}/df_0.feather",
                "initial_action": 2,
                "bin_index": 1,
                "timestep": timestep,
                "timestamp": f"t{timestep}",
                "position_before": before,
                "position_after": after,
                "step_reward": reward,
            }
            for timestep, ((before, after), reward) in enumerate(
                zip(positions, rewards, strict=True)
            )
        ]
    )


def test_extracts_one_row_per_trade_lifecycle_with_requested_fields():
    detail = _detail_frame(
        [(0, -1), (-1, -1), (-1, 0), (0, 1), (1, 0)],
        rewards=[1, 2, 3, 4, 5],
    )

    following, reversion = extract_trade_lifecycles(
        detail, max_holding_number=2
    )

    assert list(following) == LIFECYCLE_COLUMNS
    assert following == {
        "label": "label_2",
        "df_path": "fu2505/label_2/df_0.feather",
        "initial_action": 2,
        "bin_index": 1,
        "start_timestep": 0,
        "end_timestep": 2,
        "start_timestamp": "t0",
        "end_timestamp": "t2",
        "holding_duration": 3,
        "trade_direction": "Short",
        "segment_type": "下跌分片",
        "trend_type": "趋势跟随",
        "turnover": pytest.approx(0.5),
        "reward_sum": pytest.approx(6.0),
        "mean_position": pytest.approx(-2 / 3),
        "mean_abs_position": pytest.approx(2 / 3),
        "long_step": 0,
        "short_step": 2,
        "flat_step": 1,
        "long_reward_sum": pytest.approx(0.0),
        "short_reward_sum": pytest.approx(3.0),
        "flat_reward_sum": pytest.approx(3.0),
    }
    assert reversion["trend_type"] == "趋势回归"
    assert reversion["reward_sum"] == pytest.approx(9.0)


def test_initial_nonzero_position_is_a_lifecycle():
    detail = _detail_frame(
        [(-1, -1), (-1, 0)],
        rewards=[2, 3],
        label="label_1",
    )

    lifecycle = extract_trade_lifecycles(detail, max_holding_number=2)[0]

    assert lifecycle["start_timestep"] == 0
    assert lifecycle["end_timestep"] == 1
    assert lifecycle["holding_duration"] == 2
    assert lifecycle["reward_sum"] == pytest.approx(5.0)


def test_reversal_reward_is_not_duplicated_between_lifecycles():
    detail = _detail_frame(
        [(0, 1), (1, -1), (-1, 0)],
        rewards=[1, 10, 2],
        label="label_4",
    )

    first, second = extract_trade_lifecycles(detail, max_holding_number=2)

    assert first["reward_sum"] == pytest.approx(11.0)
    assert second["reward_sum"] == pytest.approx(2.0)
    assert first["reward_sum"] + second["reward_sum"] == pytest.approx(13.0)


def test_analysis_result_groups_lifecycles_by_df_path_and_trend_type():
    detail = _detail_frame(
        [(0, -1), (-1, 0), (0, -1), (-1, 0)],
        rewards=[1, 2, 3, 4],
        label="label_0",
    )
    lifecycle_rows = extract_trade_lifecycles(detail, max_holding_number=2)
    lifecycle_frame = pl.DataFrame(lifecycle_rows)

    result = build_analysis_result(lifecycle_frame, max_holding_number=2)

    assert len(result) == 1
    record = result[0]
    assert record["trend_type"] == "趋势跟随"
    assert record["contract"] == ["fu2505"]
    assert record["df_path"] == ["fu2505/label_0/df_0.feather"]
    assert record["reward_sum"] == pytest.approx([10.0])
    assert record["df_length"] == [4]
    assert record["short_step_ratio"] == pytest.approx([0.5])
    assert record["flat_step_ratio"] == pytest.approx([0.5])
    assert record["limit_down_step_ratio"] == [1.0]
    assert record["limit_down_short_reward_sum"] == pytest.approx([4.0])
    assert record["limit_down_reverse_long_ratio"] == [0.0]


def test_generate_epoch_artifacts_uses_required_paths_and_no_manifest(tmp_path):
    epoch_dir = (
        tmp_path
        / "fu"
        / "experiment"
        / "weights_advantage_pretrain"
        / "epoch_7"
    )
    epoch_dir.mkdir(parents=True)
    chinese_detail = _detail_frame(
        [(0, 1), (1, 0)], rewards=[2, 3], label="label_6"
    ).rename(
        {
            "label": "标签",
            "df_path": "数据文件",
            "initial_action": "初始动作",
            "bin_index": "分箱索引",
            "timestep": "时间步",
            "timestamp": "时间戳",
            "position_before": "执行前仓位",
            "position_after": "执行后仓位",
            "step_reward": "单步奖励",
        }
    )
    chinese_detail.write_csv(epoch_dir / "trading_action_detail_epoch_7.csv")

    lifecycle_path, analysis_path, analysis_csv_path = generate_epoch_artifacts(
        result_path=tmp_path,
        dataset_name="fu",
        experiment_name="experiment",
        epoch=7,
        max_holding_number=2,
    )

    assert lifecycle_path == epoch_dir / "agent_trade_lifecycle_detail_7.csv"
    assert analysis_path == epoch_dir / "analysis_result_with_type.npy"
    assert analysis_csv_path == epoch_dir / "analysis_result_with_type.csv"
    assert pl.read_csv(lifecycle_path).columns == LIFECYCLE_COLUMNS
    saved = np.load(analysis_path, allow_pickle=True)
    assert saved.shape == (1,)
    assert saved[0]["trend_type"] == "趋势跟随"
    assert saved[0]["limit_up_step_ratio"] == [1.0]
    assert not (epoch_dir / "selection_manifest.json").exists()
    assert analysis_csv_path.exists()
    csv_df = pl.read_csv(analysis_csv_path)
    assert "标签" in csv_df.columns
    assert "趋势类型" in csv_df.columns
    assert "数据文件" in csv_df.columns
