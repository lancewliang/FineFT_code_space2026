import json
from pathlib import Path

import polars as pl
import pytest

from RL.DiHFT.low_level.aggregate_agents_indexs import (
    MEAN_REVERSION,
    MANIFEST_FILENAME,
    OUTPUT_FILENAME,
    TREND_FOLLOWING,
    aggregate_detail_csvs,
    build_agent_manifest,
    build_episode_summary,
    classify_position_episodes,
)


def _trajectory(
    positions: list[tuple[float, float]],
    *,
    prices: list[float],
    rewards: list[float] | None = None,
    initial_action: int = 0,
    label: str = "label_1",
    df_path: str | None = None,
) -> list[dict[str, object]]:
    if rewards is None:
        rewards = [0.0] * len(positions)
    rows = []
    for timestep, ((before, after), price, reward) in enumerate(
        zip(positions, prices, rewards, strict=True)
    ):
        rows.append(
            {
                "epoch": 1,
                "label": label,
                "df_path": df_path or f"fu2505/{label}/df_0.feather",
                "initial_action": initial_action,
                "bin_index": 2,
                "timestep": timestep,
                "mark_price": price,
                "position_before": before,
                "position_after": after,
                "step_reward": reward,
            }
        )
    return rows


def test_classifies_opening_with_five_prior_prices_and_sums_episode_profit():
    rows = _trajectory(
        [(0, 0)] * 5 + [(0, 1), (1, 1), (1, 0)],
        prices=[100, 101, 102, 103, 104, 105, 106, 107],
        rewards=[0, 0, 0, 0, 0, -1, 2, 3],
    )

    episodes = classify_position_episodes(rows, epoch=1)

    assert len(episodes) == 1
    assert episodes[0].style == TREND_FOLLOWING
    assert episodes[0].profit == pytest.approx(4.0)
    assert (episodes[0].start_timestep, episodes[0].end_timestep) == (5, 7)


def test_added_exposure_votes_are_weighted_and_style_is_exclusive():
    rows = _trajectory(
        [(0, 0)] * 5
        + [(0, 1), (1, 1), (1, 1), (1, 1), (1, 3), (3, 0)],
        prices=[100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100],
    )

    episode = classify_position_episodes(rows, epoch=1)[0]

    assert episode.style == MEAN_REVERSION


def test_reversal_reward_belongs_only_to_closing_episode():
    rows = _trajectory(
        [(0, 0)] * 5
        + [(0, 1), (1, 1), (1, -1), (-1, -1), (-1, 0)],
        prices=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        rewards=[0, 0, 0, 0, 0, 1, 2, 10, 4, 5],
    )

    first, second = classify_position_episodes(rows, epoch=1)

    assert first.style == TREND_FOLLOWING
    assert first.profit == pytest.approx(13.0)
    assert second.style == MEAN_REVERSION
    assert second.profit == pytest.approx(9.0)
    summary = {row["分类"]: row for row in build_episode_summary(rows)}
    assert summary[TREND_FOLLOWING]["做多次数"] == 1
    assert summary[TREND_FOLLOWING]["做多平均利润"] == 13.0
    assert summary[MEAN_REVERSION]["做空次数"] == 1
    assert summary[MEAN_REVERSION]["做空平均利润"] == 9.0


def test_initial_position_stays_unclassified_without_later_increase():
    rows = _trajectory(
        [(1, 1)] * 6 + [(1, 0)],
        prices=[100, 101, 102, 103, 104, 105, 106],
        rewards=[1, 1, 1, 1, 1, 1, 1],
        initial_action=2,
    )

    episode = classify_position_episodes(rows, epoch=1)[0]

    assert episode.style is None
    assert episode.profit == pytest.approx(7.0)
    assert build_episode_summary(rows) == []


def test_open_episode_is_finalized_at_end_of_detail():
    rows = _trajectory(
        [(0, 0)] * 5 + [(0, -1), (-1, -1)],
        prices=[100, 101, 102, 103, 104, 105, 104],
        rewards=[0, 0, 0, 0, 0, -2, 3],
    )

    episode = classify_position_episodes(rows, epoch=1)[0]

    assert episode.end_timestep == 6
    assert episode.style == MEAN_REVERSION
    assert episode.profit == pytest.approx(1.0)


def _write_chinese_detail(path: Path) -> None:
    rows = _trajectory(
        [(0, 0)] * 5 + [(0, 1), (1, 0)],
        prices=[100, 101, 102, 103, 104, 105, 106],
        rewards=[0, 0, 0, 0, 0, -1, 3],
        label="label_4",
    )
    frame = pl.DataFrame(rows).drop("epoch").rename(
        {
            "label": "标签",
            "df_path": "数据文件",
            "initial_action": "初始动作",
            "bin_index": "分箱索引",
            "timestep": "时间步",
            "mark_price": "标记价格",
            "position_before": "执行前仓位",
            "position_after": "执行后仓位",
            "step_reward": "单步奖励",
        }
    )
    frame.write_csv(path)


def _write_label_semantics(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "label": "label_4",
                        "direction": "up",
                        "direction_sign": 1,
                        "strength": 1.0,
                        "description": "上涨",
                        "limit_state": "none",
                        "limit_state_sign": 0,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_aggregates_chinese_detail_csv_to_requested_schema(tmp_path):
    model_root = tmp_path / "models"
    epoch_dir = model_root / "epoch_1"
    epoch_dir.mkdir(parents=True)
    _write_chinese_detail(epoch_dir / "trading_action_detail_epoch_1.csv")
    output_dir = tmp_path / "analysis"
    semantics_path = tmp_path / "label_semantics.json"
    _write_label_semantics(semantics_path)

    output_path = aggregate_detail_csvs(
        model_root,
        output_dir,
        semantics_path,
        dataset_name="fu",
        experiment_name="test_experiment",
        epoch_start=1,
        epoch_end=1,
    )

    assert output_path == output_dir / OUTPUT_FILENAME
    result = pl.read_csv(output_path)
    assert result.columns == [
        "epoch",
        "label",
        "bin_index",
        "init_action",
        "分类",
        "分类次数",
        "利润",
        "做多次数",
        "做空次数",
        "做多平均利润",
        "做空平均利润",
    ]
    assert result.to_dicts() == [
        {
            "epoch": 1,
            "label": "label_4",
            "bin_index": 2,
            "init_action": 0,
            "分类": TREND_FOLLOWING,
            "分类次数": 1,
            "利润": 2.0,
            "做多次数": 1,
            "做空次数": 0,
            "做多平均利润": 2.0,
            "做空平均利润": 0.0,
        }
    ]
    manifest = json.loads((output_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["dataset_name"] == "fu"
    assert manifest["experiment_name"] == "test_experiment"
    label_result = manifest["labels"][0]
    assert label_result["label"] == "label_4"
    assert label_result["direction_sign"] == 1
    classifications = {
        item["classification"]: item for item in label_result["classifications"]
    }
    best_agent = classifications[TREND_FOLLOWING]["best_agent"]
    assert best_agent["epoch_path"].endswith("models/epoch_1")
    assert best_agent["bin_index"] == 2
    assert best_agent["score"] == 2.0
    assert best_agent["behavior_summary"]["selection_direction"] == "long"
    assert classifications[MEAN_REVERSION] == {
        "classification": MEAN_REVERSION,
        "total_classification_count": 0,
        "eligible_agent_count": 0,
        "best_agent": None,
        "second_best_agent": None,
    }


def test_manifest_ranks_distinct_agents_by_their_best_initial_action():
    summary_rows = [
        {
            "epoch": 1,
            "label": "label_2",
            "bin_index": 0,
            "init_action": 0,
            "分类": TREND_FOLLOWING,
            "分类次数": 3,
            "利润": 4.0,
            "做多次数": 1,
            "做空次数": 2,
            "做多平均利润": 20.0,
            "做空平均利润": 2.0,
        },
        {
            "epoch": 1,
            "label": "label_2",
            "bin_index": 0,
            "init_action": 1,
            "分类": TREND_FOLLOWING,
            "分类次数": 2,
            "利润": 10.0,
            "做多次数": 1,
            "做空次数": 1,
            "做多平均利润": 30.0,
            "做空平均利润": 10.0,
        },
        {
            "epoch": 2,
            "label": "label_2",
            "bin_index": 1,
            "init_action": 0,
            "分类": TREND_FOLLOWING,
            "分类次数": 4,
            "利润": 8.0,
            "做多次数": 2,
            "做空次数": 2,
            "做多平均利润": 50.0,
            "做空平均利润": 8.0,
        },
        {
            "epoch": 3,
            "label": "label_2",
            "bin_index": 2,
            "init_action": 0,
            "分类": TREND_FOLLOWING,
            "分类次数": 2,
            "利润": 100.0,
            "做多次数": 1,
            "做空次数": 1,
            "做多平均利润": 100.0,
            "做空平均利润": -1.0,
        },
    ]

    semantics = {"label_2": {"direction": "down", "direction_sign": -1}}
    manifest = build_agent_manifest(
        summary_rows,
        semantics,
        model_root=Path("models"),
        dataset_name="fu",
        experiment_name="experiment",
    )
    label_result = manifest["labels"][0]
    category = next(
        item
        for item in label_result["classifications"]
        if item["classification"] == TREND_FOLLOWING
    )

    assert category["total_classification_count"] == 11
    assert category["eligible_agent_count"] == 2
    best_agent = category["best_agent"]
    assert best_agent["epoch_path"] == "models/epoch_1"
    assert best_agent["bin_index"] == 0
    assert best_agent["score"] == 10.0
    assert best_agent["behavior_summary"]["selected_init_action"] == 1
    assert best_agent["behavior_summary"]["selection_direction"] == "short"
    assert best_agent["behavior_summary"]["classification_count"] == 5
    assert best_agent["behavior_summary"]["classification_count_ratio"] == pytest.approx(5 / 11)
    assert category["second_best_agent"]["epoch_path"] == "models/epoch_2"


def test_rejects_discontinuous_positions():
    rows = _trajectory(
        [(0, 1), (0, 0)],
        prices=[100, 101],
    )

    with pytest.raises(ValueError, match="positions are discontinuous"):
        classify_position_episodes(rows, epoch=1)
