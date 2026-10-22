from __future__ import annotations

import json

import pytest

from RL.DiHFT.trading_utility.labels import (
    CandidateObservation,
    CounterfactualEpisodeBuilder,
)


def _observation(**overrides) -> CandidateObservation:
    values = {
        "contract": "fu2509",
        "trading_day": "2025-01-02",
        "event_time": "2025-01-02 09:30:00",
        "step_index": 10,
        "slot_id": 2,
        "current_position": 0.0,
        "target_position": 1.0,
        "prices": (100.0, 99.0, 101.0, 102.0),
        "fold_ids": (1, 1, 1, 1),
        "commission_bps": 2.0,
        "slippage_bps": 3.0,
        "impact_bps": 1.0,
        "candidate_id": "agent-a",
    }
    values.update(overrides)
    return CandidateObservation(**values)


def test_open_label_charges_stressed_friction_once() -> None:
    dataset = CounterfactualEpisodeBuilder(horizon=3, friction_stress=1.5).build(
        [_observation()]
    )

    episode = dataset.episodes[0]
    assert episode.transition == "open"
    assert episode.denied_baseline_position == 0.0
    assert episode.gross_pnl_bps == pytest.approx(200.0)
    assert episode.friction_bps == pytest.approx(9.0)
    assert episode.net_pnl_bps == pytest.approx(191.0)
    assert episode.mae_bps == pytest.approx(100.0)
    assert episode.win is True


def test_add_and_reversal_labels_only_measure_new_risk_against_denied_baseline() -> None:
    builder = CounterfactualEpisodeBuilder(horizon=3)

    add = builder.build(
        [_observation(current_position=1.0, target_position=2.0)]
    ).episodes[0]
    reversal = builder.build(
        [
            _observation(
                current_position=1.0,
                target_position=-1.0,
                prices=(100.0, 101.0, 99.0, 98.0),
            )
        ]
    ).episodes[0]

    assert add.transition == "same_direction_add"
    assert add.denied_baseline_position == 1.0
    assert add.incremental_position == 1.0
    assert reversal.transition == "reversal"
    assert reversal.denied_baseline_position == 0.0
    assert reversal.incremental_position == -1.0
    assert reversal.gross_pnl_bps == pytest.approx(200.0)


def test_incomplete_and_cross_fold_episodes_are_excluded(tmp_path) -> None:
    builder = CounterfactualEpisodeBuilder(horizon=3)
    dataset = builder.build(
        [
            _observation(prices=(100.0, 101.0)),
            _observation(step_index=11, fold_ids=(1, 1, 2, 2)),
            _observation(step_index=12),
        ]
    )

    written = dataset.write(tmp_path)
    manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))

    assert len(dataset.episodes) == 1
    assert dataset.exclusion_counts == {
        "incomplete_horizon": 1,
        "cross_fold_episode": 1,
    }
    assert manifest["episode_count"] == 1
    assert manifest["horizon"] == 3
    assert manifest["dataset_hash"]
    assert written["episodes"].read_text(encoding="utf-8").count("fu2509") == 1
