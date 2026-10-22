from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from RL.DiHFT.trading_utility.eligibility import (
    EligibilityPolicy,
    evaluate_channel_eligibility,
)


def _eligible_channel(slot_id: int = 1, direction: int = 1) -> pd.DataFrame:
    rows = []
    win_counts = [6, 9, 12, 15, 18]
    winner_values = [2.0, 3.0, 5.0, 9.0, 17.0]
    row_index = 0
    for quintile, (wins, winner_value) in enumerate(
        zip(win_counts, winner_values), start=1
    ):
        for within in range(30):
            is_win = within < wins
            net = winner_value if is_win else -1.0
            score = float(quintile * 100 + within) / 100.0
            rows.append(
                {
                    "slot_id": slot_id,
                    "direction": direction,
                    "contract": f"fu{row_index % 3}",
                    "trading_day": pd.Timestamp("2024-11-01")
                    + pd.Timedelta(days=row_index % 25),
                    "utility_calibrated_bps": score,
                    "net_pnl_bps": net,
                    "win": is_win,
                }
            )
            row_index += 1
    return pd.DataFrame(rows)


def test_channel_passes_all_support_significance_and_monotonicity_gates(
    tmp_path: Path,
) -> None:
    report = evaluate_channel_eligibility(
        _eligible_channel(),
        policy=EligibilityPolicy(bootstrap_iterations=200, random_seed=5),
    )
    channel = report.channels["1:1"]
    manifest_path = report.save(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert channel.eligible is True
    assert channel.episode_count == 150
    assert channel.contract_count == 3
    assert channel.trading_day_block_count >= 20
    assert list(channel.win_rates) == sorted(channel.win_rates)
    assert list(channel.payoff_ratios) == sorted(channel.payoff_ratios)
    assert channel.win_rate_spearman >= 0.80
    assert channel.payoff_ratio_spearman >= 0.80
    assert channel.residual_q05 <= max(
        _eligible_channel()["net_pnl_bps"]
        - _eligible_channel()["utility_calibrated_bps"]
    )
    assert manifest["channels"]["1:1"]["eligible"] is True
    assert manifest["artifact_id"]


def test_insufficient_channel_is_disabled_without_pooling() -> None:
    sparse = _eligible_channel().iloc[:149].copy()

    report = evaluate_channel_eligibility(
        sparse,
        policy=EligibilityPolicy(bootstrap_iterations=50),
    )
    channel = report.channels["1:1"]

    assert channel.eligible is False
    assert "insufficient_episodes" in channel.failure_reasons


def test_undefined_payoff_ratio_fails_channel() -> None:
    frame = _eligible_channel()
    lowest_scores = frame.nsmallest(30, "utility_calibrated_bps").index
    frame.loc[lowest_scores, "net_pnl_bps"] = 2.0
    frame.loc[lowest_scores, "win"] = True

    report = evaluate_channel_eligibility(
        frame,
        policy=EligibilityPolicy(bootstrap_iterations=50),
    )

    assert report.channels["1:1"].eligible is False
    assert "undefined_payoff_ratio" in report.channels["1:1"].failure_reasons
