from __future__ import annotations

import numpy as np

from RL.DiHFT.high_level.trading_utility.eligibility import (
    ChannelEligibility,
    EligibilityPolicy,
    EligibilityReport,
)
from RL.DiHFT.high_level.trading_utility.gate import TradingUtilityGate
from RL.DiHFT.high_level.trading_utility.model_bundle import UtilityPrediction


class _Bundle:
    artifact_id = "bundle-1"
    feature_names = ("signal",)

    def __init__(self, raw_utility: float, win_probability: float = 0.01) -> None:
        self.raw_utility = raw_utility
        self.win_probability = win_probability

    def predict_one(self, features):
        return UtilityPrediction(
            mean_net_pnl_bps=self.raw_utility,
            cvar_loss_bps=0.0,
            mae_loss_bps=0.0,
            win_probability=self.win_probability,
            raw_utility_bps=self.raw_utility,
        )


class _IdentityCalibrator:
    def predict_utility(self, values):
        return np.asarray(values, dtype=float)

    def predict_win_probability(self, values):
        return np.asarray(values, dtype=float)


def _channel(*, eligible: bool, residual_q05: float = -1.0) -> ChannelEligibility:
    return ChannelEligibility(
        slot_id=2,
        direction=1,
        eligible=eligible,
        failure_reasons=() if eligible else ("insufficient_episodes",),
        episode_count=150,
        contract_count=3,
        trading_day_block_count=20,
        quintile_counts=(30, 30, 30, 30, 30),
        win_rates=(0.2, 0.3, 0.4, 0.5, 0.6),
        payoff_ratios=(1.0, 2.0, 3.0, 4.0, 5.0),
        win_rate_spearman=1.0,
        payoff_ratio_spearman=1.0,
        mean_net_pnl_bps=1.0,
        mean_net_pnl_lcb95_bps=0.1,
        raw_p_value=0.01,
        fdr_q_value=0.01,
        win_rate_trend_lcb95=0.01,
        payoff_ratio_trend_lcb95=0.01,
        residual_q05=residual_q05,
    )


def _gate(raw_utility: float, *, eligible: bool = True) -> TradingUtilityGate:
    report = EligibilityReport(
        channels={
            "2:1": _channel(eligible=eligible),
            "2:-1": replace_channel_direction(_channel(eligible=eligible), -1),
        },
        policy=EligibilityPolicy(bootstrap_iterations=20),
    )
    return TradingUtilityGate(_Bundle(raw_utility), _IdentityCalibrator(), report)


def replace_channel_direction(
    channel: ChannelEligibility, direction: int
) -> ChannelEligibility:
    values = dict(channel.__dict__)
    values["direction"] = direction
    return ChannelEligibility(**values)


def test_positive_lower_utility_admits_even_with_low_win_probability() -> None:
    decision = _gate(3.0).decide(
        slot_id=2,
        direction=1,
        features={"signal": 1.0},
        candidate_action=8,
        flat_action=4,
        hold_action=4,
        current_position=0.0,
        target_position=1.0,
    )

    assert decision.admitted is True
    assert decision.final_action == 8
    assert decision.win_probability_calibrated == 0.01
    assert decision.utility_lower_bps == 2.0


def test_rejected_open_add_and_reversal_use_distinct_safe_baselines() -> None:
    gate = _gate(0.5)

    rejected_open = gate.decide(
        slot_id=2,
        direction=1,
        features={"signal": 1.0},
        candidate_action=8,
        flat_action=4,
        hold_action=4,
        current_position=0.0,
        target_position=1.0,
    )
    rejected_add = gate.decide(
        slot_id=2,
        direction=1,
        features={"signal": 1.0},
        candidate_action=9,
        flat_action=4,
        hold_action=8,
        current_position=1.0,
        target_position=2.0,
    )
    rejected_reverse = gate.decide(
        slot_id=2,
        direction=-1,
        features={"signal": 1.0},
        candidate_action=0,
        flat_action=4,
        hold_action=8,
        current_position=1.0,
        target_position=-1.0,
    )

    assert rejected_open.final_action == 4
    assert rejected_open.transition == "open"
    assert rejected_add.final_action == 8
    assert rejected_add.transition == "same_direction_add"
    assert rejected_reverse.final_action == 4
    assert rejected_reverse.transition == "reversal"


def test_ineligible_channel_rejects_before_utility_threshold() -> None:
    decision = _gate(100.0, eligible=False).decide(
        slot_id=2,
        direction=1,
        features={"signal": 1.0},
        candidate_action=8,
        flat_action=4,
        hold_action=4,
        current_position=0.0,
        target_position=1.0,
    )

    assert decision.admitted is False
    assert decision.reject_stage == "channel_eligibility"
    assert decision.reject_reason == "insufficient_episodes"


def test_non_risk_increase_bypasses_utility_model() -> None:
    decision = _gate(-100.0).decide(
        slot_id=2,
        direction=1,
        features={"signal": 1.0},
        candidate_action=7,
        flat_action=4,
        hold_action=7,
        current_position=1.0,
        target_position=1.0,
    )

    assert decision.admitted is True
    assert decision.transition == "no_risk_increase"
    assert decision.utility_lower_bps is None
