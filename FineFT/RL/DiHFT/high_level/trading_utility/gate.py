from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from RL.DiHFT.trading_utility.eligibility import EligibilityReport
from RL.DiHFT.trading_utility.model_bundle import UtilityModelBundle


@dataclass(frozen=True)
class UtilityAdmissionDecision:
    admitted: bool
    transition: str
    candidate_action: int
    denied_baseline_action: int
    final_action: int
    channel_eligible: bool | None
    reject_stage: str | None
    reject_reason: str | None
    utility_raw_bps: float | None
    utility_calibrated_bps: float | None
    utility_lower_bps: float | None
    win_probability_calibrated: float | None
    utility_artifact_id: str | None


class TradingUtilityGate:
    """Admit risk increases only when local evidence and utility are positive."""

    def __init__(
        self,
        bundle: UtilityModelBundle | Any,
        calibrator: Any,
        eligibility: EligibilityReport,
    ) -> None:
        self.bundle = bundle
        self.calibrator = calibrator
        self.eligibility = eligibility

    def decide(
        self,
        *,
        slot_id: int,
        direction: int,
        features: Mapping[str, float],
        candidate_action: int,
        flat_action: int,
        hold_action: int,
        current_position: float,
        target_position: float,
    ) -> UtilityAdmissionDecision:
        transition, denied_action = self._transition_and_baseline(
            current_position=float(current_position),
            target_position=float(target_position),
            flat_action=int(flat_action),
            hold_action=int(hold_action),
        )
        if transition == "no_risk_increase":
            return UtilityAdmissionDecision(
                admitted=True,
                transition=transition,
                candidate_action=int(candidate_action),
                denied_baseline_action=int(candidate_action),
                final_action=int(candidate_action),
                channel_eligible=None,
                reject_stage=None,
                reject_reason=None,
                utility_raw_bps=None,
                utility_calibrated_bps=None,
                utility_lower_bps=None,
                win_probability_calibrated=None,
                utility_artifact_id=getattr(self.bundle, "artifact_id", None),
            )

        candidate_direction = 1 if target_position > 0.0 else -1
        if int(direction) != candidate_direction:
            raise ValueError("candidate direction does not match target position")
        channel_key = f"{int(slot_id)}:{candidate_direction}"
        channel = self.eligibility.channels.get(channel_key)
        if channel is None:
            return self._rejection(
                transition=transition,
                candidate_action=candidate_action,
                denied_action=denied_action,
                stage="channel_eligibility",
                reason="channel_missing",
                channel_eligible=False,
            )
        if not channel.eligible:
            return self._rejection(
                transition=transition,
                candidate_action=candidate_action,
                denied_action=denied_action,
                stage="channel_eligibility",
                reason=(
                    channel.failure_reasons[0]
                    if channel.failure_reasons
                    else "channel_ineligible"
                ),
                channel_eligible=False,
            )

        prediction = self.bundle.predict_one(features)
        calibrated_utility = float(
            self.calibrator.predict_utility([prediction.raw_utility_bps])[0]
        )
        calibrated_win = float(
            self.calibrator.predict_win_probability([prediction.win_probability])[0]
        )
        utility_lower = calibrated_utility + float(channel.residual_q05)
        if utility_lower <= 0.0:
            return UtilityAdmissionDecision(
                admitted=False,
                transition=transition,
                candidate_action=int(candidate_action),
                denied_baseline_action=int(denied_action),
                final_action=int(denied_action),
                channel_eligible=True,
                reject_stage="utility_lower_bound",
                reject_reason="utility_lower_not_positive",
                utility_raw_bps=float(prediction.raw_utility_bps),
                utility_calibrated_bps=calibrated_utility,
                utility_lower_bps=utility_lower,
                win_probability_calibrated=calibrated_win,
                utility_artifact_id=getattr(self.bundle, "artifact_id", None),
            )
        return UtilityAdmissionDecision(
            admitted=True,
            transition=transition,
            candidate_action=int(candidate_action),
            denied_baseline_action=int(denied_action),
            final_action=int(candidate_action),
            channel_eligible=True,
            reject_stage=None,
            reject_reason=None,
            utility_raw_bps=float(prediction.raw_utility_bps),
            utility_calibrated_bps=calibrated_utility,
            utility_lower_bps=utility_lower,
            win_probability_calibrated=calibrated_win,
            utility_artifact_id=getattr(self.bundle, "artifact_id", None),
        )

    def _rejection(
        self,
        *,
        transition: str,
        candidate_action: int,
        denied_action: int,
        stage: str,
        reason: str,
        channel_eligible: bool,
    ) -> UtilityAdmissionDecision:
        return UtilityAdmissionDecision(
            admitted=False,
            transition=transition,
            candidate_action=int(candidate_action),
            denied_baseline_action=int(denied_action),
            final_action=int(denied_action),
            channel_eligible=channel_eligible,
            reject_stage=stage,
            reject_reason=reason,
            utility_raw_bps=None,
            utility_calibrated_bps=None,
            utility_lower_bps=None,
            win_probability_calibrated=None,
            utility_artifact_id=getattr(self.bundle, "artifact_id", None),
        )

    @staticmethod
    def _transition_and_baseline(
        *,
        current_position: float,
        target_position: float,
        flat_action: int,
        hold_action: int,
    ) -> tuple[str, int]:
        if current_position == 0.0 and target_position != 0.0:
            return "open", flat_action
        if current_position * target_position > 0.0 and abs(target_position) > abs(
            current_position
        ):
            return "same_direction_add", hold_action
        if current_position * target_position < 0.0:
            return "reversal", flat_action
        return "no_risk_increase", candidate_baseline(hold_action, flat_action, current_position)


def candidate_baseline(hold_action: int, flat_action: int, current_position: float) -> int:
    return int(hold_action if current_position != 0.0 else flat_action)
