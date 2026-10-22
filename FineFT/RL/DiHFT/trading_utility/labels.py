from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class CandidateObservation:
    contract: str
    trading_day: str
    event_time: str
    step_index: int
    slot_id: int
    current_position: float
    target_position: float
    prices: Sequence[float]
    fold_ids: Sequence[int | str]
    commission_bps: float
    slippage_bps: float
    impact_bps: float
    candidate_id: str
    input_valid: bool = True
    feature_values: tuple[float, ...] = ()


@dataclass(frozen=True)
class CandidateEpisode:
    episode_id: str
    contract: str
    trading_day: str
    event_time: str
    step_index: int
    slot_id: int
    direction: int
    candidate_id: str
    transition: str
    current_position: float
    target_position: float
    denied_baseline_position: float
    incremental_position: float
    horizon: int
    entry_price: float
    exit_price: float
    gross_pnl_bps: float
    friction_bps: float
    net_pnl_bps: float
    downside_loss_bps: float
    mae_bps: float
    win: bool
    feature_values: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_values"] = json.dumps(list(self.feature_values))
        return payload


@dataclass(frozen=True)
class CounterfactualDataset:
    episodes: tuple[CandidateEpisode, ...]
    exclusion_counts: dict[str, int]
    horizon: int
    friction_stress: float
    feature_version: str = "utility_features_v1"
    candidate_rule_version: str = "all_available_risk_increases_v1"
    cost_model_version: str = "commission_slippage_impact_bps_v1"
    baseline_rule_version: str = "open_flat_add_hold_reversal_flat_v1"
    input_asset_hashes: tuple[str, ...] = ()

    @property
    def dataset_hash(self) -> str:
        canonical = json.dumps(
            [episode.to_dict() for episode in self.episodes],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def write(self, output_dir: str | Path) -> dict[str, Path]:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        episodes_path = target / "candidate_episodes.csv"
        manifest_path = target / "candidate_episode_manifest.json"
        rows = [episode.to_dict() for episode in self.episodes]
        fieldnames = list(rows[0]) if rows else [field.name for field in CandidateEpisode.__dataclass_fields__.values()]
        with episodes_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "schema_version": 1,
            "horizon": self.horizon,
            "friction_stress": self.friction_stress,
            "feature_version": self.feature_version,
            "candidate_rule_version": self.candidate_rule_version,
            "cost_model_version": self.cost_model_version,
            "baseline_rule_version": self.baseline_rule_version,
            "episode_count": len(self.episodes),
            "exclusion_counts": dict(sorted(self.exclusion_counts.items())),
            "cost_accounting": "gross_pnl_minus_once_stressed_commission_slippage_impact",
            "counterfactual_only": True,
            "data_range": {
                "first_trading_day": min(
                    (episode.trading_day for episode in self.episodes), default=None
                ),
                "last_trading_day": max(
                    (episode.trading_day for episode in self.episodes), default=None
                ),
            },
            "input_asset_hashes": list(self.input_asset_hashes),
            "dataset_hash": self.dataset_hash,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return {"episodes": episodes_path, "manifest": manifest_path}


class CounterfactualEpisodeBuilder:
    """Build after-cost labels for risk-increasing candidate transitions."""

    def __init__(
        self,
        horizon: int,
        friction_stress: float = 1.0,
        *,
        feature_version: str = "utility_features_v1",
        candidate_rule_version: str = "all_available_risk_increases_v1",
        cost_model_version: str = "commission_slippage_impact_bps_v1",
        baseline_rule_version: str = "open_flat_add_hold_reversal_flat_v1",
        input_asset_hashes: Sequence[str] = (),
    ) -> None:
        self.horizon = int(horizon)
        self.friction_stress = float(friction_stress)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.friction_stress <= 0.0:
            raise ValueError("friction_stress must be positive")
        self.feature_version = str(feature_version)
        self.candidate_rule_version = str(candidate_rule_version)
        self.cost_model_version = str(cost_model_version)
        self.baseline_rule_version = str(baseline_rule_version)
        self.input_asset_hashes = tuple(sorted(str(value) for value in input_asset_hashes))

    def build(
        self, observations: Iterable[CandidateObservation]
    ) -> CounterfactualDataset:
        episodes: list[CandidateEpisode] = []
        exclusions: Counter[str] = Counter()
        for observation in observations:
            episode, reason = self._build_one(observation)
            if episode is None:
                exclusions[reason] += 1
            else:
                episodes.append(episode)
        episodes.sort(key=lambda item: (item.contract, item.step_index, item.candidate_id))
        return CounterfactualDataset(
            episodes=tuple(episodes),
            exclusion_counts=dict(exclusions),
            horizon=self.horizon,
            friction_stress=self.friction_stress,
            feature_version=self.feature_version,
            candidate_rule_version=self.candidate_rule_version,
            cost_model_version=self.cost_model_version,
            baseline_rule_version=self.baseline_rule_version,
            input_asset_hashes=self.input_asset_hashes,
        )

    def _build_one(
        self, observation: CandidateObservation
    ) -> tuple[CandidateEpisode | None, str]:
        required_length = self.horizon + 1
        if len(observation.prices) < required_length:
            return None, "incomplete_horizon"
        if len(observation.fold_ids) < required_length:
            return None, "incomplete_horizon"
        if len(set(observation.fold_ids[:required_length])) != 1:
            return None, "cross_fold_episode"
        if not observation.input_valid:
            return None, "invalid_input"

        transition, denied_baseline, incremental = self._transition(
            float(observation.current_position), float(observation.target_position)
        )
        if transition is None:
            return None, "not_risk_increase"

        prices = [float(value) for value in observation.prices[:required_length]]
        entry_price = prices[0]
        if entry_price <= 0.0 or any(price <= 0.0 for price in prices):
            return None, "invalid_price"
        direction = 1 if incremental > 0.0 else -1
        exposure = abs(incremental)
        path_returns = [
            direction * exposure * ((price / entry_price) - 1.0) * 10_000.0
            for price in prices[1:]
        ]
        gross_pnl_bps = path_returns[-1]
        raw_friction = (
            float(observation.commission_bps)
            + float(observation.slippage_bps)
            + float(observation.impact_bps)
        )
        friction_bps = exposure * raw_friction * self.friction_stress
        net_pnl_bps = gross_pnl_bps - friction_bps
        mae_bps = max(0.0, -min(path_returns))
        identity = "|".join(
            [
                observation.contract,
                observation.trading_day,
                str(observation.step_index),
                str(observation.slot_id),
                observation.candidate_id,
                transition,
                str(observation.target_position),
            ]
        )
        episode_id = "utility_" + hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()[:20]
        return (
            CandidateEpisode(
                episode_id=episode_id,
                contract=str(observation.contract),
                trading_day=str(observation.trading_day),
                event_time=str(observation.event_time),
                step_index=int(observation.step_index),
                slot_id=int(observation.slot_id),
                direction=direction,
                candidate_id=str(observation.candidate_id),
                transition=transition,
                current_position=float(observation.current_position),
                target_position=float(observation.target_position),
                denied_baseline_position=denied_baseline,
                incremental_position=incremental,
                horizon=self.horizon,
                entry_price=entry_price,
                exit_price=prices[-1],
                gross_pnl_bps=gross_pnl_bps,
                friction_bps=friction_bps,
                net_pnl_bps=net_pnl_bps,
                downside_loss_bps=max(0.0, -net_pnl_bps),
                mae_bps=mae_bps,
                win=net_pnl_bps > 0.0,
                feature_values=tuple(float(value) for value in observation.feature_values),
            ),
            "none",
        )

    @staticmethod
    def _transition(
        current_position: float, target_position: float
    ) -> tuple[str | None, float, float]:
        if current_position == 0.0 and target_position != 0.0:
            return "open", 0.0, target_position
        if current_position * target_position > 0.0 and abs(target_position) > abs(
            current_position
        ):
            return (
                "same_direction_add",
                current_position,
                target_position - current_position,
            )
        if current_position * target_position < 0.0:
            return "reversal", 0.0, target_position
        return None, current_position, 0.0
