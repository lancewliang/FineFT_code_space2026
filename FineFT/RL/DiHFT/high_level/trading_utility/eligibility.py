from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class EligibilityPolicy:
    minimum_episodes: int = 150
    minimum_per_quintile: int = 30
    minimum_contracts: int = 3
    minimum_trading_day_blocks: int = 20
    bootstrap_iterations: int = 1000
    alpha: float = 0.05
    fdr_alpha: float = 0.05
    minimum_spearman: float = 0.80
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.minimum_episodes < 1 or self.minimum_per_quintile < 1:
            raise ValueError("eligibility sample minimums must be positive")
        if self.bootstrap_iterations < 20:
            raise ValueError("bootstrap_iterations must be at least 20")


@dataclass(frozen=True)
class ChannelEligibility:
    slot_id: int
    direction: int
    eligible: bool
    failure_reasons: tuple[str, ...]
    episode_count: int
    contract_count: int
    trading_day_block_count: int
    quintile_counts: tuple[int, ...]
    win_rates: tuple[float, ...]
    payoff_ratios: tuple[float, ...]
    win_rate_spearman: float
    payoff_ratio_spearman: float
    mean_net_pnl_bps: float
    mean_net_pnl_lcb95_bps: float
    raw_p_value: float
    fdr_q_value: float
    win_rate_trend_lcb95: float
    payoff_ratio_trend_lcb95: float
    residual_q05: float


@dataclass(frozen=True)
class EligibilityReport:
    channels: dict[str, ChannelEligibility]
    policy: EligibilityPolicy
    evidence_metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: str | Path) -> Path:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        channels = {
            key: asdict(value) for key, value in sorted(self.channels.items())
        }
        core = {
            "schema_version": 1,
            "policy": asdict(self.policy),
            "channels": channels,
            "residual_definition": "realized_net_pnl_bps_minus_calibrated_utility_bps",
            "bootstrap_block": "contract_x_TradingDay",
            "multiple_testing": "Benjamini-Hochberg FDR",
            "evidence_metadata": dict(self.evidence_metadata),
        }
        artifact_id = hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        payload = {**core, "artifact_id": artifact_id}
        path = target / "utility_channel_eligibility.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EligibilityReport":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported utility eligibility schema_version")
        core = {key: value for key, value in payload.items() if key != "artifact_id"}
        expected_artifact_id = hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if payload.get("artifact_id") != expected_artifact_id:
            raise ValueError("utility eligibility artifact_id mismatch")
        policy = EligibilityPolicy(**payload["policy"])
        channels = {
            key: ChannelEligibility(
                **{
                    **value,
                    "failure_reasons": tuple(value["failure_reasons"]),
                    "quintile_counts": tuple(value["quintile_counts"]),
                    "win_rates": tuple(value["win_rates"]),
                    "payoff_ratios": tuple(value["payoff_ratios"]),
                }
            )
            for key, value in payload["channels"].items()
        }
        return cls(
            channels=channels,
            policy=policy,
            evidence_metadata=dict(payload.get("evidence_metadata", {})),
        )


def evaluate_channel_eligibility(
    frame: pd.DataFrame,
    *,
    policy: EligibilityPolicy | None = None,
) -> EligibilityReport:
    active_policy = policy or EligibilityPolicy()
    data = _validated_frame(frame)
    evaluated: dict[str, ChannelEligibility] = {}
    for (slot_id, direction), channel_frame in data.groupby(
        ["slot_id", "direction"], sort=True
    ):
        key = f"{int(slot_id)}:{int(direction)}"
        evaluated[key] = _evaluate_one(
            channel_frame.reset_index(drop=True), active_policy
        )
    q_values = _benjamini_hochberg(
        {key: result.raw_p_value for key, result in evaluated.items()}
    )
    finalized: dict[str, ChannelEligibility] = {}
    for key, result in evaluated.items():
        reasons = list(result.failure_reasons)
        q_value = q_values[key]
        if q_value > active_policy.fdr_alpha:
            reasons.append("fdr_significance_failed")
        eligible = not reasons
        finalized[key] = replace(
            result,
            eligible=eligible,
            failure_reasons=tuple(dict.fromkeys(reasons)),
            fdr_q_value=q_value,
        )
    return EligibilityReport(channels=finalized, policy=active_policy)


def _validated_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "slot_id",
        "direction",
        "contract",
        "trading_day",
        "utility_calibrated_bps",
        "net_pnl_bps",
        "win",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"eligibility input is missing columns: {missing}")
    data = frame.copy()
    data["trading_day"] = pd.to_datetime(data["trading_day"]).dt.normalize()
    numeric = data[["utility_calibrated_bps", "net_pnl_bps"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("eligibility scores and outcomes must be finite")
    data["win"] = data["win"].astype(bool)
    data["block_id"] = (
        data["contract"].astype(str)
        + "|"
        + data["trading_day"].dt.strftime("%Y-%m-%d")
    )
    return data


def _evaluate_one(
    frame: pd.DataFrame, policy: EligibilityPolicy
) -> ChannelEligibility:
    slot_id = int(frame["slot_id"].iloc[0])
    direction = int(frame["direction"].iloc[0])
    episode_count = len(frame)
    contract_count = int(frame["contract"].nunique())
    block_count = int(frame["block_id"].nunique())
    reasons: list[str] = []
    if episode_count < policy.minimum_episodes:
        reasons.append("insufficient_episodes")
    if contract_count < policy.minimum_contracts:
        reasons.append("insufficient_contracts")
    if block_count < policy.minimum_trading_day_blocks:
        reasons.append("insufficient_trading_day_blocks")

    ranked = frame["utility_calibrated_bps"].rank(method="first")
    frame = frame.copy()
    try:
        frame["quintile"] = pd.qcut(ranked, 5, labels=False) + 1
    except ValueError:
        frame["quintile"] = 0
        reasons.append("quintile_assignment_failed")
    quintile_counts: list[int] = []
    win_rates: list[float] = []
    payoff_ratios: list[float] = []
    undefined_payoff = False
    for quintile in range(1, 6):
        group = frame[frame["quintile"] == quintile]
        quintile_counts.append(len(group))
        if len(group) < policy.minimum_per_quintile:
            reasons.append("insufficient_quintile_samples")
        win_rates.append(float(group["win"].mean()) if len(group) else float("nan"))
        ratio = _payoff_ratio(group["net_pnl_bps"].to_numpy(dtype=float))
        if ratio is None:
            undefined_payoff = True
            payoff_ratios.append(float("nan"))
        else:
            payoff_ratios.append(ratio)
    if undefined_payoff:
        reasons.append("undefined_payoff_ratio")

    win_strict = _strictly_increasing(win_rates)
    payoff_strict = _strictly_increasing(payoff_ratios)
    if not win_strict:
        reasons.append("win_rate_not_strictly_increasing")
    if not payoff_strict:
        reasons.append("payoff_ratio_not_strictly_increasing")
    axis = np.arange(1, 6, dtype=float)
    win_spearman = _safe_spearman(axis, win_rates)
    payoff_spearman = _safe_spearman(axis, payoff_ratios)
    if win_spearman < policy.minimum_spearman:
        reasons.append("win_rate_spearman_failed")
    if payoff_spearman < policy.minimum_spearman:
        reasons.append("payoff_ratio_spearman_failed")

    rng = np.random.default_rng(policy.random_seed + slot_id * 101 + direction * 17)
    bootstrap = _block_bootstrap(frame, policy.bootstrap_iterations, rng)
    mean_lcb = float(np.quantile(bootstrap["means"], policy.alpha))
    raw_p = float(
        (1 + np.sum(np.asarray(bootstrap["means"]) <= 0.0))
        / (len(bootstrap["means"]) + 1)
    )
    if mean_lcb <= 0.0:
        reasons.append("mean_net_pnl_lcb_not_positive")
    win_trends = bootstrap["win_trends"]
    payoff_trends = bootstrap["payoff_trends"]
    win_trend_lcb = (
        float(np.quantile(win_trends, policy.alpha)) if win_trends else float("-inf")
    )
    payoff_trend_lcb = (
        float(np.quantile(payoff_trends, policy.alpha))
        if payoff_trends
        else float("-inf")
    )
    if win_trend_lcb <= 0.0:
        reasons.append("win_rate_trend_lcb_not_positive")
    if payoff_trend_lcb <= 0.0:
        reasons.append("payoff_ratio_trend_lcb_not_positive")

    residual = (
        frame["net_pnl_bps"].to_numpy(dtype=float)
        - frame["utility_calibrated_bps"].to_numpy(dtype=float)
    )
    return ChannelEligibility(
        slot_id=slot_id,
        direction=direction,
        eligible=False,
        failure_reasons=tuple(dict.fromkeys(reasons)),
        episode_count=episode_count,
        contract_count=contract_count,
        trading_day_block_count=block_count,
        quintile_counts=tuple(quintile_counts),
        win_rates=tuple(win_rates),
        payoff_ratios=tuple(payoff_ratios),
        win_rate_spearman=win_spearman,
        payoff_ratio_spearman=payoff_spearman,
        mean_net_pnl_bps=float(frame["net_pnl_bps"].mean()),
        mean_net_pnl_lcb95_bps=mean_lcb,
        raw_p_value=raw_p,
        fdr_q_value=1.0,
        win_rate_trend_lcb95=win_trend_lcb,
        payoff_ratio_trend_lcb95=payoff_trend_lcb,
        residual_q05=float(np.quantile(residual, 0.05)),
    )


def _block_bootstrap(
    frame: pd.DataFrame,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    blocks = [group for _, group in frame.groupby("block_id", sort=False)]
    means: list[float] = []
    win_trends: list[float] = []
    payoff_trends: list[float] = []
    axis = np.arange(1, 6, dtype=float)
    for _ in range(iterations):
        sampled_indices = rng.integers(0, len(blocks), size=len(blocks))
        sample = pd.concat([blocks[index] for index in sampled_indices], ignore_index=True)
        means.append(float(sample["net_pnl_bps"].mean()))
        win_rates: list[float] = []
        payoff_ratios: list[float] = []
        valid = True
        for quintile in range(1, 6):
            group = sample[sample["quintile"] == quintile]
            if group.empty:
                valid = False
                break
            win_rates.append(float(group["win"].mean()))
            ratio = _payoff_ratio(group["net_pnl_bps"].to_numpy(dtype=float))
            if ratio is None:
                valid = False
                break
            payoff_ratios.append(ratio)
        if valid:
            win_trends.append(float(np.polyfit(axis, win_rates, 1)[0]))
            payoff_trends.append(float(np.polyfit(axis, payoff_ratios, 1)[0]))
    return {
        "means": means,
        "win_trends": win_trends,
        "payoff_trends": payoff_trends,
    }


def _payoff_ratio(values: np.ndarray) -> float | None:
    winners = values[values > 0.0]
    losers = values[values < 0.0]
    if len(winners) == 0 or len(losers) == 0:
        return None
    denominator = abs(float(np.mean(losers)))
    if denominator <= 0.0:
        return None
    ratio = float(np.mean(winners)) / denominator
    return ratio if np.isfinite(ratio) else None


def _strictly_increasing(values: list[float]) -> bool:
    return all(
        np.isfinite(left) and np.isfinite(right) and left < right
        for left, right in zip(values, values[1:])
    )


def _safe_spearman(x: np.ndarray, y: list[float]) -> float:
    values = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(values)):
        return float("-inf")
    coefficient = float(spearmanr(x, values).statistic)
    return coefficient if np.isfinite(coefficient) else float("-inf")


def _benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    q_values: dict[str, float] = {}
    running = 1.0
    for rank, (key, p_value) in reversed(list(enumerate(ordered, start=1))):
        running = min(running, float(p_value) * count / rank)
        q_values[key] = min(1.0, running)
    return q_values
