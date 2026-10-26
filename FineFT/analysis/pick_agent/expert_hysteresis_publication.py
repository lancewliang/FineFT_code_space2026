"""Fail-closed step-8 publication for a selected expert-hysteresis Trial.

Step 8 consumes the explicit step-7 selection evidence.  It never scans trial
directories or ranks results: one COMPLETE Trial, its frozen configuration, and
its full replay are the only permitted source of publication diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import optuna
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ArtifactNames, AuditArtifactNames, HistoryArtifactNames
from common.routing_audit import RoutingRecord
from RL.DiHFT.high_level.frozen_experiment import (
    POLICY_FIELDS,
    RUNTIME_FIELDS,
    SCHEMA_VERSION,
    SEARCH_FIELDS,
    SELECTION_FIELDS,
    STRATEGY_VERSION,
)

SELECTION_EVIDENCE_FILENAME = "selection_evidence.json"
PUBLICATION_RECEIPT_FILENAME = "step8_publication_receipt.json"


def _identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read_json_value(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"required step-8 evidence is missing: {path}")
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _read_json(path: Path) -> dict[str, Any]:
    value = _read_json_value(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    return value


def _resolve_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty explicit path")
    return Path(value).expanduser().resolve()


def _same_path(left: Path, right: Path, *, name: str) -> None:
    if left != right:
        raise ValueError(f"{name} does not match the explicit step-7 evidence")


@dataclass(frozen=True, slots=True)
class SelectedReplay:
    """Validated immutable input of a step-8 publication."""

    evidence_path: Path
    frozen_config_path: Path
    replay_path: Path
    evidence: dict[str, Any]
    config: dict[str, Any]

    @property
    def config_hash(self) -> str:
        return str(self.config["config_hash"])


def load_selected_replay(
    selection_evidence: str | Path,
    *,
    frozen_config: str | Path | None = None,
    best_replay: str | Path | None = None,
) -> SelectedReplay:
    """Validate one selected Trial, without looking at any other Trial."""
    evidence_path = Path(selection_evidence).resolve()
    evidence = _read_json(evidence_path)
    evidence_hash = evidence.pop("evidence_hash", None)
    if evidence_hash != _identity(evidence):
        raise ValueError("step-7 selection evidence hash mismatch")
    if (
        evidence.get("schema_version") != SCHEMA_VERSION
        or evidence.get("document_type") != "expert_hysteresis_calibration_selection"
        or evidence.get("strategy_version") != STRATEGY_VERSION
        or evidence.get("stage") != "valid"
        or evidence.get("purpose") != "calibration_selection"
    ):
        raise ValueError("incompatible step-7 selection evidence")

    frozen_descriptor = evidence.get("frozen_config")
    replay_descriptor = evidence.get("best_replay")
    if not isinstance(frozen_descriptor, dict) or not isinstance(replay_descriptor, dict):
        raise ValueError("selection evidence lacks frozen config or best replay")
    frozen_path = _resolve_path(frozen_descriptor.get("path"), name="frozen_config.path")
    replay_path = _resolve_path(replay_descriptor.get("artifact_path"), name="best_replay.artifact_path")
    if frozen_config is not None:
        _same_path(Path(frozen_config).resolve(), frozen_path, name="--frozen-config")
    if best_replay is not None:
        _same_path(Path(best_replay).resolve(), replay_path, name="--best-replay")
    if not replay_path.is_dir():
        raise FileNotFoundError(f"selected best replay is missing: {replay_path}")
    if replay_descriptor.get("matches_lean_trial") is not True:
        raise ValueError("step-7 evidence does not certify the best replay")

    config = _read_json(frozen_path)
    config_hash = config.pop("config_hash", None)
    if config_hash != _identity(config):
        raise ValueError("frozen configuration hash mismatch")
    config["config_hash"] = config_hash
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("document_type") != "expert_hysteresis_frozen_config"
        or config.get("selected_on") != "calibration"
    ):
        raise ValueError("step 8 requires a calibration frozen expert configuration")
    if frozen_descriptor.get("config_hash") != config_hash:
        raise ValueError("selection evidence and frozen configuration hash differ")
    if config.get("experiment_identity") != evidence.get("experiment_identity"):
        raise ValueError("selection evidence and frozen configuration experiment identities differ")

    configuration = config.get("configuration")
    required_fields = set(POLICY_FIELDS + SEARCH_FIELDS + RUNTIME_FIELDS)
    if not isinstance(configuration, dict) or set(configuration) != required_fields:
        raise ValueError("frozen configuration is incomplete")
    if configuration.get("gating_strategy") != "expert_hysteresis":
        raise ValueError("frozen configuration does not select expert_hysteresis")
    selection = config.get("selection")
    selected_trial = evidence.get("selected_trial")
    if not isinstance(selection, dict) or set(selection) != set(SELECTION_FIELDS):
        raise ValueError("frozen selected Trial identity is incomplete")
    if not isinstance(selected_trial, dict):
        raise ValueError("selection evidence lacks selected Trial")
    evidence_selection_fields = {
        "number": "trial_number",
        "state": "trial_state",
        "objective_value": "objective_value",
    }
    for evidence_field, config_field in evidence_selection_fields.items():
        if selected_trial.get(evidence_field) != selection.get(config_field):
            raise ValueError(f"selected Trial differs between evidence and config: {config_field}")
    if selection["trial_state"] != "COMPLETE":
        raise ValueError("step 8 may publish only a COMPLETE selected Trial")
    expected_effective = {
        **{field: configuration[field] for field in SEARCH_FIELDS},
        "slope_hold_threshold": configuration["slope_hold_threshold"],
        "volatility_hold_threshold": configuration["volatility_hold_threshold"],
    }
    if selected_trial.get("effective_configuration") != expected_effective:
        raise ValueError("selected Trial effective configuration is incomplete or changed")
    _validate_study(evidence, selection, config)
    _validate_replay_identity(replay_path, evidence, config_hash)
    evidence["evidence_hash"] = evidence_hash
    return SelectedReplay(evidence_path, frozen_path, replay_path, evidence, config)


def _validate_study(evidence: dict[str, Any], selection: dict[str, Any], config: dict[str, Any]) -> None:
    study_evidence = evidence.get("study")
    if not isinstance(study_evidence, dict):
        raise ValueError("selection evidence lacks study identity")
    storage_path = _resolve_path(study_evidence.get("storage_path"), name="study.storage_path")
    if not storage_path.is_file():
        raise FileNotFoundError(f"selected study is missing: {storage_path}")
    expected_identity = _identity(
        {"experiment_identity": config["experiment_identity"], "study_name": selection["study_name"], "storage_path": str(storage_path)}
    )
    if (
        study_evidence.get("name") != selection["study_name"]
        or study_evidence.get("storage_identity") != selection["study_storage_identity"]
        or selection["study_storage_identity"] != expected_identity
    ):
        raise ValueError("study identity does not match the frozen selected Trial")
    study = optuna.load_study(study_name=selection["study_name"], storage=f"sqlite:///{storage_path}")
    if study.direction.name.lower() != selection["objective_direction"]:
        raise ValueError("Optuna study direction differs from frozen selection")
    matching = [trial for trial in study.trials if trial.number == selection["trial_number"]]
    if len(matching) != 1:
        raise ValueError("selected Trial is not uniquely present in its selected study")
    trial = matching[0]
    if trial.state.name != "COMPLETE" or trial.value is None:
        raise ValueError("selected Trial in study is not COMPLETE")
    if not math.isclose(float(trial.value), float(selection["objective_value"]), rel_tol=0, abs_tol=1e-12):
        raise ValueError("selected Trial objective differs from frozen selection")
    if trial.params != evidence["selected_trial"].get("parameters"):
        raise ValueError("selected Trial parameters differ from explicit step-7 evidence")


def _validate_replay_identity(replay_path: Path, evidence: dict[str, Any], config_hash: str) -> None:
    metadata_paths = [replay_path / AuditArtifactNames.AUDIT_METADATA_JSON]
    metadata_paths.extend(
        path / AuditArtifactNames.AUDIT_METADATA_JSON
        for path in (replay_path / "contracts").glob("*")
        if path.is_dir()
    )
    found = [path for path in metadata_paths if path.is_file()]
    if not found:
        raise ValueError("selected best replay has no audit metadata proving its configuration")
    for path in found:
        metadata = _read_json(path)
        if metadata.get("configuration_identity") != config_hash:
            raise ValueError(f"best replay configuration hash conflict: {path}")
        if metadata.get("experiment_identity") != evidence["experiment_identity"]:
            raise ValueError(f"best replay experiment identity conflict: {path}")
        if metadata.get("gating_strategy") != "expert_hysteresis":
            raise ValueError(f"best replay strategy conflict: {path}")


def _contracts(replay_path: Path) -> list[tuple[str, Path]]:
    root = replay_path / "contracts"
    contracts = (
        [(path.name, path) for path in sorted(root.iterdir()) if path.is_dir()]
        if root.is_dir()
        else [(replay_path.name, replay_path)]
    )
    if not contracts:
        raise ValueError("selected best replay has no contract artifacts")
    return contracts


def _market_frame(market_data_dir: Path, contract: str) -> pd.DataFrame:
    candidates = (market_data_dir / f"{contract}.feather", market_data_dir / f"{contract}.csv")
    for path in candidates:
        if path.is_file():
            frame = pd.read_feather(path) if path.suffix == ".feather" else pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(f"missing explicit market data for replay contract {contract}")
    if not {"timestamp", "mark_price"}.issubset(frame.columns):
        raise ValueError(f"market data lacks timestamp/mark_price for {contract}")
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise", utc=True)
    prices = pd.to_numeric(frame["mark_price"], errors="raise").to_numpy(dtype=float)
    if len(frame) == 0 or timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError(f"market timestamps are not monotonic and complete for {contract}")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"market prices are invalid for {contract}")
    result = frame.copy()
    result["timestamp"] = timestamps
    return result


def _array(path: Path, *, name: str, length: int | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"selected replay lacks {name}: {path}")
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1 or not np.isfinite(values).all() or (length is not None and len(values) != length):
        raise ValueError(f"invalid {name} history: {path}")
    return values


def _contract_payload(
    contract: str,
    artifact_dir: Path,
    market: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rewards = _array(artifact_dir / HistoryArtifactNames.REWARD_HISTORY_NPY, name="reward")
    steps = len(rewards)
    equities = _array(artifact_dir / HistoryArtifactNames.TOTAL_ASSET_HISTORY_NPY, name="equity", length=steps + 1)
    indices = _array(artifact_dir / HistoryArtifactNames.MARKET_INDEX_HISTORY_NPY, name="market index", length=steps + 1)
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"market index history must be integral: {contract}")
    indices = indices.astype(int)
    if (
        len(indices) == 0
        or indices[0] != 0
        or (np.diff(indices) < 0).any()
        or (np.diff(indices) > 1).any()
        or indices[-1] >= len(market)
    ):
        raise ValueError(f"market index history is not aligned for {contract}")
    if not np.allclose(np.diff(equities), rewards, rtol=0, atol=1e-8):
        raise ValueError(f"reward/equity history does not reconcile for {contract}")
    actions = _array(artifact_dir / HistoryArtifactNames.MICRO_ACTION_HISTORY_NPY, name="micro action", length=steps)
    if not np.issubdtype(actions.dtype, np.integer):
        raise ValueError(f"micro action history must be integral for {contract}")
    fee_path = artifact_dir / HistoryArtifactNames.COMMISSION_FEE_HISTORY_NPY
    fees = _array(fee_path, name="commission", length=steps) if fee_path.is_file() else np.zeros(steps)
    routing_rows = _read_json_value(artifact_dir / AuditArtifactNames.AUDIT_ROUTING_JSON)
    if not isinstance(routing_rows, list) or len(routing_rows) != steps:
        raise ValueError(f"routing audit must cover every replay step for {contract}")
    records = [RoutingRecord.from_dict(row) for row in routing_rows]
    expected_times = pd.to_datetime(market["timestamp"].iloc[indices[:-1]], utc=True).astype(str).tolist()
    for step, record in enumerate(records):
        record.validate()
        if record.contract != contract or record.step_index != step or record.event_time != expected_times[step]:
            raise ValueError(f"routing audit is not aligned to market step {step} for {contract}")

    prices = market["mark_price"].to_numpy(dtype=float)[indices]
    baseline = prices / prices[0] - 1.0
    strategy = equities / equities[0] - 1.0
    drawdown = 1.0 - equities / np.maximum.accumulate(equities)
    route = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(market["timestamp"].iloc[indices[:-1]], utc=True).to_numpy(),
            "dominant_slot": [row.dominant_slot for row in records],
            "active_slot": [row.active_slot_after for row in records],
            "candidate_slot": [row.candidate_slot_after for row in records],
            "candidate_count": [row.candidate_count_after for row in records],
            "volatility_score": [max(row.volatility_scores, default=np.nan) for row in records],
            "slope_score": [max(row.slope_scores, default=np.nan) for row in records],
            "volatility_entry": [row.volatility_entry_threshold for row in records],
            "slope_entry": [row.slope_entry_threshold for row in records],
            "volatility_hold": [row.volatility_holding_threshold for row in records],
            "slope_hold": [row.slope_holding_threshold for row in records],
            "routing_reason": [row.routing_reason for row in records],
            "transition_reason": [row.transition_reason for row in records],
            "incumbent_invalidated": [row.incumbent_invalidated for row in records],
        }
    )
    metrics = {
        "contract": contract,
        "steps": steps,
        "initial_equity": float(equities[0]),
        "net_pnl": float(equities[-1] - equities[0]),
        "tr": float(strategy[-1]),
        "buy_and_hold_tr": float(baseline[-1]),
        "mdd": float(drawdown.max()),
        "friction": float(fees.sum()),
        "trade_events": int(np.count_nonzero(np.diff(np.r_[actions[0], actions]))),
        "switch_events": int(sum(row.transition_reason == "expert_replaced" for row in records)),
        "invalidation_events": int(sum(row.incumbent_invalidated for row in records)),
        "rule_events": int(sum(row.routing_reason in {"entry_confirmation", "entry_threshold", "empty_model"} for row in records)),
        "empty_slot_events": int(sum(row.routing_reason == "empty_model" for row in records)),
        "risk_events": int(sum("risk" in row.action_reason.lower() or "stop" in row.action_reason.lower() for row in records)),
    }
    timeline = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(market["timestamp"].iloc[indices], utc=True).to_numpy(),
            "strategy_return": strategy,
            "buy_and_hold_return": baseline,
            "equity": equities,
            "drawdown": drawdown,
        }
    )
    return metrics, timeline.join(route.set_index("timestamp"), on="timestamp")


def _plot_contract(contract: str, timeline: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    axes[0].plot(timeline["timestamp"], timeline["strategy_return"] * 100, label="Strategy net return")
    axes[0].plot(timeline["timestamp"], timeline["buy_and_hold_return"] * 100, label="Buy & hold")
    axes[0].set_ylabel("Return (%)")
    axes[0].set_title(f"{contract}: selected COMPLETE Trial replay")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(timeline["timestamp"], timeline["dominant_slot"], label="Dominant Slot observed (not trade)")
    axes[1].plot(timeline["timestamp"], timeline["active_slot"], label="Active expert Slot")
    axes[1].plot(timeline["timestamp"], timeline["candidate_slot"], label="Pending candidate Slot", alpha=0.65)
    axes[1].plot(timeline["timestamp"], timeline["candidate_count"], label="Candidate confirmations", linestyle=":")
    axes[1].set_ylabel("Routing state")
    axes[1].legend(ncol=2)
    axes[1].grid(alpha=0.3)
    axes[2].plot(timeline["timestamp"], timeline["volatility_score"], label="Volatility score")
    axes[2].plot(timeline["timestamp"], timeline["slope_score"], label="Slope score")
    axes[2].plot(timeline["timestamp"], timeline["volatility_entry"], label="Volatility E", linestyle="--")
    axes[2].plot(timeline["timestamp"], timeline["volatility_hold"], label="Volatility H", linestyle=":")
    axes[2].plot(timeline["timestamp"], timeline["slope_entry"], label="Slope E", linestyle="--")
    axes[2].plot(timeline["timestamp"], timeline["slope_hold"], label="Slope H", linestyle=":")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("Score / threshold")
    axes[2].legend(ncol=3)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def publish_selected_replay(
    selected: SelectedReplay,
    *,
    market_data_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Publish validation diagnostics in a new, purpose-specific directory."""
    market_root = Path(market_data_dir).resolve()
    output = Path(output_dir).resolve()
    if not market_root.is_dir():
        raise FileNotFoundError(f"explicit validation market-data directory is missing: {market_root}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing step-8 publication: {output}")
    if output == selected.replay_path or selected.replay_path in output.parents:
        raise ValueError("step-8 publication must not write into the selected replay")
    if "final_result" in output.parts:
        raise ValueError("step-8 publication output must not use the step-10 final_result root")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        metrics: list[dict[str, Any]] = []
        for contract, artifacts in _contracts(selected.replay_path):
            contract_metrics, timeline = _contract_payload(contract, artifacts, _market_frame(market_root, contract))
            metrics.append(contract_metrics)
            timeline.to_csv(temporary / f"{contract}_aligned_timeline.csv", index=False)
            _plot_contract(contract, timeline, temporary / f"{contract}_routing_and_returns.png")
        metrics_frame = pd.DataFrame(metrics)
        metrics_frame.to_csv(temporary / ArtifactNames.CONTRACT_RESULTS_CSV, index=False)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "document_type": "expert_hysteresis_step8_publication",
            "stage": "valid",
            "purpose": "selected_trial_publication",
            "experiment_id": selected.config["experiment_id"],
            "experiment_identity": selected.config["experiment_identity"],
            "config_hash": selected.config_hash,
            "selection": selected.config["selection"],
            "configuration": selected.config["configuration"],
            "best_replay": str(selected.replay_path),
            "contracts": metrics,
            "aggregate": {
                "net_pnl": float(metrics_frame["net_pnl"].sum()),
                "initial_equity": float(metrics_frame["initial_equity"].sum()),
                "tr": float(metrics_frame["net_pnl"].sum() / metrics_frame["initial_equity"].sum()),
                "max_contract_mdd": float(metrics_frame["mdd"].max()),
                "friction": float(metrics_frame["friction"].sum()),
            },
        }
        (temporary / ArtifactNames.SUMMARY_JSON).write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        shutil.copy2(selected.frozen_config_path, temporary / "frozen_config.json")
        shutil.copy2(selected.evidence_path, temporary / SELECTION_EVIDENCE_FILENAME)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "document_type": "expert_hysteresis_step8_publication_receipt",
            "experiment_identity": selected.config["experiment_identity"],
            "config_hash": selected.config_hash,
            "stage": "valid",
            "purpose": "selected_trial_publication",
            "selected_trial": selected.config["selection"],
            "source_replay": str(selected.replay_path),
            "source_selection_evidence": str(selected.evidence_path),
        }
        (temporary / PUBLICATION_RECEIPT_FILENAME).write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-evidence", required=True)
    parser.add_argument("--frozen-config")
    parser.add_argument("--best-replay")
    parser.add_argument("--market-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = load_selected_replay(
        args.selection_evidence,
        frozen_config=args.frozen_config,
        best_replay=args.best_replay,
    )
    output = publish_selected_replay(
        selected, market_data_dir=args.market_data_dir, output_dir=args.output_dir
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
