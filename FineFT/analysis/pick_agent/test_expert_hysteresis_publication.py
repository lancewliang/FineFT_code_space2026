"""Controlled public-entry tests for the fail-closed step-8 publisher."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from analysis.pick_agent.expert_hysteresis_publication import _identity, main
from common import AuditArtifactNames, HistoryArtifactNames
from common.routing_audit import ROUTING_AUDIT_SCHEMA_VERSION
from RL.DiHFT.high_level.frozen_experiment import (
    POLICY_FIELDS,
    RUNTIME_FIELDS,
    SCHEMA_VERSION,
    SEARCH_FIELDS,
    SELECTION_FIELDS,
    STRATEGY_VERSION,
)


def _write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    params = {
        "slope_window_length": 3,
        "volatility_window_length": 4,
        "slope_gamma": 0.8,
        "volatility_gamma": 0.7,
        "slope_rule_base_threshold": 0.8,
        "volatility_rule_base_threshold": 0.75,
        "hysteresis_exit_ratio": 0.5,
        "expert_switch_lead": 0.1,
        "expert_confirm_steps": 2,
    }
    storage = tmp_path / "selected.sqlite3"
    study = optuna.create_study(
        study_name="selected-study", direction="maximize", storage=f"sqlite:///{storage}"
    )
    trial = study.ask()
    trial.suggest_int("slope_window_length", 3, 3)
    trial.suggest_int("volatility_window_length", 4, 4)
    for field in ("slope_gamma", "volatility_gamma", "slope_rule_base_threshold", "volatility_rule_base_threshold", "hysteresis_exit_ratio", "expert_switch_lead"):
        trial.suggest_float(field, params[field], params[field])
    trial.suggest_int("expert_confirm_steps", 2, 2)
    study.tell(trial, 1.23)
    selection = {
        "study_name": "selected-study",
        "study_storage_identity": _identity(
            {"experiment_identity": "experiment-hash", "study_name": "selected-study", "storage_path": str(storage.resolve())}
        ),
        "trial_number": 0,
        "trial_state": "COMPLETE",
        "objective_name": "expert_hysteresis_calibration_score",
        "objective_version": "v1",
        "objective_direction": "maximize",
        "objective_value": 1.23,
    }
    assert set(selection) == set(SELECTION_FIELDS)
    configuration = {field: 1 for field in POLICY_FIELDS + RUNTIME_FIELDS}
    configuration.update(params)
    configuration.update(
        {
            "gating_strategy": "expert_hysteresis",
            "strategy_version": STRATEGY_VERSION,
            "slope_hold_threshold": 0.4,
            "volatility_hold_threshold": 0.375,
            "leverage_choices": [1],
            "candidate_slot_order": ["slot-0"],
        }
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "document_type": "expert_hysteresis_frozen_config",
        "experiment_id": "fixture",
        "experiment_manifest": str(tmp_path / "manifest.json"),
        "experiment_identity": "experiment-hash",
        "selected_on": "calibration",
        "selection": selection,
        "search_space": {},
        "configuration": configuration,
    }
    config["config_hash"] = _identity(config)
    config_path = tmp_path / "selected.json"
    _write_json(config_path, config)

    replay = tmp_path / "best_replay" / "contracts" / "CU"
    replay.mkdir(parents=True)
    _write_json(
        replay / AuditArtifactNames.AUDIT_METADATA_JSON,
        {
            "configuration_identity": config["config_hash"],
            "experiment_identity": "experiment-hash",
            "gating_strategy": "expert_hysteresis",
        },
    )
    np.save(replay / HistoryArtifactNames.REWARD_HISTORY_NPY, np.array([1.0, -0.5]))
    np.save(replay / HistoryArtifactNames.TOTAL_ASSET_HISTORY_NPY, np.array([100.0, 101.0, 100.5]))
    np.save(replay / HistoryArtifactNames.MARKET_INDEX_HISTORY_NPY, np.array([0, 1, 2]))
    np.save(replay / HistoryArtifactNames.MICRO_ACTION_HISTORY_NPY, np.array([1, 2]))
    np.save(replay / HistoryArtifactNames.COMMISSION_FEE_HISTORY_NPY, np.array([0.1, 0.2]))
    records = []
    for step, timestamp in enumerate(("2025-01-01 00:00:00+00:00", "2025-01-01 00:10:00+00:00")):
        records.append(
            {
                "schema_version": ROUTING_AUDIT_SCHEMA_VERSION,
                "decision_id": f"dec_run_CU_{step:06d}",
                "run_id": "run",
                "contract": "CU",
                "step_index": step,
                "event_time": timestamp,
                "trading_day": "20250101",
                "strategy_version": STRATEGY_VERSION,
                "dominant_slot": 0,
                "active_slot_before": 0,
                "active_slot_after": 0,
                "candidate_slot_before": None,
                "candidate_slot_after": None,
                "candidate_count_before": 0,
                "candidate_count_after": 0,
                "completed_candidate_slot": None,
                "completed_candidate_count": 0,
                "incumbent_invalidated": False,
                "routing_reason": "expert_retained",
                "transition_reason": "expert_retained",
                "admission_kind": "",
                "action_reason": "POLICY_INFERENCE",
                "queried_slot": None,
                "proposed_action": None,
                "final_action": 1,
                "prediction_id": "",
                "volatility_scores": [0.9],
                "slope_scores": [0.9],
                "volatility_entry_threshold": 0.75,
                "slope_entry_threshold": 0.8,
                "volatility_holding_threshold": 0.375,
                "slope_holding_threshold": 0.4,
                "hysteresis_exit_ratio": 0.5,
                "expert_switch_lead": 0.1,
                "expert_confirm_steps": 2,
            }
        )
    _write_json(replay / AuditArtifactNames.AUDIT_ROUTING_JSON, records)
    market = tmp_path / "market"
    market.mkdir()
    pd.DataFrame(
        {
            "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T00:10:00Z", "2025-01-01T00:20:00Z"],
            "mark_price": [10.0, 11.0, 10.5],
        }
    ).to_csv(market / "CU.csv", index=False)
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "document_type": "expert_hysteresis_calibration_selection",
        "strategy_version": STRATEGY_VERSION,
        "experiment_id": "fixture",
        "experiment_identity": "experiment-hash",
        "stage": "valid",
        "purpose": "calibration_selection",
        "study": {"name": "selected-study", "storage_identity": selection["study_storage_identity"], "storage_path": str(storage.resolve())},
        "objective": {"direction": "maximize"},
        "selected_trial": {"number": 0, "state": "COMPLETE", "objective_value": 1.23, "parameters": params, "effective_configuration": {**params, "slope_hold_threshold": 0.4, "volatility_hold_threshold": 0.375}, "raw_metrics": {"tr": 0.9}},
        "frozen_config": {"path": str(config_path.resolve()), "config_hash": config["config_hash"]},
        "best_replay": {"artifact_path": str((tmp_path / "best_replay").resolve()), "objective_value": 1.23, "raw_metrics": {"tr": 0.9}, "matches_lean_trial": True},
        "raw_score_source_identity": "score-source",
    }
    evidence["evidence_hash"] = _identity(evidence)
    evidence_path = tmp_path / "selection_evidence.json"
    _write_json(evidence_path, evidence)
    return evidence_path, config_path, market


def test_public_entry_publishes_selected_complete_trial_without_reselection(tmp_path: Path) -> None:
    evidence, config, market = _fixture(tmp_path)
    output = tmp_path / "publication"
    assert main(["--selection-evidence", str(evidence), "--frozen-config", str(config), "--market-data-dir", str(market), "--output-dir", str(output)]) == 0
    receipt = json.loads((output / "step8_publication_receipt.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert receipt["selected_trial"]["trial_number"] == 0
    assert summary["contracts"][0]["tr"] == pytest.approx(0.005)
    assert summary["selection"]["objective_value"] == 1.23
    assert (output / "CU_routing_and_returns.png").is_file()


@pytest.mark.parametrize("mutation", ["missing_replay", "config_hash", "missing_confirm_steps", "other_study"])
def test_public_entry_rejects_invalid_selected_evidence(tmp_path: Path, mutation: str) -> None:
    evidence, config, market = _fixture(tmp_path)
    if mutation == "missing_replay":
        shutil_target = Path(json.loads(evidence.read_text())["best_replay"]["artifact_path"])
        shutil_target.rename(shutil_target.with_name("gone"))
    elif mutation == "config_hash":
        payload = json.loads(evidence.read_text())
        payload["frozen_config"]["config_hash"] = "wrong"
        payload["evidence_hash"] = _identity({key: value for key, value in payload.items() if key != "evidence_hash"})
        _write_json(evidence, payload)
    elif mutation == "missing_confirm_steps":
        payload = json.loads(config.read_text())
        del payload["configuration"]["expert_confirm_steps"]
        payload["config_hash"] = _identity({key: value for key, value in payload.items() if key != "config_hash"})
        _write_json(config, payload)
        evidence_payload = json.loads(evidence.read_text())
        evidence_payload["frozen_config"]["config_hash"] = payload["config_hash"]
        evidence_payload["evidence_hash"] = _identity({key: value for key, value in evidence_payload.items() if key != "evidence_hash"})
        _write_json(evidence, evidence_payload)
    else:
        payload = json.loads(evidence.read_text())
        other = tmp_path / "other.sqlite3"
        optuna.create_study(study_name="other", storage=f"sqlite:///{other}")
        payload["study"]["storage_path"] = str(other.resolve())
        payload["evidence_hash"] = _identity({key: value for key, value in payload.items() if key != "evidence_hash"})
        _write_json(evidence, payload)
    with pytest.raises((FileNotFoundError, ValueError)):
        main(["--selection-evidence", str(evidence), "--market-data-dir", str(market), "--output-dir", str(tmp_path / "out")])
