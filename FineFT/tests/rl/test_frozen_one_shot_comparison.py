from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.audit_records import (
    ACCOUNT_RECORD_COLUMNS,
    PREDICTION_RECORD_COLUMNS,
    TRADE_RECORD_COLUMNS,
)
from RL.DiHFT.trading_utility.eligibility import (
    ChannelEligibility,
    EligibilityPolicy,
    EligibilityReport,
)
from RL.DiHFT.trading_utility.frozen_test import (
    AUDIT_SCHEMA_VERSION,
    StackTestResult,
    deterministic_frame_identity,
    freeze_production_assets,
    run_one_shot_test_comparison,
)
from RL.DiHFT.trading_utility.model_bundle import UtilityModelBundle
from RL.DiHFT.trading_utility.model_training import (
    UtilityModelTrainer,
    UtilityTrainingData,
)
from RL.DiHFT.trading_utility.prequential import PrequentialProtocol
from RL.DiHFT.trading_utility.prequential import run_strict_prequential
from RL.DiHFT.trading_utility.runtime_artifacts import publish_runtime_manifest


class _IdentityCalibrator:
    def predict_utility(self, values):
        return np.asarray(values, dtype=float)

    def predict_win_probability(self, values):
        return np.asarray(values, dtype=float)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _freeze(tmp_path: Path, *, evidence_protocol: str | None = None) -> Path:
    protocol = PrequentialProtocol()
    values = np.linspace(-2.0, 2.0, 24)
    training = UtilityTrainingData(
        features=values.reshape(-1, 1),
        feature_names=("signal",),
        net_pnl_bps=values * 2.0,
        downside_loss_bps=np.maximum(0.0, -values * 2.0),
        mae_bps=np.abs(values),
        wins=values > 0.0,
        contracts=tuple(f"fu{index % 3}" for index in range(len(values))),
        slots=np.zeros(len(values), dtype=int),
        directions=np.ones(len(values), dtype=int),
    )
    bundle = UtilityModelTrainer.fit(training, backend="sklearn", random_seed=7)
    bundle_manifest = bundle.save(tmp_path / "bundle")
    channel = ChannelEligibility(
        slot_id=0,
        direction=1,
        eligible=True,
        failure_reasons=(),
        episode_count=500,
        contract_count=3,
        trading_day_block_count=40,
        quintile_counts=(100, 100, 100, 100, 100),
        win_rates=(0.1, 0.2, 0.3, 0.4, 0.5),
        payoff_ratios=(1.0, 1.1, 1.2, 1.3, 1.4),
        win_rate_spearman=1.0,
        payoff_ratio_spearman=1.0,
        mean_net_pnl_bps=1.0,
        mean_net_pnl_lcb95_bps=0.1,
        raw_p_value=0.01,
        fdr_q_value=0.01,
        win_rate_trend_lcb95=0.01,
        payoff_ratio_trend_lcb95=0.01,
        residual_q05=-2.0,
    )
    eligibility = EligibilityReport(
        channels={"0:1": channel},
        policy=EligibilityPolicy(bootstrap_iterations=20),
        evidence_metadata={
            "outer_fold_count": 2,
            "calibration_source": "outer_unseen_predictions",
            "protocol_fingerprint": evidence_protocol or protocol.fingerprint,
        },
    ).save(tmp_path / "eligibility")
    protocol_manifest = {
        "schema_version": 1,
        "protocol_version": protocol.protocol_version,
        "protocol_fingerprint": protocol.fingerprint,
        "frozen_protocol": dict(protocol.__dict__),
        "outer_fold_count": 2,
        "final_calibration_source": "outer_unseen_predictions",
        "test_accessed": False,
        "code_version": "commit-123",
    }
    runtime = publish_runtime_manifest(
        output_dir=tmp_path / "runtime",
        bundle_manifest=bundle_manifest,
        calibrator=_IdentityCalibrator(),
        eligibility_manifest=eligibility,
        protocol_manifest=protocol_manifest,
    )
    selector = _write_json(
        tmp_path / "selector.json", {"schema_version": 1, "slots": [{"slot_id": 0}]}
    )
    cost = _write_json(
        tmp_path / "cost.json",
        {
            "cost_model_version": protocol.cost_model_version,
            "commission": "charged_once",
            "slippage": "charged_once",
            "impact": "charged_once",
        },
    )
    ood_assets = {}
    for name in (
        "vae_weights",
        "train_scaler",
        "train_id_likelihood_reference",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("utf-8"))
        ood_assets[name] = path
    return freeze_production_assets(
        output_dir=tmp_path / "frozen",
        utility_runtime_manifest=runtime,
        selector_manifest=selector,
        selector_lineage={
            "protocol_fingerprint": protocol.fingerprint,
            "valid_start": protocol.initial_start,
            "valid_end": protocol.fold_2_end,
            "build_scope": "final_full_valid",
            "code_version": "commit-123",
            "source_data_sha256": "valid-data-hash",
        },
        cost_model_manifest=cost,
        ood_assets=ood_assets,
        code_version="commit-123",
    )


def _test_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trading_day": ["2026-02-05", "2025-06-30"],
            "contract": ["fu2", "fu1"],
            "step_index": [2, 1],
            "signal": [0.5, -0.5],
        }
    )


def _prediction(stack: str, run_id: str, utility_id: str | None) -> dict:
    row = {column: None for column in PREDICTION_RECORD_COLUMNS}
    row.update(
        {
            "prediction_id": f"pred-{stack}",
            "decision_id": f"decision-{stack}",
            "run_id": run_id,
            "model_id": f"model-{stack}",
            "feature_version": "v1",
            "parameter_version": "p1",
            "contract": "fu1",
            "step_index": 1,
            "event_time": "2025-06-30T09:00:00",
            "trading_day": "2025-06-30",
            "market_index": 1,
            "routing_slot": 0,
            "routing_evidence": "{}",
            "action_value_short": 0.0,
            "action_value_flat": 0.0,
            "action_value_long": 1.0,
            "action_value_margin": 1.0,
            "available_actions": "[0,1,2]",
            "selected_action": 2,
            "gating_stack": stack,
            "final_action": 2,
        }
    )
    if stack == "tri_module":
        row.update(
            {
                "candidate_id": "candidate-1",
                "candidate_kind": "model",
                "candidate_action": 2,
                "denied_baseline_action": 1,
                "utility_raw": 3.0,
                "utility_calibrated": 2.0,
                "utility_lower": 0.5,
                "win_probability": 0.6,
                "channel_eligible": True,
                "utility_artifact_id": utility_id,
            }
        )
    return row


def _accounts(stack: str) -> list[dict]:
    rows = []
    for kind in ("pre", "post"):
        row = {column: 0.0 for column in ACCOUNT_RECORD_COLUMNS}
        row.update(
            {
                "snapshot_id": f"{stack}-{kind}",
                "run_id": f"run-{stack}",
                "contract": "fu1",
                "account_id": "account",
                "account_scope": "contract",
                "step_index": 1,
                "snapshot_type": kind,
                "event_time": "2025-06-30T09:00:00",
                "trading_day": "2025-06-30",
            }
        )
        rows.append(row)
    return rows


def _trade(stack: str) -> dict:
    row = {column: 0.0 for column in TRADE_RECORD_COLUMNS}
    row.update(
        {
            "trade_id": f"trade-{stack}",
            "decision_id": f"decision-{stack}",
            "prediction_id": f"pred-{stack}",
            "run_id": f"run-{stack}",
            "contract": "fu1",
            "step_index": 1,
            "event_time": "2025-06-30T09:00:00",
            "trading_day": "2025-06-30",
            "leg_index": 0,
            "event_type": "fill",
            "execution_status": "filled",
            "decision_reason": "POLICY_INFERENCE",
            "pre_snapshot_id": f"{stack}-pre",
            "post_snapshot_id": f"{stack}-post",
        }
    )
    return row


def _result(stack: str, frame: pd.DataFrame, freeze: dict | None = None) -> StackTestResult:
    utility_id = freeze["utility_runtime_artifact_id"] if freeze else None
    metrics = {
        name: {"observations": len(frame)}
        for name in (
            "performance",
            "risk",
            "coverage",
            "ood",
            "route",
            "utility_rejection",
            "channel_support",
            "audit_integrity",
        )
    }
    return StackTestResult(
        stack_type=stack,
        test_input_identity=deterministic_frame_identity(frame),
        cost_model_version="after_cost_bps_v1",
        audit_schema_version=AUDIT_SCHEMA_VERSION,
        metrics=metrics,
        prediction_records=[_prediction(stack, f"run-{stack}", utility_id)],
        trade_records=[_trade(stack)],
        account_records=_accounts(stack),
        scenario_coverage=(
            (
                "open",
                "same_direction_add",
                "reversal",
                "null_candidate",
                "ineligible_channel",
                "utility_lower_non_positive",
                "hard_stop",
                "circuit_breaker",
                "ood_missing_data",
            )
            if stack == "tri_module"
            else ("legacy_without_utility_assets",)
        ),
    )


def test_freeze_and_formal_test_are_read_only_comparable_and_one_shot(tmp_path) -> None:
    manifest = _freeze(tmp_path)
    freeze = json.loads(manifest.read_text(encoding="utf-8"))
    asset_hashes = {
        name: hashlib.sha256(Path(metadata["path"]).read_bytes()).hexdigest()
        for name, metadata in freeze["assets"].items()
    }

    report_path = run_one_shot_test_comparison(
        frozen_manifest=manifest,
        test_frame=_test_frame(),
        legacy_evaluator=lambda frame: _result("legacy", frame),
        tri_module_evaluator=lambda frame, frozen: _result(
            "tri_module", frame, dict(frozen)
        ),
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed_once"
    assert report["test_range"] == ["2025-06-30", "2026-02-05"]
    assert report["declarations"]["test_used_for_tuning"] is False
    assert report["declarations"]["automatic_default_switch"] is False
    assert report["declarations"]["automatic_legacy_deletion"] is False
    assert report["legacy"]["prediction_records"][0]["utility_raw"] is None
    assert report["tri_module"]["prediction_records"][0]["utility_raw"] == 3.0
    assert asset_hashes == {
        name: hashlib.sha256(Path(metadata["path"]).read_bytes()).hexdigest()
        for name, metadata in freeze["assets"].items()
    }
    with pytest.raises(FileExistsError, match="already been published"):
        run_one_shot_test_comparison(
            frozen_manifest=manifest,
            test_frame=_test_frame(),
            legacy_evaluator=lambda frame: _result("legacy", frame),
            tri_module_evaluator=lambda frame, frozen: _result("tri_module", frame, dict(frozen)),
        )


def test_failed_formal_test_leaves_no_partial_official_report(tmp_path) -> None:
    manifest = _freeze(tmp_path)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_one_shot_test_comparison(
            frozen_manifest=manifest,
            test_frame=_test_frame(),
            legacy_evaluator=lambda frame: _result("legacy", frame),
            tri_module_evaluator=lambda frame, frozen: (_ for _ in ()).throw(
                RuntimeError("synthetic failure")
            ),
        )

    assert not (manifest.parent / "official_test").exists()
    assert not list(manifest.parent.glob(".official_test.*"))


@pytest.mark.parametrize(
    ("invalid_kind", "message"),
    [
        ("shadow", "shadow/counterfactual fills"),
        ("exchange_rejection", "utility rejection cannot be classified"),
    ],
)
def test_formal_test_rejects_false_ledger_semantics_atomically(
    tmp_path, invalid_kind, message
) -> None:
    manifest = _freeze(tmp_path)

    def invalid_tri(frame: pd.DataFrame, frozen: dict) -> StackTestResult:
        result = _result("tri_module", frame, dict(frozen))
        trade = dict(result.trade_records[0])
        if invalid_kind == "shadow":
            trade["event_type"] = "shadow_fill"
        else:
            trade["decision_reason"] = "UTILITY_REJECTION"
            trade["execution_status"] = "exchange_rejected"
        return replace(result, trade_records=[trade])

    with pytest.raises(ValueError, match=message):
        run_one_shot_test_comparison(
            frozen_manifest=manifest,
            test_frame=_test_frame(),
            legacy_evaluator=lambda frame: _result("legacy", frame),
            tri_module_evaluator=invalid_tri,
        )

    assert not (manifest.parent / "official_test").exists()


def test_repeated_non_test_freeze_has_stable_content_identity(tmp_path) -> None:
    manifest = _freeze(tmp_path)
    first = json.loads(manifest.read_text(encoding="utf-8"))

    second_manifest = _refreeze_from_manifest(first, tmp_path / "frozen-copy")
    second = json.loads(second_manifest.read_text(encoding="utf-8"))

    assert second["freeze_id"] == first["freeze_id"]


def test_freeze_fails_closed_for_stale_eligibility_or_protocol(tmp_path) -> None:
    with pytest.raises(ValueError, match="eligibility protocol fingerprint"):
        _freeze(tmp_path, evidence_protocol="stale")
    assert not (tmp_path / "frozen").exists()


def _refreeze_from_manifest(frozen: dict, target: Path) -> Path:
    return freeze_production_assets(
        output_dir=target,
        utility_runtime_manifest=frozen["assets"]["utility_runtime_manifest"]["path"],
        selector_manifest=frozen["assets"]["selector_manifest"]["path"],
        selector_lineage=frozen["selector"]["lineage"],
        cost_model_manifest=frozen["assets"]["cost_model_manifest"]["path"],
        ood_assets={
            name.split(":", 1)[1]: metadata["path"]
            for name, metadata in frozen["assets"].items()
            if name.startswith("ood:")
        },
        code_version="commit-123",
    )


@pytest.mark.parametrize("failure", ["missing_head", "damaged_hash", "feature_schema"])
def test_freeze_rejects_incompatible_model_assets_atomically(tmp_path, failure) -> None:
    manifest = _freeze(tmp_path)
    frozen = json.loads(manifest.read_text(encoding="utf-8"))
    head = Path(frozen["assets"]["utility_head:mean_net_pnl"]["path"])
    if failure == "missing_head":
        head.unlink()
        message = "utility head is missing"
    elif failure == "damaged_hash":
        head.write_bytes(head.read_bytes() + b"damage")
        message = "utility head hash mismatch"
    else:
        runtime_path = Path(frozen["assets"]["utility_runtime_manifest"]["path"])
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime["feature_schema"] = ["wrong_feature"]
        core = {key: value for key, value in runtime.items() if key != "artifact_id"}
        runtime["artifact_id"] = hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
        message = "feature schema mismatch"

    target = tmp_path / "must-not-publish"
    with pytest.raises(ValueError, match=message):
        _refreeze_from_manifest(frozen, target)
    assert not target.exists()


class _PrefixModel:
    def __init__(self, train_mean: float) -> None:
        self.train_mean = train_mean

    def predict_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        signal = frame["signal"].to_numpy(dtype=float)
        return pd.DataFrame(
            {
                "mean_net_pnl_bps": self.train_mean + signal,
                "cvar_loss_bps": np.full(len(frame), 2.0),
                "mae_loss_bps": np.full(len(frame), 1.0),
                "win_probability_raw": 1.0 / (1.0 + np.exp(-signal)),
            }
        )


def test_two_fold_tracer_bullet_reaches_one_frozen_test_and_joined_ledgers(tmp_path) -> None:
    dates = pd.date_range("2024-07-23", "2025-06-27", freq="D")
    indexes = np.arange(len(dates))
    valid = pd.DataFrame(
        {
            "trading_day": dates,
            "contract": [f"fu{index % 4}" for index in indexes],
            "slot_id": indexes % 2,
            "direction": np.where(indexes % 3 == 0, -1, 1),
            "signal": np.where(indexes % 2 == 0, 1.0, -1.0),
            "net_pnl_bps": np.where(indexes % 2 == 0, 8.0, -6.0),
            "win": indexes % 2 == 0,
        }
    )
    prefix_ends: list[pd.Timestamp] = []

    result = run_strict_prequential(
        valid,
        feature_names=("signal",),
        fit_model=lambda prefix, lambda_cvar, lambda_mae: _PrefixModel(
            float(prefix["net_pnl_bps"].mean())
        ),
        build_selector=lambda prefix, name: prefix_ends.append(
            prefix["trading_day"].max()
        ) or name,
        protocol=PrequentialProtocol(inner_splits=2),
    )
    assert result.outer_predictions.groupby("outer_fold").size().index.tolist() == [1, 2]
    assert result.outer_predictions.index.is_unique
    assert prefix_ends[0] < pd.Timestamp("2024-11-01")
    assert prefix_ends[1] < pd.Timestamp("2025-03-01")
    assert prefix_ends[-1] == pd.Timestamp("2025-06-27")
    assert result.final_calibration_source == "outer_unseen_predictions"

    manifest = _freeze(tmp_path / "assets")
    report_path = run_one_shot_test_comparison(
        frozen_manifest=manifest,
        test_frame=_test_frame(),
        legacy_evaluator=lambda frame: _result("legacy", frame),
        tri_module_evaluator=lambda frame, frozen: _result("tri_module", frame, dict(frozen)),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tri = report["tri_module"]
    assert tri["prediction_records"][0]["candidate_id"] == "candidate-1"
    assert tri["trade_records"][0]["prediction_id"] == tri["prediction_records"][0]["prediction_id"]
    assert tri["trade_records"][0]["post_snapshot_id"] == tri["account_records"][1]["snapshot_id"]
    assert set(tri["scenario_coverage"]) >= {
        "open",
        "same_direction_add",
        "reversal",
        "null_candidate",
        "ineligible_channel",
        "utility_lower_non_positive",
        "hard_stop",
        "circuit_breaker",
    }
