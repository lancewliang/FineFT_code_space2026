from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from RL.DiHFT.high_level.trading_utility.model_bundle import (
    UtilityModelBundle,
    UtilityTrainingData,
)
from RL.DiHFT.high_level.trading_utility.eligibility import (
    EligibilityPolicy,
    evaluate_channel_eligibility,
)
from RL.DiHFT.high_level.trading_utility.prequential import (
    PrequentialProtocol,
    build_utility_model_fitter,
    run_strict_prequential,
)
from RL.DiHFT.high_level.trading_utility.runtime import publish_runtime_manifest


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_from_candidate_csv(
    *,
    episodes_csv: str | Path,
    output_dir: str | Path,
    feature_names: Sequence[str],
    lambda_cvar: float = 0.5,
    lambda_mae: float = 0.5,
    random_seed: int = 42,
    backend: str = "auto",
) -> Path:
    source = Path(episodes_csv)
    if not source.is_file():
        raise ValueError(f"candidate episode CSV does not exist: {source}")
    frame = pd.read_csv(source)
    required = {
        "feature_values",
        "net_pnl_bps",
        "downside_loss_bps",
        "mae_bps",
        "win",
        "contract",
        "slot_id",
        "direction",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate episode CSV is missing columns: {missing}")
    names = tuple(str(name).strip() for name in feature_names if str(name).strip())
    if not names:
        raise ValueError("at least one feature name is required")
    rows = [json.loads(str(value)) for value in frame["feature_values"]]
    features = np.asarray(rows, dtype=float)
    if features.ndim != 2 or features.shape[1] != len(names):
        raise ValueError("feature_values do not match the declared feature names")
    data = UtilityTrainingData(
        features=features,
        feature_names=names,
        net_pnl_bps=frame["net_pnl_bps"].to_numpy(dtype=float),
        downside_loss_bps=frame["downside_loss_bps"].to_numpy(dtype=float),
        mae_bps=frame["mae_bps"].to_numpy(dtype=float),
        wins=frame["win"].astype(bool).to_numpy(),
        contracts=tuple(frame["contract"].astype(str)),
        slots=frame["slot_id"].to_numpy(dtype=int),
        directions=frame["direction"].to_numpy(dtype=int),
    )
    bundle = UtilityModelBundle.fit(
        data,
        lambda_cvar=lambda_cvar,
        lambda_mae=lambda_mae,
        random_seed=random_seed,
        backend=backend,
    )
    target = Path(output_dir)
    manifest_path = bundle.save(target)
    run_manifest = {
        "schema_version": 1,
        "candidate_episode_source": str(source),
        "candidate_episode_sha256": _source_hash(source),
        "feature_schema": list(names),
        "candidate_label_schema": [
            "net_pnl_bps",
            "downside_loss_bps",
            "mae_bps",
            "win",
        ],
        "lambda_grid": [0.25, 0.5, 1.0, 2.0],
        "training_row_count": int(len(frame)),
        "contracts": sorted(frame["contract"].astype(str).unique()),
        "utility_bundle_manifest": manifest_path.name,
        "utility_artifact_id": bundle.artifact_id,
    }
    (target / "utility_training_run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_path


def train_prequential_from_candidate_csv(
    *,
    episodes_csv: str | Path,
    output_dir: str | Path,
    feature_names: Sequence[str],
    random_seed: int = 42,
    backend: str = "auto",
    code_version: str = "unspecified",
) -> Path:
    """Run the frozen two-Fold protocol and publish loadable production assets."""
    source = Path(episodes_csv)
    if not source.is_file():
        raise ValueError(f"candidate episode CSV does not exist: {source}")
    frame = pd.read_csv(source)
    names = tuple(str(name).strip() for name in feature_names if str(name).strip())
    if not names:
        raise ValueError("at least one feature name is required")
    if "feature_values" not in frame:
        raise ValueError("candidate episode CSV is missing feature_values")
    values = np.asarray(
        [json.loads(str(value)) for value in frame["feature_values"]], dtype=float
    )
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("feature_values do not match the declared feature names")
    for index, name in enumerate(names):
        frame[name] = values[:, index]

    protocol = PrequentialProtocol()
    result = run_strict_prequential(
        frame,
        feature_names=names,
        fit_model=build_utility_model_fitter(
            names, backend=backend, random_seed=random_seed
        ),
        protocol=protocol,
    )
    if not isinstance(result.final_model, UtilityModelBundle):
        raise ValueError("final prequential model does not implement bundle publication")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    bundle_manifest = result.final_model.save(target / "bundle")
    outer_path = target / "outer_unseen_predictions.csv"
    result.outer_predictions.to_csv(outer_path, index=False)
    eligibility = evaluate_channel_eligibility(
        result.outer_predictions,
        policy=EligibilityPolicy(
            minimum_episodes=protocol.minimum_channel_episodes,
            minimum_per_quintile=protocol.minimum_quintile_episodes,
            minimum_contracts=protocol.minimum_contracts,
            minimum_trading_day_blocks=protocol.minimum_trading_day_blocks,
            fdr_alpha=protocol.fdr_alpha,
            minimum_spearman=protocol.minimum_spearman,
        ),
    )
    eligibility = replace(
        eligibility,
        evidence_metadata={
            "outer_fold_count": 2,
            "calibration_source": result.final_calibration_source,
            "calibrator": "isotonic_utility_plus_platt_win_probability",
            "lambda_cvar": result.final_lambda_cvar,
            "lambda_mae": result.final_lambda_mae,
            "protocol_fingerprint": result.protocol_manifest[
                "protocol_fingerprint"
            ],
        },
    )
    eligibility_manifest = eligibility.save(target / "eligibility")
    protocol_manifest = {
        **result.protocol_manifest,
        "candidate_episode_source": str(source.resolve()),
        "candidate_episode_sha256": _source_hash(source),
        "outer_predictions_sha256": _source_hash(outer_path),
        "high_level_gating_stack": "tri_module",
        "code_version": str(code_version),
    }
    runtime_manifest = publish_runtime_manifest(
        output_dir=target / "runtime",
        bundle_manifest=bundle_manifest,
        calibrator=result.final_calibrator,
        eligibility_manifest=eligibility_manifest,
        protocol_manifest=protocol_manifest,
    )
    report = {
        "schema_version": 1,
        "status": "frozen",
        "outer_fold_count": 2,
        "final_calibration_source": result.final_calibration_source,
        "lambda_cvar": result.final_lambda_cvar,
        "lambda_mae": result.final_lambda_mae,
        "protocol": protocol_manifest,
        "bundle_manifest": str(bundle_manifest.resolve()),
        "eligibility_manifest": str(eligibility_manifest.resolve()),
        "runtime_manifest": str(runtime_manifest.resolve()),
        "test_accessed": False,
    }
    (target / "prequential_training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return runtime_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a calibrated-trading-utility model bundle"
    )
    parser.add_argument("--episodes-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--feature-names",
        required=True,
        help="Comma-separated names matching feature_values order",
    )
    parser.add_argument("--lambda-cvar", type=float, default=0.5)
    parser.add_argument("--lambda-mae", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        choices=["auto", "catboost", "sklearn"],
        default="auto",
    )
    parser.add_argument(
        "--prequential",
        action="store_true",
        help="run the frozen two-Fold workflow and publish the runtime manifest",
    )
    parser.add_argument(
        "--code-version",
        default="unspecified",
        help="immutable source revision recorded for production freeze compatibility",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.prequential:
        train_prequential_from_candidate_csv(
            episodes_csv=args.episodes_csv,
            output_dir=args.output_dir,
            feature_names=tuple(args.feature_names.split(",")),
            random_seed=args.random_seed,
            backend=args.backend,
            code_version=args.code_version,
        )
    else:
        train_from_candidate_csv(
            episodes_csv=args.episodes_csv,
            output_dir=args.output_dir,
            feature_names=tuple(args.feature_names.split(",")),
            lambda_cvar=args.lambda_cvar,
            lambda_mae=args.lambda_mae,
            random_seed=args.random_seed,
            backend=args.backend,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
