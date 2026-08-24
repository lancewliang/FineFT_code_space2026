"""Atomic, directory-level calibration of Valid Dynamic Labels."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from . import label_util as util
    from .manifests import (
        SliceContractManifest,
        SliceFileManifest,
        SliceLabelManifest,
        SliceManifest,
        SkippedContractManifest,
    )
except ImportError:
    import label_util as util
    from manifests import (
        SliceContractManifest,
        SliceFileManifest,
        SliceLabelManifest,
        SliceManifest,
        SkippedContractManifest,
    )


@dataclass
class _ContractFit:
    contract: str
    source_path: Path
    processed_path: Path
    prepared: pd.DataFrame
    groups: list[tuple[Any, list[int], list[float], np.ndarray]]
    slopes: list[float]


def _contract_name(path: Path) -> str:
    stem = path.stem
    return stem[3:] if stem.startswith("df_") else stem


def _validate_input(
    raw_data: pd.DataFrame, path: Path, timestamp: str, tic: str
) -> None:
    required = ["bid1_price", tic]
    if timestamp != "index":
        required.append(timestamp)
    missing = [column for column in required if column not in raw_data.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    prices = pd.to_numeric(raw_data["bid1_price"], errors="coerce").to_numpy()
    if not np.isfinite(prices).all():
        raise ValueError(f"{path} contains non-finite bid1_price values")
    if (prices == 0).any():
        raise ValueError(f"{path} contains zero bid1_price values")
    if raw_data[tic].isna().any():
        raise ValueError(f"{path} contains null {tic} values")
    timestamp_values = raw_data[timestamp] if timestamp != "index" else raw_data.index
    if pd.isna(timestamp_values).any():
        raise ValueError(f"{path} contains null or non-finite {timestamp} values")
    if pd.api.types.is_numeric_dtype(timestamp_values):
        if not np.isfinite(np.asarray(timestamp_values, dtype=float)).all():
            raise ValueError(f"{path} contains null or non-finite {timestamp} values")
    elif pd.api.types.is_object_dtype(timestamp_values):
        if (timestamp_values.astype(str).str.strip() == "").any():
            raise ValueError(f"{path} contains empty {timestamp} values")


def _prepare_raw_data(
    raw_data: pd.DataFrame,
    path: Path,
    *,
    key_indicator: str,
    timestamp: str,
    tic: str,
) -> pd.DataFrame:
    _validate_input(raw_data, path, timestamp, tic)
    prepared = raw_data.copy()
    if timestamp == "index":
        prepared[timestamp] = prepared.index
    prepared[key_indicator] = prepared["bid1_price"]
    return prepared.reset_index(drop=True)


def _point(value: Any) -> int:
    if isinstance(value, (list, tuple, np.ndarray)):
        return int(value[0])
    return int(value)


def _finite_slopes(values: Any, path: Path) -> list[float]:
    slopes = [float(np.asarray(value).reshape(-1)[0]) for value in values]
    if not slopes or not np.isfinite(slopes).all():
        raise ValueError(f"{path} produced no finite final segment slopes")
    return slopes


def _fit_contract(
    source_path: Path,
    prepared: pd.DataFrame,
    processed_path: Path,
    stage_root: Path,
    *,
    key_indicator: str,
    timestamp: str,
    tic: str,
    filter_strength: int,
    min_length_limit: int,
    merging_threshold: float,
    merging_metric: str,
    merging_dynamic_constraint: int,
    max_length_expectation: int,
    dynamic_number: int,
) -> _ContractFit:
    contract = _contract_name(source_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_feather(processed_path)
    worker_data = prepared.copy()
    for _, positions in worker_data.groupby(tic, sort=False).groups.items():
        first_price = worker_data.loc[positions[0], key_indicator]
        worker_data.loc[positions, key_indicator] = (
            worker_data.loc[positions, key_indicator] / first_price
        )
    worker_path = stage_root / "worker" / processed_path.name
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    worker_data.to_feather(worker_path)
    worker = util.Worker(
        str(worker_path),
        "slice_and_merge",
        filter_strength=filter_strength,
        key_indicator=key_indicator,
        timestamp=timestamp,
        tic=tic,
        labeling_method="slope",
        min_length_limit=min_length_limit,
        merging_threshold=merging_threshold,
        merging_metric=merging_metric,
        merging_dynamic_constraint=merging_dynamic_constraint,
    )
    worker.fit(dynamic_number, max_length_expectation, min_length_limit)

    groups: list[tuple[Any, list[int], list[float], np.ndarray]] = []
    slopes: list[float] = []
    for worker_tic in worker.tics:
        points = [_point(value) for value in worker.turning_points_dict[worker_tic]]
        group_slopes = _finite_slopes(
            worker.norm_coef_list_dict[worker_tic], source_path
        )
        if len(points) != len(group_slopes) + 1:
            raise ValueError(f"{source_path} has inconsistent segment boundaries")
        row_positions = np.flatnonzero(prepared[tic].to_numpy() == worker_tic)
        if points[0] != 0 or points[-1] != len(row_positions):
            raise ValueError(f"{source_path} has invalid segment boundaries")
        groups.append((worker_tic, points, group_slopes, row_positions))
        slopes.extend(group_slopes)
    return _ContractFit(
        contract=contract,
        source_path=source_path,
        processed_path=processed_path,
        prepared=prepared,
        groups=groups,
        slopes=slopes,
    )


def _label_for_slope(slope: float, thresholds: list[float]) -> int:
    for index, threshold in enumerate(thresholds):
        if slope <= threshold:
            return index
    return len(thresholds)


def _slope_statistics(slopes: list[float]) -> dict[str, float | int]:
    values = np.asarray(slopes, dtype=float)
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
    }


def _build_contract_outputs(
    fit: _ContractFit,
    stage_root: Path,
    valid_root: Path,
    *,
    dynamic_number: int,
    thresholds: list[float],
) -> SliceContractManifest:
    labels = {
        f"label_{label}": SliceLabelManifest(label=f"label_{label}")
        for label in range(dynamic_number)
    }
    row_labels = np.full(len(fit.prepared), -1, dtype=int)
    for _, points, slopes, row_positions in fit.groups:
        for start, end, slope in zip(points[:-1], points[1:], slopes):
            label = _label_for_slope(slope, thresholds)
            row_labels[row_positions[start:end]] = label
    if (row_labels < 0).any():
        raise ValueError(f"{fit.source_path} has unaccounted input rows")

    contract_root = stage_root / fit.contract
    counters = [0] * dynamic_number
    previous_start = 0
    previous_label = int(row_labels[0])

    def write_segment(start: int, end: int, label: int) -> None:
        if end <= start:
            return
        label_name = f"label_{label}"
        stage_path = contract_root / label_name / f"df_{counters[label]}.feather"
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        fit.prepared.iloc[start:end].reset_index(drop=True).to_feather(stage_path)
        final_path = valid_root / fit.contract / label_name / stage_path.name
        output_rows = int(end - start)
        labels[label_name].file_count += 1
        labels[label_name].total_row_count += output_rows
        labels[label_name].files.append(
            SliceFileManifest(path=str(final_path), output_row_count=output_rows)
        )
        counters[label] += 1

    for index in range(1, len(row_labels)):
        current_label = int(row_labels[index])
        if current_label != previous_label:
            write_segment(previous_start, index, previous_label)
            previous_start = index
            previous_label = current_label
    write_segment(previous_start, len(row_labels), previous_label)
    return SliceContractManifest(
        contract=fit.contract,
        processed_path=str(valid_root / "processed" / fit.processed_path.name),
        input_row_count=int(len(fit.prepared)),
        labels=labels,
    )


def _backup_published_outputs(
    valid_root: Path, staging_path: Path, backup_root: Path
) -> None:
    backup_root.mkdir()
    for child in valid_root.iterdir():
        if child == staging_path or child == backup_root:
            continue
        if child.name.startswith(".valid-cross-contract-staging-"):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            continue
        if child.name == "slice_manifest.json" or child.is_dir():
            shutil.move(str(child), str(backup_root / child.name))


def build_valid_dataset(
    valid_dir: str | Path,
    *,
    dynamic_number: int = 5,
    labeling_method: str = "slope",
    key_indicator: str = "mark_price",
    timestamp: str = "timestamp",
    tic: str = "symbol",
    filter_strength: int = 1,
    min_length_limit: int = 288,
    merging_threshold: float = 0.0003,
    merging_metric: str = "DTW_distance",
    merging_dynamic_constraint: int = 1,
    max_length_expectation: int = 864,
    filter_padlen: int = 15,
) -> SliceManifest:
    """Fit and atomically publish one final slope calibration for all valid files."""
    if labeling_method != "slope":
        raise ValueError(
            "production valid cross-contract calibration supports final slope "
            "labeling only"
        )
    if dynamic_number < 1:
        raise ValueError("dynamic_number must be positive")
    valid_root = Path(valid_dir).resolve()
    if not valid_root.is_dir():
        raise FileNotFoundError(f"missing valid directory: {valid_root}")
    source_paths = sorted(valid_root.glob("*.feather"), key=lambda path: path.name)
    if not source_paths:
        raise ValueError(f"valid directory has no contract files: {valid_root}")
    contract_names = [_contract_name(path) for path in source_paths]
    if len(set(contract_names)) != len(contract_names):
        raise ValueError("valid directory contains duplicate contract identities")

    stage_root = valid_root / f".valid-cross-contract-staging-{uuid.uuid4().hex}"
    stage_root.mkdir()
    manifest_path = valid_root / "slice_manifest.json"
    fits: list[_ContractFit] = []
    skipped: dict[str, SkippedContractManifest] = {}
    try:
        for source_path in source_paths:
            contract = _contract_name(source_path)
            raw_data = pd.read_feather(source_path)
            prepared = _prepare_raw_data(
                raw_data,
                source_path,
                key_indicator=key_indicator,
                timestamp=timestamp,
                tic=tic,
            )
            processed_path = (
                stage_root / "processed" / f"valid_processed_{contract}.feather"
            )
            if len(prepared) <= filter_padlen:
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                prepared.to_feather(processed_path)
                skipped[contract] = SkippedContractManifest(
                    contract=contract,
                    processed_path=str(valid_root / "processed" / processed_path.name),
                    reason=(
                        "insufficient rows for dynamic slicing: "
                        f"{len(prepared)} <= filter padlen {filter_padlen}"
                    ),
                    input_row_count=int(len(prepared)),
                )
                continue
            fits.append(
                _fit_contract(
                    source_path,
                    prepared,
                    processed_path,
                    stage_root,
                    key_indicator=key_indicator,
                    timestamp=timestamp,
                    tic=tic,
                    filter_strength=filter_strength,
                    min_length_limit=min_length_limit,
                    merging_threshold=merging_threshold,
                    merging_metric=merging_metric,
                    merging_dynamic_constraint=merging_dynamic_constraint,
                    max_length_expectation=max_length_expectation,
                    dynamic_number=dynamic_number,
                )
            )

        shutil.rmtree(stage_root / "worker", ignore_errors=True)
        pooled_slopes = [slope for fit in fits for slope in fit.slopes]
        if not pooled_slopes:
            raise ValueError("valid dataset produced zero final segments")
        thresholds = [
            float(value)
            for value in util.calculate_slope_thresholds(
                pooled_slopes, dynamic_number, risk_bond=0.1
            )
        ]
        contracts = {
            fit.contract: _build_contract_outputs(
                fit,
                stage_root,
                valid_root,
                dynamic_number=dynamic_number,
                thresholds=thresholds,
            )
            for fit in fits
        }
        for fit in fits:
            contract = contracts[fit.contract]
            if contract.total_row_count != contract.input_row_count:
                raise ValueError(f"{fit.source_path} failed output row accounting")

        manifest = SliceManifest(
            valid_path=str(valid_root),
            contracts=contracts,
            skipped_contracts=skipped,
            calibration={
                "fit_scope": "valid_all_contracts",
                "participating_contracts": sorted(contracts),
                "skipped_contracts": sorted(skipped),
                "skipped_contract_details": [
                    skipped[contract].to_dict() for contract in sorted(skipped)
                ],
                "final_segment_count": len(pooled_slopes),
                "dynamic_number": dynamic_number,
                "labeling_method": "slope",
                "shared_thresholds": thresholds,
                "thresholds": thresholds,
                "slope_statistics": _slope_statistics(pooled_slopes),
            },
        )
        manifest.rebuild_labels()
        for label in range(dynamic_number):
            manifest.labels.setdefault(
                f"label_{label}", SliceLabelManifest(label=f"label_{label}")
            )
        manifest.sort()
        _publish(valid_root, stage_root, manifest_path, manifest)
        return manifest
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def _publish(
    valid_root: Path,
    stage_root: Path,
    manifest_path: Path,
    manifest: SliceManifest,
) -> None:
    backup_root = valid_root / f".valid-cross-contract-backup-{uuid.uuid4().hex}"
    manifest_stage_path = valid_root / f".slice-manifest-{uuid.uuid4().hex}.json"
    published_names: list[str] = []
    try:
        _backup_published_outputs(valid_root, stage_root, backup_root)
        for child in sorted(stage_root.iterdir(), key=lambda path: path.name):
            shutil.move(str(child), str(valid_root / child.name))
            published_names.append(child.name)
        manifest_stage_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_stage_path.replace(manifest_path)
    except Exception:
        if manifest_stage_path.exists():
            manifest_stage_path.unlink()
        for name in published_names:
            published_path = valid_root / name
            if published_path.is_dir():
                shutil.rmtree(published_path)
            elif published_path.exists():
                published_path.unlink()
        if backup_root.exists():
            for child in sorted(backup_root.iterdir(), key=lambda path: path.name):
                shutil.move(str(child), str(valid_root / child.name))
        shutil.rmtree(backup_root, ignore_errors=True)
        raise
    else:
        shutil.rmtree(backup_root, ignore_errors=True)
        shutil.rmtree(stage_root, ignore_errors=True)


build_valid_cross_contract_labels = build_valid_dataset
run_valid_dataset = build_valid_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--valid_dir", "--data_dir", dest="valid_dir", type=Path, required=True
    )
    parser.add_argument("--dynamic_number", type=int, default=5)
    parser.add_argument("--labeling_method", default="slope")
    parser.add_argument("--timestamp", default="timestamp")
    parser.add_argument("--tic", default="symbol")
    parser.add_argument("--min_length_limit", type=int, default=288)
    parser.add_argument("--merging_threshold", type=float, default=0.0003)
    parser.add_argument("--merging_dynamic_constraint", type=int, default=1)
    parser.add_argument("--max_length_expectation", type=int, default=864)
    return parser


def main(args: list[str] | None = None) -> None:
    parsed = build_parser().parse_args(args)
    build_valid_dataset(
        parsed.valid_dir,
        dynamic_number=parsed.dynamic_number,
        labeling_method=parsed.labeling_method,
        timestamp=parsed.timestamp,
        tic=parsed.tic,
        min_length_limit=parsed.min_length_limit,
        merging_threshold=parsed.merging_threshold,
        merging_dynamic_constraint=parsed.merging_dynamic_constraint,
        max_length_expectation=parsed.max_length_expectation,
    )


if __name__ == "__main__":
    main()
