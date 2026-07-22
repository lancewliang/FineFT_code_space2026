from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl

from operator_futures.commodity.schema import get_reward_execution_columns
from operator_futures.data_quality import DataQualityValidator
from operator_futures.feature_selection.cor_util import select_feature
from operator_futures.feature_selection.muti_contract.metrics import (
    DEFAULT_WINDOWS_LIST,
    aggregate_metric_frames,
    calculate_metric_frame,
)
from operator_futures.feature_selection.manifests import (
    FeatureSelectionContractRecord,
    FeatureSelectionManifest,
    FeatureSelectionResult,
    FilteredOutputRecord,
)


NON_STATE_COLUMNS = {"timestamp", "trading_day", "TradingDay", "symbol", "contract"}


def _stage_input_dir(
    root_path: Path, split_path: str, target_freq: str, symbol: str, stage: str
) -> Path:
    return root_path / split_path / target_freq / symbol / stage


def _stage_output_dir(
    root_path: Path, save_path: str, target_freq: str, symbol: str, stage: str
) -> Path:
    return root_path / save_path / target_freq / symbol / stage


def _load_contract_frames(input_dir: Path) -> dict[str, pl.DataFrame]:
    if not input_dir.exists():
        raise FileNotFoundError(f"split input directory does not exist: {input_dir}")
    paths = sorted(input_dir.glob("*.feather"))
    if not paths:
        raise FileNotFoundError(
            f"split input directory contains no contract feather files: {input_dir}"
        )
    return {path.stem: pl.read_ipc(path) for path in paths}


def _state_features(df: pl.DataFrame, *, orderbook_depth: int) -> list[str]:
    reward = set(get_reward_execution_columns(orderbook_depth))
    return [
        column
        for column in df.columns
        if column not in reward and column not in NON_STATE_COLUMNS
    ]


def _load_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"feature list file does not exist: {path}")
    values = np.load(path, allow_pickle=True).tolist()
    values = [str(value) for value in values]
    if not values:
        raise ValueError(f"feature list is empty: {path}")
    return values


def _ordered_filter_features(
    frames: dict[str, pl.DataFrame],
    aggregate: pl.DataFrame,
    feature_universe: list[str],
    *,
    min_abs_ic: float,
    max_metric_std: float,
    max_correlation: float,
    composite_drop_ratio: float = 0.1,
) -> tuple[list[str], dict[str, list[str]]]:
    if composite_drop_ratio < 0 or composite_drop_ratio >= 1:
        raise ValueError("composite_drop_ratio must be in [0, 1)")

    selected = aggregate.filter(pl.col("feature").is_in(feature_universe))
    hard = selected.filter(pl.col("RankIC_Mean").abs() >= min_abs_ic)[
        "feature"
    ].to_list()
    if not hard:
        raise ValueError("feature selection produced an empty list after Hard Filter")

    stability = (
        selected.filter(pl.col("feature").is_in(hard))
        .filter(pl.col("IC_Std") <= max_metric_std)
        ["feature"]
        .to_list()
    )
    if not stability:
        raise ValueError(
            "feature selection produced an empty list after Stability Filter"
        )

    scored_input = selected.filter(pl.col("feature").is_in(stability))
    secondary_score = (
        pl.col("Permutation Importance_Mean").fill_null(0.0)
        + pl.col("Sharpe_Mean").abs().fill_null(0.0)
    )
    if "SHAP Importance_Mean" in scored_input.columns:
        secondary_score = secondary_score + pl.col("SHAP Importance_Mean").fill_null(
            0.0
        )

    scored = (
        scored_input.with_columns(
            [
                pl.col("RankIC_Mean")
                .abs()
                .fill_null(0.0)
                .alias("Composite RankIC Score"),
                secondary_score.alias("Composite Secondary Score"),
                pl.col("CatBoost Importance_Mean")
                .fill_null(0.0)
                .alias("Composite Importance Score"),
            ]
        )
        .with_columns(
            (
                pl.col("Composite RankIC Score")
                + pl.col("Composite Secondary Score")
                + pl.col("Composite Importance Score")
            ).alias("Composite Score")
        )
        .sort(
            [
                "Composite RankIC Score",
                "Composite Secondary Score",
                "Composite Importance Score",
            ],
            descending=[True, True, True],
        )
    )
    drop_count = min(
        math.ceil(scored.height * composite_drop_ratio),
        max(scored.height - 1, 0),
    )
    kept = scored.head(scored.height - drop_count) if drop_count else scored
    dropped = scored.tail(drop_count)["feature"].to_list() if drop_count else []
    composite = kept["feature"].to_list()
    if not composite:
        raise ValueError(
            "feature selection produced an empty list after Composite Score"
        )

    combined = pl.concat(
        [frame.select(composite) for frame in frames.values()], how="vertical"
    )
    correlation = select_feature(
        features=composite, df=combined, theshold=max_correlation
    )
    if not correlation:
        raise ValueError(
            "feature selection produced an empty list after Correlation Filter"
        )
    return correlation, {
        "Hard Filter": hard,
        "Stability Filter": stability,
        "Composite Score": composite,
        "Composite Score Dropped": dropped,
        "Correlation Filter": correlation,
    }


def _validate_contract_frame(
    frame: pl.DataFrame,
    *,
    stage: str,
    contract: str,
    feature_universe: list[str],
) -> None:
    DataQualityValidator.validate_no_illegal_values(
        frame,
        stage=f"{stage}_feature_selection_input",
        feature_name="FEATURE_SELECTION",
        contract=contract,
        trading_day="-",
        columns=["mark_price", *feature_universe],
    )


def _write_filtered_outputs(
    frames: dict[str, pl.DataFrame],
    output_dir: Path,
    selected_features: list[str],
    *,
    symbol: str,
    orderbook_depth: int,
) -> list[FilteredOutputRecord]:
    reward_columns = get_reward_execution_columns(orderbook_depth)
    outputs: list[FilteredOutputRecord] = []
    for contract, frame in frames.items():
        missing = [feature for feature in selected_features if feature not in frame.columns]
        if missing:
            raise ValueError(
                f"contract {contract} is missing selected feature columns: {missing}"
            )
        reward_present = [column for column in reward_columns if column in frame.columns]
        filtered = frame.select([*reward_present, *selected_features]).with_columns(
            pl.lit(symbol).alias("symbol")
        )
        contract_dir = output_dir / contract
        contract_dir.mkdir(parents=True, exist_ok=True)
        output_path = contract_dir / "df.feather"
        filtered.write_ipc(output_path)
        outputs.append(
            FilteredOutputRecord(
                contract=contract,
                output_path=str(output_path),
                output_row_count=filtered.height,
                output_column_count=len(filtered.columns),
            )
        )
    return outputs


def run_feature_selection(
    *,
    root_path,
    split_path: str = "PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
    save_path: str = "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
    symbol: str,
    target_freq: str,
    stage: str,
    orderbook_depth: int = 5,
    min_abs_ic: float = 0.01,
    max_metric_std: float = 1.0,
    max_correlation: float = 0.7,
    windows_list: list[int] | None = None,
    composite_drop_ratio: float = 0.1,
) -> FeatureSelectionResult:
    if stage not in {"train", "valid"}:
        raise ValueError("stage must be 'train' or 'valid'")

    windows_list = list(DEFAULT_WINDOWS_LIST if windows_list is None else windows_list)
    root_path = Path(root_path)
    input_dir = _stage_input_dir(root_path, split_path, target_freq, symbol, stage)
    output_dir = _stage_output_dir(root_path, save_path, target_freq, symbol, stage)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _load_contract_frames(input_dir)

    if stage == "train":
        first_frame = next(iter(frames.values()))
        feature_universe = _state_features(first_frame, orderbook_depth=orderbook_depth)
        train_feature_file = None
    else:
        train_feature_file = (
            _stage_output_dir(root_path, save_path, target_freq, symbol, "train")
            / "state_features.npy"
        )
        feature_universe = _load_feature_list(train_feature_file)
    if not feature_universe:
        raise ValueError(f"{stage} feature universe is empty")

    per_contract_dir = output_dir / "per_contract"
    per_contract_dir.mkdir(parents=True, exist_ok=True)
    metric_frames = []
    per_contract: list[FeatureSelectionContractRecord] = []
    for contract, frame in frames.items():
        missing = [feature for feature in feature_universe if feature not in frame.columns]
        if missing:
            raise ValueError(
                f"contract {contract} is missing required feature columns: {missing}"
            )
        _validate_contract_frame(
            frame,
            stage=stage,
            contract=contract,
            feature_universe=feature_universe,
        )
        metrics = calculate_metric_frame(frame, feature_universe, windows_list=windows_list)
        metric_path = per_contract_dir / f"{contract}_metrics.csv"
        metrics.write_csv(metric_path)
        metric_frames.append(metrics)
        per_contract.append(
            FeatureSelectionContractRecord(
                contract=contract,
                input_path=str(input_dir / f"{contract}.feather"),
                metric_path=str(metric_path),
            )
        )

    aggregate = aggregate_metric_frames(metric_frames)
    aggregate_path = output_dir / "aggregate_metrics.csv"
    aggregate.write_csv(aggregate_path)
    if stage == "valid":
        manifest = FeatureSelectionManifest(
            symbol=symbol,
            target_freq=target_freq,
            stage=stage,
            split_input_dir=str(input_dir),
            evaluated_feature_file=str(train_feature_file),
            evaluated_feature_count=len(feature_universe),
            evaluated_features=feature_universe,
            windows_list=windows_list,
            aggregate_metrics_path=str(aggregate_path),
            contracts=per_contract,
            report_only=True,
        )
        manifest_path = output_dir / "feature_selection_manifest.json"
        manifest.write_json(manifest_path)
        return FeatureSelectionResult(output_dir=output_dir, manifest=manifest)

    selected_features, filter_results = _ordered_filter_features(
        frames,
        aggregate,
        feature_universe,
        min_abs_ic=min_abs_ic,
        max_metric_std=max_metric_std,
        max_correlation=max_correlation,
        composite_drop_ratio=composite_drop_ratio,
    )
    selected_file = output_dir / "state_features.npy"
    np.save(selected_file, np.array(selected_features))
    filtered_outputs = _write_filtered_outputs(
        frames,
        output_dir,
        selected_features,
        symbol=symbol,
        orderbook_depth=orderbook_depth,
    )
    manifest = FeatureSelectionManifest(
        symbol=symbol,
        target_freq=target_freq,
        stage=stage,
        split_input_dir=str(input_dir),
        selected_feature_file=str(selected_file),
        selected_feature_count=len(selected_features),
        selected_features=selected_features,
        windows_list=windows_list,
        composite_drop_ratio=composite_drop_ratio,
        aggregate_metrics_path=str(aggregate_path),
        filter_results=filter_results,
        contracts=per_contract,
        filtered_outputs=filtered_outputs,
    )
    manifest_path = output_dir / "feature_selection_manifest.json"
    manifest.write_json(manifest_path)
    return FeatureSelectionResult(output_dir=output_dir, manifest=manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=Path, default=Path("."))
    parser.add_argument(
        "--split_path",
        type=str,
        default="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION",
    )
    parser.add_argument("--symbol", "--symbols", dest="symbol", type=str, required=True)
    parser.add_argument("--target_freq", type=str, required=True)
    parser.add_argument("--stage", choices=["train", "valid"], required=True)
    parser.add_argument("--orderbook_depth", type=int, default=5)
    parser.add_argument("--min_abs_ic", type=float, default=0.01)
    parser.add_argument("--max_metric_std", type=float, default=1.0)
    parser.add_argument("--max_correlation", type=float, default=0.7)
    parser.add_argument("--composite_drop_ratio", type=float, default=0.1)
    parser.add_argument(
        "--windows_list",
        "--windows",
        dest="windows_list",
        type=int,
        nargs="*",
        default=None,
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_feature_selection(
        root_path=args.root_path,
        split_path=args.split_path,
        save_path=args.save_path,
        symbol=args.symbol,
        target_freq=args.target_freq,
        stage=args.stage,
        orderbook_depth=args.orderbook_depth,
        min_abs_ic=args.min_abs_ic,
        max_metric_std=args.max_metric_std,
        max_correlation=args.max_correlation,
        windows_list=args.windows_list,
        composite_drop_ratio=args.composite_drop_ratio,
    )
