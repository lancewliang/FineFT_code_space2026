from __future__ import annotations
from operator_futures.feature_selection.muti_contract.regime_audit import (
    compute_regime_quantiles,
    audit_regimes,
    audit_16_regimes,
    default_target_regime_bins,
)

import argparse
import json
import math
import re
from pathlib import Path
from typing import Sequence

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
    PersistenceDiagnostic,
)



def _parse_windows_list(value: list[int | str] | str | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    result: list[int] = []
    for item in value:
        if isinstance(item, int):
            result.append(item)
        elif isinstance(item, str):
            for part in item.split(","):
                part = part.strip()
                if part:
                    result.append(int(part))
    return result if result else None


def _parse_target_regime_bins(
    value: list[tuple[int, int]] | list[str] | str | None,
) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    result: list[tuple[int, int]] = []
    for item in value:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            result.append((int(item[0]), int(item[1])))
        elif isinstance(item, str):
            cleaned = item.strip("()[] ")
            for part in cleaned.split():
                part = part.strip("()[] ,")
                if not part:
                    continue
                if "," in part:
                    s, v = part.split(",", 1)
                    result.append((int(s.strip()), int(v.strip())))
    return result if result else None


NON_STATE_COLUMNS = {"timestamp", "trading_day", "TradingDay", "symbol", "contract"}
DEFAULT_PERSISTENCE_FILTER_PATTERN = r"_log_return_(1|2)$"


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
    frames = {}
    for path in paths:
        df = pl.read_ipc(path)
        if "timestamp" in df.columns:
            df = df.sort("timestamp")
        frames[path.stem] = df
    return frames


def _state_features(df: pl.DataFrame, *, orderbook_depth: int) -> list[str]:
    reward = set(get_reward_execution_columns(orderbook_depth))
    schema = df.schema
    return [
        column
        for column in df.columns
        if column not in reward
        and column not in NON_STATE_COLUMNS
        and not column.endswith("timestamp")
        and not column.endswith("_right")
        and (schema[column].is_numeric() or schema[column] == pl.Boolean)
    ]


def _load_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"feature list file does not exist: {path}")
    values = np.load(path, allow_pickle=True).tolist()
    values = [str(value) for value in values]
    if not values:
        raise ValueError(f"feature list is empty: {path}")
    return values


def _lag1_autocorrelation(values: Sequence[float] | np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return None
    left = values[:-1]
    right = values[1:]
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(value):
        return None
    return value


def _directional_half_life_bars(autocorrelation: float | None) -> float | None:
    if autocorrelation is None:
        return None
    if autocorrelation <= 0.0:
        return 0.0
    if autocorrelation >= 1.0:
        return None
    return float(math.log(0.5) / math.log(autocorrelation))


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _calculate_persistence_diagnostics(
    frames: dict[str, pl.DataFrame],
    feature_universe: list[str],
    *,
    active_feature_pattern: str,
) -> list[PersistenceDiagnostic]:
    active_regex = re.compile(active_feature_pattern)
    rows: list[PersistenceDiagnostic] = []
    for feature in feature_universe:
        autocorrelations: list[float] = []
        half_lives: list[float] = []
        for frame in frames.values():
            autocorrelation = _lag1_autocorrelation(frame[feature].to_numpy())
            if autocorrelation is not None:
                autocorrelations.append(autocorrelation)
            half_life = _directional_half_life_bars(autocorrelation)
            if half_life is not None:
                half_lives.append(half_life)
        rows.append(
            {
                "feature": feature,
                "lag1_autocorrelation_median": _median_or_none(autocorrelations),
                "half_life_bars_median": _median_or_none(half_lives),
                "active_filter": bool(active_regex.search(feature)),
            }
        )
    return rows


def _filter_by_persistence(
    features: list[str],
    diagnostics_by_feature: dict[str, PersistenceDiagnostic],
    *,
    min_half_life_bars: float,
) -> tuple[list[str], list[str]]:
    if min_half_life_bars <= 0.0:
        return features, []

    kept: list[str] = []
    dropped: list[str] = []
    for feature in features:
        diagnostics = diagnostics_by_feature.get(feature)
        half_life = (
            diagnostics.get("half_life_bars_median")
            if diagnostics is not None
            else None
        )
        if (
            diagnostics is not None
            and diagnostics.get("active_filter") is True
            and half_life is not None
            and float(half_life) < min_half_life_bars
        ):
            dropped.append(feature)
        else:
            kept.append(feature)
    return kept, dropped


def _ordered_filter_features(
    frames: dict[str, pl.DataFrame],
    aggregate: pl.DataFrame,
    feature_universe: list[str],
    *,
    min_abs_ic: float,
    max_metric_std: float,
    max_correlation: float,
    min_rank_ic_ir: float = 0.0,
    composite_drop_ratio: float = 0.1,
    min_half_life_bars: float = 0.0,
    persistence_diagnostics: list[PersistenceDiagnostic] | None = None,
    rank_ic_mode: str = "absolute",
) -> tuple[list[str], dict[str, list[str]]]:
    if composite_drop_ratio < 0 or composite_drop_ratio >= 1:
        raise ValueError("composite_drop_ratio must be in [0, 1)")
    if rank_ic_mode not in {"absolute", "signed"}:
        raise ValueError("rank_ic_mode must be 'absolute' or 'signed'")

    selected = aggregate.filter(pl.col("feature").is_in(feature_universe))
    rank_ic_filter = (
        pl.col("RankIC_Mean") >= min_abs_ic
        if rank_ic_mode == "signed"
        else pl.col("RankIC_Mean").abs() >= min_abs_ic
    )
    hard = selected.filter(rank_ic_filter)[
        "feature"
    ].to_list()
    if not hard:
        raise ValueError("feature selection produced an empty list after Hard Filter")

    persistence = hard
    persistence_dropped: list[str] = []
    if min_half_life_bars > 0.0:
        diagnostics_by_feature = {
            str(row["feature"]): row for row in (persistence_diagnostics or [])
        }
        persistence, persistence_dropped = _filter_by_persistence(
            hard,
            diagnostics_by_feature,
            min_half_life_bars=min_half_life_bars,
        )
        if not persistence:
            raise ValueError(
                "feature selection produced an empty list after Persistence Filter"
            )

    stability_cond = pl.col("IC_Std") <= max_metric_std
    if "RankIC_Std" in selected.columns:
        stability_cond = stability_cond & (pl.col("RankIC_Std") <= max_metric_std)
    if min_rank_ic_ir > 0.0 and "RankIC_Std" in selected.columns:
        rank_ic_ir = pl.col("RankIC_Mean").abs() / (pl.col("RankIC_Std") + 1e-6)
        stability_cond = stability_cond & (rank_ic_ir >= min_rank_ic_ir)

    stability = (
        selected.filter(pl.col("feature").is_in(persistence))
        .filter(stability_cond)
        ["feature"]
        .to_list()
    )
    if not stability:
        raise ValueError(
            "feature selection produced an empty list after Stability Filter"
        )

    scored_input = selected.filter(pl.col("feature").is_in(stability))
    height = float(max(scored_input.height, 1))
    secondary_raw = (
        pl.col("Permutation Importance_Mean").fill_null(0.0)
        + pl.col("Sharpe_Mean").abs().fill_null(0.0)
    )
    if "SHAP Importance_Mean" in scored_input.columns:
        secondary_raw = secondary_raw + pl.col("SHAP Importance_Mean").fill_null(0.0)

    # Normalize secondary components to percentile rank [0, 1] to prevent magnitude distortion
    secondary_score = (secondary_raw.rank() / height) + (pl.col("CatBoost Importance_Mean").fill_null(0.0).rank() / height)
    rank_ic_score = (
        pl.col("RankIC_Mean").abs()
        if rank_ic_mode == "absolute"
        else pl.col("RankIC_Mean")
    ).fill_null(0.0)

    scored = (
        scored_input.with_columns(
            [
                rank_ic_score.alias("Composite RankIC Score"),
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
    filter_results = {"Hard Filter": hard}
    if min_half_life_bars > 0.0:
        filter_results.update(
            {
                "Persistence Filter": persistence,
                "Persistence Filter Dropped": persistence_dropped,
            }
        )
    filter_results.update(
        {
            "Stability Filter": stability,
            "Composite Score": composite,
            "Composite Score Dropped": dropped,
            "Correlation Filter": correlation,
        }
    )
    return correlation, filter_results


def _apply_feature_blacklist(
    selected_features: list[str], feature_blacklist: list[str] | None
) -> tuple[list[str], list[str]]:
    if not feature_blacklist:
        return selected_features, []
    blacklist = set(feature_blacklist)
    filtered = [feature for feature in selected_features if feature not in blacklist]
    dropped = [feature for feature in selected_features if feature in blacklist]
    return filtered, dropped


def _apply_feature_ablation_patterns(
    features: list[str], patterns: Sequence[str]
) -> tuple[list[str], list[str]]:
    if not patterns:
        return features, []
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    kept = [
        feature
        for feature in features
        if not any(pattern.search(feature) for pattern in compiled_patterns)
    ]
    dropped = [feature for feature in features if feature not in kept]
    return kept, dropped


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
    min_rank_ic_ir: float = 0.0,
    windows_list: list[int] | None = None,
    composite_drop_ratio: float = 0.1,
    feature_blacklist: list[str] | None = None,
    feature_ablation_patterns: Sequence[str] | None = None,
    mandatory_state_features: list[str] | None = None,
    min_half_life_bars: float = 0.0,
    persistence_filter_pattern: str = DEFAULT_PERSISTENCE_FILTER_PATTERN,
    rank_ic_mode: str = "absolute",
    enable_conditional_anchors: bool = True,
    regime_bins: int = 4,
    target_regime_bins: Sequence[tuple[int, int]] | None = None,
) -> FeatureSelectionResult:
    if stage not in {"train", "valid"}:
        raise ValueError("stage must be 'train' or 'valid'")
    if rank_ic_mode not in {"absolute", "signed"}:
        raise ValueError("rank_ic_mode must be 'absolute' or 'signed'")

    mandatory_features = list(mandatory_state_features or [])
    feature_ablation_patterns = tuple(feature_ablation_patterns or ())
    if feature_blacklist and mandatory_features:
        conflict = sorted(set(feature_blacklist).intersection(mandatory_features))
        if conflict:
            raise ValueError(
                f"feature blacklist contains mandatory state feature column(s): {conflict}"
            )
    ablation_conflict = [
        feature
        for feature in mandatory_features
        if any(
            re.search(pattern, feature, flags=re.IGNORECASE)
            for pattern in feature_ablation_patterns
        )
    ]
    if ablation_conflict:
        raise ValueError(
            "feature ablation patterns target mandatory state feature(s): "
            f"{ablation_conflict}"
        )

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
    feature_universe, ablation_dropped = _apply_feature_ablation_patterns(
        feature_universe, feature_ablation_patterns
    )
    if not feature_universe:
        raise ValueError(f"{stage} feature universe is empty after feature ablation")

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
        candidate_universe = [f for f in feature_universe if f not in mandatory_features]
        metrics = calculate_metric_frame(frame, candidate_universe, windows_list=windows_list)
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

    train_manifest_path = _stage_output_dir(root_path, save_path, target_freq, symbol, "train") / "feature_selection_manifest.json"
    if stage == "train":
        regime_quantiles = compute_regime_quantiles(frames, num_bins=regime_bins)
        effective_target_bins = (
            list(target_regime_bins)
            if target_regime_bins is not None
            else default_target_regime_bins(regime_bins, regime_bins)
        )
    else:
        regime_quantiles = None
        effective_target_bins = list(target_regime_bins) if target_regime_bins is not None else None
        if train_manifest_path.exists():
            try:
                train_manifest_data = json.loads(train_manifest_path.read_text(encoding="utf-8"))
                regime_quantiles = train_manifest_data.get("regime_quantiles")
                if "regime_bins" in train_manifest_data and train_manifest_data["regime_bins"] is not None:
                    regime_bins = int(train_manifest_data["regime_bins"])
                if (
                    effective_target_bins is None
                    and "target_regime_bins" in train_manifest_data
                    and train_manifest_data["target_regime_bins"] is not None
                ):
                    effective_target_bins = [
                        (int(item[0]), int(item[1]))
                        for item in train_manifest_data["target_regime_bins"]
                    ]
            except Exception:
                pass
        if regime_quantiles is None:
            regime_quantiles = compute_regime_quantiles(frames, num_bins=regime_bins)
        if effective_target_bins is None:
            num_s = len(regime_quantiles["slope"]) + 1
            num_v = len(regime_quantiles["volatility"]) + 1
            effective_target_bins = default_target_regime_bins(num_s, num_v)

    candidate_universe = [f for f in feature_universe if f not in mandatory_features]
    regime_audit_df, retained_anchors, retention_details = audit_regimes(
        frames,
        candidate_universe,
        regime_quantiles,
        windows_list,
        target_regime_bins=effective_target_bins,
        min_abs_ic=min_abs_ic,
        enable_conditional_anchors=enable_conditional_anchors,
    )
    regime_audit_path = output_dir / "regime_audit_metrics.csv"
    regime_audit_df.write_csv(regime_audit_path)

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
            feature_ablation_patterns=list(feature_ablation_patterns),
            rank_ic_mode=rank_ic_mode,
            report_only=True,
            regime_bins=regime_bins,
            target_regime_bins=effective_target_bins,
            regime_quantiles=regime_quantiles,
            regime_audit_path=str(regime_audit_path),
            conditional_anchors_retained=retention_details if retention_details else None,
        )
        manifest_path = output_dir / "feature_selection_manifest.json"
        manifest.write_json(manifest_path)
        return FeatureSelectionResult(output_dir=output_dir, manifest=manifest)

    candidate_universe = [f for f in feature_universe if f not in mandatory_features]
    persistence_diagnostics = None
    if min_half_life_bars > 0.0:
        persistence_diagnostics = _calculate_persistence_diagnostics(
            frames,
            candidate_universe,
            active_feature_pattern=persistence_filter_pattern,
        )
    selected_features, filter_results = _ordered_filter_features(
        frames,
        aggregate,
        candidate_universe,
        min_abs_ic=min_abs_ic,
        max_metric_std=max_metric_std,
        max_correlation=max_correlation,
        min_rank_ic_ir=min_rank_ic_ir,
        composite_drop_ratio=composite_drop_ratio,
        min_half_life_bars=min_half_life_bars,
        persistence_diagnostics=persistence_diagnostics,
        rank_ic_mode=rank_ic_mode,
    )
    if ablation_dropped:
        filter_results = {
            "Feature Ablation Dropped": ablation_dropped,
            **filter_results,
        }
    selected_features, blacklisted_features = _apply_feature_blacklist(
        selected_features, feature_blacklist
    )
    if feature_blacklist and not selected_features:
        raise ValueError(
            "feature selection produced an empty list after Feature Blacklist"
        )
    if blacklisted_features:
        filter_results = {
            **filter_results,
            "Feature Blacklist Dropped": blacklisted_features,
        }
    normal_selected = [f for f in selected_features if f not in mandatory_features]
    if enable_conditional_anchors and retained_anchors:
        newly_retained = [a for a in retained_anchors if a in candidate_universe and a not in normal_selected]
        if newly_retained:
            normal_selected.extend(newly_retained)
            filter_results["Conditional Anchor Retention"] = newly_retained
    selected_features = normal_selected + mandatory_features

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
        feature_blacklist=(
            list(feature_blacklist) if feature_blacklist is not None else None
        ),
        feature_ablation_patterns=list(feature_ablation_patterns),
        rank_ic_mode=rank_ic_mode,
        mandatory_state_features=mandatory_features if mandatory_features else None,
        persistence_filter=(
            {
                "min_half_life_bars": float(min_half_life_bars),
                "active_feature_pattern": persistence_filter_pattern,
            }
            if min_half_life_bars > 0.0
            else None
        ),
        persistence_diagnostics=persistence_diagnostics,
        aggregate_metrics_path=str(aggregate_path),
        filter_results=filter_results,
        contracts=per_contract,
        filtered_outputs=filtered_outputs,
        regime_bins=regime_bins,
        target_regime_bins=effective_target_bins,
        regime_quantiles=regime_quantiles,
        regime_audit_path=str(regime_audit_path),
        conditional_anchors_retained=retention_details if retention_details else None,
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
    parser.add_argument("--min_rank_ic_ir", type=float, default=0.0)
    parser.add_argument("--composite_drop_ratio", type=float, default=0.1)
    parser.add_argument("--feature_blacklist", nargs="*", default=None)
    parser.add_argument(
        "--feature_ablation_patterns",
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--rank_ic_mode",
        choices=["absolute", "signed"],
        default="absolute",
    )
    parser.add_argument("--min_half_life_bars", type=float, default=0.0)
    parser.add_argument(
        "--persistence_filter_pattern",
        type=str,
        default=DEFAULT_PERSISTENCE_FILTER_PATTERN,
    )
    parser.add_argument(
        "--mandatory_state_features",
        "--mandatory_features",
        dest="mandatory_state_features",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--windows_list",
        "--windows",
        dest="windows_list",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--regime_bins",
        "--num_regime_bins",
        dest="regime_bins",
        type=int,
        default=4,
        help="Number of bins per dimension for market state regime audit (e.g. 4 for 4x4, 3 for 3x3)",
    )
    parser.add_argument(
        "--target_regime_bins",
        nargs="*",
        default=None,
        help="Target regime bins for conditional anchor retention (e.g. '0,0' '2,0' or '0,0 3,0 0,1 3,1')",
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
        min_rank_ic_ir=args.min_rank_ic_ir,
        windows_list=_parse_windows_list(args.windows_list),
        composite_drop_ratio=args.composite_drop_ratio,
        feature_blacklist=args.feature_blacklist,
        feature_ablation_patterns=args.feature_ablation_patterns,
        mandatory_state_features=args.mandatory_state_features,
        min_half_life_bars=args.min_half_life_bars,
        persistence_filter_pattern=args.persistence_filter_pattern,
        rank_ic_mode=args.rank_ic_mode,
        regime_bins=args.regime_bins,
        target_regime_bins=_parse_target_regime_bins(args.target_regime_bins),
    )
