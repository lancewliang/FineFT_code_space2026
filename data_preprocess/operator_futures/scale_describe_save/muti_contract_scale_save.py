import argparse
from dataclasses import asdict, dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import polars as pl

from operator_futures.scale_describe_save.scale_save import (
    configure_logging,
    validate_no_nan,
)


logger = logging.getLogger(__name__)
SPLIT_STAGES = ("train", "valid", "test")


@dataclass(frozen=True)
class ScalerFeatureStats:
    feature: str
    center: float
    scale: float
    scale_method: str
    q25: float
    q75: float
    iqr: float
    std: float
    min: float
    max: float
    row_count: int
    fallback_reason: str | None = None
    is_log_transformed: bool = False


@dataclass(frozen=True)
class ScaleManifest:
    symbol: str
    target_freq: str
    scaler_version: str
    fit_scope: str
    feature_list_path: str
    train_input_files: list[str]
    row_count: int
    clip_enabled: bool
    clip_min: float | None
    clip_max: float | None
    features: list[ScalerFeatureStats]
    passthrough_state_features: list[str] = field(default_factory=list)
    scale_method: str = "robust"
    rolling_window: int = 48
    rolling_min_periods: int = 1
    clip_mode: str = "hard"
    soft_clip_m: float = 4.0
    rl_feature_list_path: str = ""
    vae_feature_list_path: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["clip"] = {
            "enabled": self.clip_enabled,
            "min": self.clip_min,
            "max": self.clip_max,
            "mode": self.clip_mode,
            "soft_clip_m": self.soft_clip_m,
        }
        payload["scale_method"] = self.scale_method
        payload["rolling_window"] = self.rolling_window
        return payload


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default=".", help="the path of storing the data")
parser.add_argument(
    "--data_path",
    type=str,
    default="PREPROCESS_DATASET/commodity-futures/SPLIT-TRAIN-VALID-TEST",
    help="the split train/valid/test input root",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/",
    help="the scale-save output root",
)
parser.add_argument("--symbols", type=str, default="fu", help="the commodity symbol")
parser.add_argument("--start_date", type=str, default=None, help="accepted for pipeline compatibility")
parser.add_argument("--end_date", type=str, default=None, help="accepted for pipeline compatibility")
parser.add_argument(
    "--target_freq",
    type=str,
    default="5min",
    choices=["10s", "1min", "5min", "10min", "30min", "1H", "1D"],
)
parser.add_argument("--clip_theshold", type=float, default=10, help="scale mean clipping threshold")
parser.add_argument("--base", type=float, default=10, help="scaling log base")
parser.add_argument(
    "--clip_min",
    type=float,
    default=-20.0,
    help="minimum clipped robust-scaled value",
)
parser.add_argument(
    "--clip_max",
    type=float,
    default=20.0,
    help="maximum clipped robust-scaled value",
)
parser.add_argument(
    "--disable_clip",
    action="store_true",
    help="disable robust scaler clipping",
)
parser.add_argument(
    "--scale_method",
    type=str,
    default="robust",
    choices=["rolling_zscore", "robust"],
    help="feature scaling method: rolling_zscore (adaptive) or robust (static train-wide)",
)
parser.add_argument(
    "--rolling_window",
    type=int,
    default=48,
    help="rolling window size for rolling_zscore",
)
parser.add_argument(
    "--rolling_min_periods",
    type=int,
    default=1,
    help="minimum observations for rolling window",
)
parser.add_argument(
    "--clip_mode",
    type=str,
    default="hard",
    choices=["tanh", "hard"],
    help="clipping mode: tanh (continuous soft saturation) or hard (np.clip)",
)
parser.add_argument(
    "--soft_clip_m",
    type=float,
    default=4.0,
    help="saturation bound scale for tanh soft clipping: m * tanh(z / m)",
)
parser.add_argument(
    "--iqr_epsilon",
    type=float,
    default=1e-8,
    help="minimum usable IQR scale",
)
parser.add_argument(
    "--std_epsilon",
    type=float,
    default=1e-8,
    help="minimum usable std fallback scale",
)
parser.add_argument(
    "--market_type",
    type=str,
    default="commodity_futures",
    choices=["crypto_futures", "commodity_futures"],
    help="the market type of the preprocessed data",
)
parser.add_argument(
    "--orderbook_depth",
    type=int,
    default=5,
    help="the available orderbook depth",
)
parser.add_argument(
    "--feature_selection_dir",
    type=str,
    default=None,
    help="Directory containing rl_state_features.npy and vae_state_features.npy",
)
parser.add_argument(
    "--rl_feature_list_path",
    type=str,
    default=None,
    help="Explicit path to rl_state_features.npy",
)
parser.add_argument(
    "--vae_feature_list_path",
    type=str,
    default=None,
    help="Explicit path to vae_state_features.npy",
)
parser.add_argument(
    "--passthrough_features",
    dest="passthrough_features",
    nargs="*",
    default=None,
    help="features to passthrough without scaling",
)


def resolve_path(root_path: Path, path: str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return root_path / resolved


def resolve_feature_paths(root_path: Path, args) -> tuple[Path, Path]:
    if args.feature_selection_dir is not None:
        fs_dir = resolve_path(root_path, args.feature_selection_dir)
    else:
        fs_dir = (
            root_path
            / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION"
            / args.target_freq
            / args.symbols
            / "train"
        )
    rl_path = (
        resolve_path(root_path, args.rl_feature_list_path)
        if args.rl_feature_list_path is not None
        else fs_dir / "rl_state_features.npy"
    )
    vae_path = (
        resolve_path(root_path, args.vae_feature_list_path)
        if args.vae_feature_list_path is not None
        else fs_dir / "vae_state_features.npy"
    )
    return rl_path, vae_path


def load_state_features(feature_list_path: Path) -> list[str]:
    state_features = np.load(feature_list_path, allow_pickle=True).tolist()
    if not state_features:
        raise ValueError(f"state feature list is empty: {feature_list_path}")
    return state_features


def load_dual_stream_state_features(
    rl_path: Path, vae_path: Path
) -> tuple[list[str], list[str], list[str]]:
    if not rl_path.exists():
        raise FileNotFoundError(f"Missing required RL state feature file: {rl_path}")
    if not vae_path.exists():
        raise FileNotFoundError(f"Missing required VAE state feature file: {vae_path}")
    rl_features = load_state_features(rl_path)
    vae_features = load_state_features(vae_path)
    union_features = list(dict.fromkeys(rl_features + vae_features))
    return rl_features, vae_features, union_features


def split_stage_inputs(data_root: Path, symbol: str, target_freq: str) -> list[tuple[str, str, Path]]:
    inputs = []
    for stage in SPLIT_STAGES:
        stage_dir = data_root / target_freq / symbol / stage
        if not stage_dir.exists():
            continue
        for input_file in sorted(stage_dir.glob("*.feather")):
            inputs.append((stage, input_file.stem, input_file))
    return inputs


def reward_features_for(df: pl.DataFrame, args) -> list[str]:
    from operator_futures.commodity.schema import get_reward_execution_columns
    return [col for col in get_reward_execution_columns(args.orderbook_depth) if col in df.columns]
    


def validate_clip_args(args) -> None:
    if args.iqr_epsilon <= 0 or not math.isfinite(args.iqr_epsilon):
        raise ValueError("iqr_epsilon must be a finite positive value")
    if args.std_epsilon <= 0 or not math.isfinite(args.std_epsilon):
        raise ValueError("std_epsilon must be a finite positive value")
    if args.rolling_window <= 0:
        raise ValueError("rolling_window must be a positive integer")
    if args.rolling_min_periods <= 0:
        raise ValueError("rolling_min_periods must be a positive integer")
    if args.soft_clip_m <= 0 or not math.isfinite(args.soft_clip_m):
        raise ValueError("soft_clip_m must be a finite positive value")
    if args.disable_clip:
        return
    if not math.isfinite(args.clip_min) or not math.isfinite(args.clip_max):
        raise ValueError("clip_min and clip_max must be finite values")
    if args.clip_min >= args.clip_max:
        raise ValueError("clip_min must be less than clip_max")


def ensure_finite(value: float, feature: str, field: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite scaler statistic for feature={feature} field={field}: {value}")
    return value


def validate_state_features_present(
    df: pl.DataFrame,
    state_features: list[str],
    input_file: Path,
) -> None:
    missing_features = [feature for feature in state_features if feature not in df.columns]
    if missing_features:
        raise ValueError(
            f"missing selected state feature columns in {input_file}: {missing_features}"
        )


def preflight_validate_split_inputs(
    inputs: list[tuple[str, str, Path]],
    state_features: list[str],
) -> None:
    for stage, contract, input_file in inputs:
        logger.info(
            "Preflight validating split-stage input: stage=%s contract=%s input_file=%s",
            stage,
            contract,
            input_file,
        )
        df = pl.read_ipc(input_file)
        validate_no_nan(df, path=input_file, stage="input_preflight")
        validate_state_features_present(df, state_features, input_file)


def fit_feature_stats(
    feature: str,
    values: np.ndarray,
    iqr_epsilon: float,
    std_epsilon: float,
) -> ScalerFeatureStats:
    if values.size == 0:
        raise ValueError(f"no train values available for feature={feature}")
    is_log = is_log_transformed_feature(feature)
    if is_log:
        values = transform_volatility_values(values)
    q25 = ensure_finite(np.nanquantile(values, 0.25), feature, "q25")
    q75 = ensure_finite(np.nanquantile(values, 0.75), feature, "q75")
    center = ensure_finite(np.nanmedian(values), feature, "center")
    iqr = ensure_finite(q75 - q25, feature, "iqr")
    std = ensure_finite(np.nanstd(values, ddof=0), feature, "std")
    minimum = ensure_finite(np.nanmin(values), feature, "min")
    maximum = ensure_finite(np.nanmax(values), feature, "max")

    scale = iqr
    scale_method = "iqr"
    fallback_reason = None
    if scale <= iqr_epsilon:
        scale = std
        scale_method = "std"
        fallback_reason = "iqr_below_epsilon"
    if not math.isfinite(scale) or scale <= std_epsilon:
        scale = 1.0
        scale_method = "unit"
        fallback_reason = "std_nonfinite_or_below_epsilon"

    return ScalerFeatureStats(
        feature=feature,
        center=center,
        scale=ensure_finite(scale, feature, "scale"),
        scale_method=scale_method,
        q25=q25,
        q75=q75,
        iqr=iqr,
        std=std,
        min=minimum,
        max=maximum,
        row_count=int(values.size),
        fallback_reason=fallback_reason,
        is_log_transformed=is_log,
    )


def fit_robust_scaler(
    *,
    train_inputs: list[tuple[str, str, Path]],
    state_features: list[str],
    rl_feature_path: Path,
    vae_feature_path: Path,
    args,
) -> ScaleManifest:
    passthrough_targets = set(args.passthrough_features or [])
    passthrough_features = [f for f in state_features if f in passthrough_targets]
    scaled_features = [f for f in state_features if f not in passthrough_targets]

    values_by_feature: dict[str, list[np.ndarray]] = {
        feature: [] for feature in scaled_features
    }
    train_row_count = 0
    train_input_files = []
    for stage, contract, input_file in train_inputs:
        logger.info(
            "Loading train split for robust scaler fit: stage=%s contract=%s input_file=%s",
            stage,
            contract,
            input_file,
        )
        df = pl.read_ipc(input_file)
        validate_no_nan(df, path=input_file, stage="train_fit_input")
        validate_state_features_present(df, state_features, input_file)
        train_row_count += df.height
        train_input_files.append(str(input_file))
        for feature in scaled_features:
            values_by_feature[feature].append(
                df.get_column(feature).to_numpy().astype(float, copy=False)
            )

    if train_row_count == 0:
        raise ValueError(f"train split inputs contain no rows for symbol={args.symbols}")

    feature_stats = [
        fit_feature_stats(
            feature,
            np.concatenate(values_by_feature[feature]),
            args.iqr_epsilon,
            args.std_epsilon,
        )
        for feature in scaled_features
    ]
    return ScaleManifest(
        symbol=args.symbols,
        target_freq=args.target_freq,
        scaler_version=(
            "rolling_zscore_v1"
            if args.scale_method == "rolling_zscore"
            else "robust_v1"
        ),
        fit_scope="train_all_contracts",
        feature_list_path=str(rl_feature_path.parent),
        rl_feature_list_path=str(rl_feature_path),
        vae_feature_list_path=str(vae_feature_path),
        train_input_files=train_input_files,
        row_count=train_row_count,
        clip_enabled=not args.disable_clip,
        clip_min=None if args.disable_clip else float(args.clip_min),
        clip_max=None if args.disable_clip else float(args.clip_max),
        features=feature_stats,
        passthrough_state_features=passthrough_features,
        scale_method=args.scale_method,
        rolling_window=args.rolling_window,
        rolling_min_periods=args.rolling_min_periods,
        clip_mode=args.clip_mode,
        soft_clip_m=float(args.soft_clip_m),
    )


VOLATILITY_FEATURE_PATTERNS: tuple[str, ...] = (
    "realized_volatility",
    "rolling_volatility",
    "garman_klass_volatility",
    "parkinson_volatility",
    "historical_volatility",
    "bollinger_bandwidth",
)
NON_NEGATIVE_ACTIVITY_PATTERNS: tuple[str, ...] = (
    "relative_volume",
    "relative_amount",
    "vma_",
    "wvma_",
)
SIGNED_FEATURE_EXCLUSIONS: tuple[str, ...] = (
    "trend",
    "slope",
    "imblance",
    "log_return",
    "diff",
    "persistence",
    "interaction",
)
LOG_FEATURE_PATTERNS: tuple[str, ...] = (
    *VOLATILITY_FEATURE_PATTERNS,
    *NON_NEGATIVE_ACTIVITY_PATTERNS,
)
VOLATILITY_LOG_EPSILON: float = 1e-6


def is_log_transformed_feature(feature: str) -> bool:
    if any(pattern in feature for pattern in VOLATILITY_FEATURE_PATTERNS):
        return True
    if any(pattern in feature for pattern in NON_NEGATIVE_ACTIVITY_PATTERNS):
        return not any(excl in feature for excl in SIGNED_FEATURE_EXCLUSIONS)
    return False


def is_volatility_feature(feature: str) -> bool:
    return is_log_transformed_feature(feature)


def transform_volatility_values(values: np.ndarray) -> np.ndarray:
    if values.size > 0 and np.nanmin(values) < 0.0:
        return values
    return np.log(np.maximum(values, 0.0) + VOLATILITY_LOG_EPSILON)


FAT_TAILED_FEATURE_PATTERNS: tuple[str, ...] = (
    "vstd_",
    "spread_velocity",
    "shift_speed",
    "_increments",
    "spread_oe_max",
)
FAT_TAILED_CLIP_MIN: float = -4.0
FAT_TAILED_CLIP_MAX: float = 4.0


def resolve_feature_clip_bounds(
    feature: str,
    clip_min: float,
    clip_max: float,
) -> tuple[float, float]:
    for pattern in FAT_TAILED_FEATURE_PATTERNS:
        if pattern in feature:
            return max(clip_min, FAT_TAILED_CLIP_MIN), min(clip_max, FAT_TAILED_CLIP_MAX)
    return clip_min, clip_max


def apply_robust_scaler(
    df_state: pl.DataFrame,
    manifest: ScaleManifest,
) -> tuple[pl.DataFrame, dict[str, object]]:
    columns = {}
    clipped_by_feature = {}
    rows = df_state.height
    for stats in manifest.features:
        values = df_state.get_column(stats.feature).to_numpy().astype(float, copy=False)
        if stats.is_log_transformed:
            values = transform_volatility_values(values)

        if manifest.scale_method == "rolling_zscore":
            r_mean = (
                pl.Series(values)
                .rolling_mean(
                    window_size=manifest.rolling_window,
                    min_samples=manifest.rolling_min_periods,
                )
                .to_numpy()
            )
            r_std = (
                pl.Series(values)
                .rolling_std(
                    window_size=manifest.rolling_window,
                    min_samples=2,
                )
                .to_numpy()
            )
            fallback_scale = (
                stats.scale
                if stats.scale > 1e-8
                else (stats.std if stats.std > 1e-8 else 1.0)
            )
            denom = np.where(np.isnan(r_std) | (r_std <= 1e-8), fallback_scale, r_std)
            scaled = (values - r_mean) / denom
        else:
            scaled = (values - stats.center) / stats.scale

        clipped_count = 0
        if manifest.clip_enabled:
            clip_min, clip_max = resolve_feature_clip_bounds(
                stats.feature,
                float(manifest.clip_min),
                float(manifest.clip_max),
            )
            if manifest.clip_mode == "tanh":
                m = min(abs(clip_min), abs(clip_max), float(manifest.soft_clip_m))
                clipped_mask = (scaled < -m) | (scaled > m)
                clipped_count = int(np.count_nonzero(clipped_mask))
                scaled = m * np.tanh(scaled / m)
            else:
                clipped_mask = (scaled < clip_min) | (scaled > clip_max)
                clipped_count = int(np.count_nonzero(clipped_mask))
                scaled = np.clip(scaled, clip_min, clip_max)

        columns[stats.feature] = scaled
        clipped_by_feature[stats.feature] = clipped_count

    total_clipped_values = int(sum(clipped_by_feature.values()))
    total_state_values = rows * len(manifest.features)
    max_clipped_feature = None
    max_feature_clip_ratio = 0.0
    if rows:
        for feature, count in clipped_by_feature.items():
            ratio = count / rows
            if ratio > max_feature_clip_ratio:
                max_feature_clip_ratio = ratio
                max_clipped_feature = feature

    diagnostics = {
        "state_feature_count": len(manifest.features),
        "clip_enabled": manifest.clip_enabled,
        "total_clipped_values": total_clipped_values,
        "clipped_value_ratio": (
            total_clipped_values / total_state_values if total_state_values else 0.0
        ),
        "max_clipped_feature": max_clipped_feature or "",
        "max_feature_clip_ratio": max_feature_clip_ratio,
        "clipped_by_feature_json": json.dumps(
            clipped_by_feature, ensure_ascii=False, sort_keys=True
        ),
    }
    return pl.DataFrame(columns), diagnostics


def write_manifest(manifest: ScaleManifest, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "scaler_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def write_diagnostics(rows: list[dict[str, object]], output_root: Path) -> Path:
    diagnostics_path = output_root / "scale_diagnostics.csv"
    pl.DataFrame(rows).write_csv(diagnostics_path)
    return diagnostics_path


def scale_one_input(
    *,
    stage: str,
    contract: str,
    input_file: Path,
    output_file: Path,
    state_features: list[str],
    manifest: ScaleManifest,
    args,
) -> dict[str, object]:
    logger.info("Loading split-stage scale-save input: input_file=%s output_file=%s", input_file, output_file)
    df = pl.read_ipc(input_file)
    validate_no_nan(df, path=input_file, stage="input")
    validate_state_features_present(df, state_features, input_file)

    state_set = set(state_features)
    reward_features = [f for f in reward_features_for(df, args) if f not in state_set]
    df_reward = df.select(reward_features)

    passthrough_targets = set(args.passthrough_features or [])
    passthrough_features = [f for f in state_features if f in passthrough_targets]
    scaled_features = [f for f in state_features if f not in passthrough_targets]

    df_scaled = df.select(scaled_features)
    df_scaled, diagnostics = apply_robust_scaler(df_scaled, manifest)
    df_passthrough = df.select(passthrough_features)

    state_cols = {}
    for f in state_features:
        if f in df_scaled.columns:
            state_cols[f] = df_scaled.get_column(f)
        elif f in df_passthrough.columns:
            state_cols[f] = df_passthrough.get_column(f)

    df_state_aligned = pl.DataFrame(state_cols)
    out = pl.concat([df_reward, df_state_aligned], how="horizontal").with_columns(
        pl.lit(args.symbols).alias("symbol")
    )

    validate_no_nan(out, path=output_file, stage="output")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.write_ipc(output_file)
    csv_output_file = output_file.with_suffix(".csv")
    out.write_csv(csv_output_file)
    logger.info(
        "Wrote split-stage scale-save output: output_file=%s csv_output_file=%s rows=%d columns=%d",
        output_file,
        csv_output_file,
        out.height,
        len(out.columns),
    )
    return {
        "stage": stage,
        "contract": contract,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "rows": out.height,
        **diagnostics,
    }


def main(args) -> None:
    started_at = time.monotonic()
    root_path = Path(args.root_path)
    data_root = resolve_path(root_path, args.data_path)
    save_root = resolve_path(root_path, args.save_path)
    rl_feature_path, vae_feature_path = resolve_feature_paths(root_path, args)
    validate_clip_args(args)
    rl_features, vae_features, state_features = load_dual_stream_state_features(
        rl_feature_path, vae_feature_path
    )

    inputs = split_stage_inputs(data_root, args.symbols, args.target_freq)
    if not inputs:
        raise ValueError(f"no split-stage inputs found for symbol={args.symbols}")
    preflight_validate_split_inputs(inputs, state_features)
    train_inputs = [item for item in inputs if item[0] == "train"]
    if not train_inputs:
        raise ValueError(f"no train split-stage inputs found for symbol={args.symbols}")

    output_root = save_root / args.symbols / args.target_freq
    manifest = fit_robust_scaler(
        train_inputs=train_inputs,
        state_features=state_features,
        rl_feature_path=rl_feature_path,
        vae_feature_path=vae_feature_path,
        args=args,
    )
    manifest_path = write_manifest(manifest, output_root)

    logger.info(
        "Starting multi-contract scale-save: symbol=%s target_freq=%s inputs=%d data_root=%s save_root=%s rl_feature_path=%s vae_feature_path=%s manifest_path=%s",
        args.symbols,
        args.target_freq,
        len(inputs),
        data_root,
        save_root,
        rl_feature_path,
        vae_feature_path,
        manifest_path,
    )
    diagnostics_rows = []
    for stage, contract, input_file in inputs:
        output_file = output_root / stage / f"{contract}.feather"
        diagnostics_rows.append(scale_one_input(
            stage=stage,
            contract=contract,
            input_file=input_file,
            output_file=output_file,
            state_features=state_features,
            manifest=manifest,
            args=args,
        ))
    diagnostics_path = write_diagnostics(diagnostics_rows, output_root)
    shutil.copyfile(rl_feature_path, output_root / "rl_state_features.npy")
    shutil.copyfile(vae_feature_path, output_root / "vae_state_features.npy")
    logger.info(
        "Finished multi-contract scale-save: inputs=%d diagnostics_path=%s elapsed_seconds=%.2f",
        len(inputs),
        diagnostics_path,
        time.monotonic() - started_at,
    )


if __name__ == "__main__":
    configure_logging()
    args = parser.parse_args()
    main(args)
