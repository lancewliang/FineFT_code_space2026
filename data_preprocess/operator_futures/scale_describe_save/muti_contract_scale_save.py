import argparse
import logging
import os
from pathlib import Path
import time

import numpy as np
import polars as pl

from operator_futures.scale_describe_save.scale_save import (
    configure_logging,
    scale_mean,
    scale_std,
    validate_no_nan,
)


logger = logging.getLogger(__name__)
SPLIT_STAGES = ("train", "valid", "test")


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
    "--feature_list_path",
    type=str,
    default=None,
    help="train-stage state_features.npy selected by feature selection",
)


def resolve_path(root_path: Path, path: str | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return root_path / resolved


def default_feature_list_path(root_path: Path, args) -> Path:
    return (
        root_path
        / "PREPROCESS_DATASET/commodity-futures/FEATURE_SELECTION"
        / args.target_freq
        / args.symbols
        / "train"
        / "state_features.npy"
    )


def load_state_features(feature_list_path: Path) -> list[str]:
    state_features = np.load(feature_list_path, allow_pickle=True).tolist()
    if not state_features:
        raise ValueError(f"state feature list is empty: {feature_list_path}")
    return state_features


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
    if args.market_type == "commodity_futures":
        from operator_futures.commodity.schema import get_reward_execution_columns

        return [col for col in get_reward_execution_columns(args.orderbook_depth) if col in df.columns]
    return list(df.columns[:106])


def scale_one_input(
    *,
    input_file: Path,
    output_file: Path,
    state_features: list[str],
    args,
) -> None:
    logger.info("Loading split-stage scale-save input: input_file=%s output_file=%s", input_file, output_file)
    df = pl.read_ipc(input_file)
    validate_no_nan(df, path=input_file, stage="input")

    missing_features = [feature for feature in state_features if feature not in df.columns]
    if missing_features:
        raise ValueError(
            f"missing selected state feature columns in {input_file}: {missing_features}"
        )

    reward_features = reward_features_for(df, args)
    df_reward = df.select(reward_features)
    df_state = df.select(state_features)
    df_state = scale_std(df_state, args.base)
    df_state = scale_mean(df_state, args.base, args.clip_theshold)
    out = pl.concat([df_reward, df_state], how="horizontal").with_columns(
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


def main(args) -> None:
    started_at = time.monotonic()
    root_path = Path(args.root_path)
    data_root = resolve_path(root_path, args.data_path)
    save_root = resolve_path(root_path, args.save_path)
    feature_list_path = resolve_path(root_path, args.feature_list_path) or default_feature_list_path(
        root_path, args
    )
    state_features = load_state_features(feature_list_path)

    inputs = split_stage_inputs(data_root, args.symbols, args.target_freq)
    if not inputs:
        raise ValueError(f"no split-stage inputs found for symbol={args.symbols}")

    logger.info(
        "Starting multi-contract scale-save: symbol=%s target_freq=%s inputs=%d data_root=%s save_root=%s feature_list_path=%s",
        args.symbols,
        args.target_freq,
        len(inputs),
        data_root,
        save_root,
        feature_list_path,
    )
    for stage, contract, input_file in inputs:
        output_file = save_root / args.symbols / args.target_freq / stage / f"{contract}.feather"
        scale_one_input(
            input_file=input_file,
            output_file=output_file,
            state_features=state_features,
            args=args,
        )
    logger.info(
        "Finished multi-contract scale-save: inputs=%d elapsed_seconds=%.2f",
        len(inputs),
        time.monotonic() - started_at,
    )


if __name__ == "__main__":
    configure_logging()
    args = parser.parse_args()
    main(args)
