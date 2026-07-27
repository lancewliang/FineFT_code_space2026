import polars as pl
import numpy as np
import os
import re
from datetime import datetime
import argparse
import logging
import sys
import time

sys.path.append(".")
from operator_futures.util import match_strings_in_range, symbol_contract_path_parts
from memory_profiler import profile


logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

parser = argparse.ArgumentParser()
# data path
parser.add_argument(
    "--root_path",
    type=str,
    default=".",
    help="the path of storing the data",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/",
    help="the path of storing the data",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/MERGE_CONCAT",
    help="the path of storing the data",
)
parser.add_argument(
    "--symbols", type=str, default="BTCUSDT", help="the name of the ticker"
)
parser.add_argument("--contract", type=str, default=None, help="the contract of the ticker")
# date
parser.add_argument(
    "--date",
    type=str,
    default="2023-01-01",
    help="the path to save the data",
)
# freq
parser.add_argument(
    "--target_freq",
    type=str,
    default="10s",
    help="the date of start",
    choices=["10s", "1min", "5min","10min", "30min", "1H", "1D"],
)


def build_daily_feature_frames(
    snapshot: pl.DataFrame,
    der: pl.DataFrame,
    base_feature: pl.DataFrame,
    snapshot_feature: pl.DataFrame,
    quotes_feature: pl.DataFrame,
    kline_feature: pl.DataFrame,
    base_time_feature: pl.DataFrame | None = None,
    contract: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    der_without_symbol = der.drop("symbol") if "symbol" in der.columns else der
    reward_feature = (
        snapshot.join(der_without_symbol, on="timestamp", how="left")
        .join(snapshot_feature, on="timestamp", how="left")
    )
    if contract is not None:
        reward_feature = reward_feature.with_columns(pl.lit(contract).alias("contract"))
    future_feature = (
        base_feature.join(quotes_feature, on="timestamp", how="left")
        .join(kline_feature, on="timestamp", how="left")
    )
    if base_time_feature is not None:
        future_feature = future_feature.join(base_time_feature, on="timestamp", how="left")
    return reward_feature, future_feature


def main(args):
    started_at = time.monotonic()
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    symbol_parts = symbol_contract_path_parts(args.symbols, args.contract)
    logger.info(
        "Starting daily merge process: symbol=%s date=%s target_freq=%s data_path=%s save_path=%s",
        args.symbols,
        args.date,
        args.target_freq,
        args.data_path,
        args.save_path,
    )
    snapshot_path = os.path.join(
        args.data_path,
        "DOWNSCALE_ORDERBOOK_25",
        *symbol_parts,
        args.target_freq,
        args.date + ".feather",
    )
    der_path = os.path.join(
        args.data_path,
        "DOWNSCALE_DERTIC",
        *symbol_parts,
        args.target_freq,
        args.date + ".feather",
    )
    base_feature_path = os.path.join(
        args.data_path,
        "BASE_FEATURE",
        *symbol_parts,
        args.target_freq,
        args.date + ".feather",
    )
    snapshot_feature_path = os.path.join(
        args.data_path,
        "CROSS_SECTION",
        "SNAPSHOT_FEATURE",
        *symbol_parts,
        args.target_freq,
        args.date + ".feather",
    )
    quotes_feature_path = os.path.join(
        args.data_path,
        "CROSS_SECTION",
        "QUOTES_FEATURE",
        *symbol_parts,
        args.target_freq,
        args.date + ".feather",
    )
    kline_feature_path = os.path.join(
        args.data_path,
        "CROSS_SECTION",
        "KLINE_FEATURE",
        *symbol_parts,
        args.target_freq,
        args.date + ".feather",
    )
    logger.info(
        "Reading daily merge inputs: snapshot=%s der=%s base=%s snapshot_feature=%s quotes_feature=%s kline_feature=%s",
        snapshot_path,
        der_path,
        base_feature_path,
        snapshot_feature_path,
        quotes_feature_path,
        kline_feature_path,
    )
    snapshot = pl.read_ipc(snapshot_path)
    der = pl.read_ipc(der_path)
    base_feature = pl.read_ipc(base_feature_path)
    snapshot_feature = pl.read_ipc(snapshot_feature_path)
    quotes_feature = pl.read_ipc(quotes_feature_path)
    kline_feature = pl.read_ipc(kline_feature_path)

    base_time_feature_path = os.path.join(
        args.data_path,
        "BASE_TIME_FEATURE",
        *symbol_parts,
        args.target_freq,
        args.date + ".feather",
    )
    base_time_feature = (
        pl.read_ipc(base_time_feature_path)
        if os.path.exists(base_time_feature_path)
        else None
    )
    if base_time_feature is not None:
        if base_feature["timestamp"].to_list() != base_time_feature["timestamp"].to_list():
            raise ValueError(
                f"BASE_TIME_FEATURE timestamp set does not match BASE_FEATURE for date {args.date}: {base_time_feature_path}"
            )
    logger.info(
        "Loaded daily merge inputs: snapshot_rows=%d der_rows=%d base_rows=%d snapshot_feature_rows=%d quotes_feature_rows=%d kline_feature_rows=%d",
        snapshot.height,
        der.height,
        base_feature.height,
        snapshot_feature.height,
        quotes_feature.height,
        kline_feature.height,
    )

    logger.info("Building daily merged feature frames")
    reward_feature, base_feature = build_daily_feature_frames(
        snapshot,
        der,
        base_feature,
        snapshot_feature,
        quotes_feature,
        kline_feature,
        base_time_feature=base_time_feature,
        contract=args.contract,
    )

    # merged_feature = pd.concat(
    #     [
    #         snapshot,
    #         der,
    #         base_feature,
    #         snapshot_feature,
    #         quotes_feature,
    #         kline_feature,
    #     ],
    #     axis=1,
    # )
    # merged_feature.reset_index(inplace=True)

    if not os.path.exists(
        os.path.join(
            args.save_path,
            "MERGED_FEATURE",
            *symbol_parts,
            args.target_freq,
            "CONCURRENT_FEATURE",
        )
    ):
        os.makedirs(
            os.path.join(
                args.save_path,
                "MERGED_FEATURE",
                *symbol_parts,
                args.target_freq,
                "CONCURRENT_FEATURE",
            )
        )
    if not os.path.exists(
        os.path.join(
            args.save_path,
            "MERGED_FEATURE",
            *symbol_parts,
            args.target_freq,
            "FUTURE_FEATURE",
        )
    ):
        os.makedirs(
            os.path.join(
                args.save_path,
                "MERGED_FEATURE",
                *symbol_parts,
                args.target_freq,
                "FUTURE_FEATURE",
            )
        )
    reward_output_path = os.path.join(
        args.save_path,
        "MERGED_FEATURE",
        *symbol_parts,
        args.target_freq,
        "CONCURRENT_FEATURE",
        args.date + ".feather",
    )
    future_output_path = os.path.join(
        args.save_path,
        "MERGED_FEATURE",
        *symbol_parts,
        args.target_freq,
        "FUTURE_FEATURE",
        args.date + ".feather",
    )
    logger.info(
        "Writing daily merge outputs: concurrent=%s future=%s",
        reward_output_path,
        future_output_path,
    )
    reward_feature.write_ipc(reward_output_path)
    base_feature.write_ipc(future_output_path)
    logger.info(
        "Finished daily merge process: concurrent_rows=%d future_rows=%d elapsed_seconds=%.2f",
        reward_feature.height,
        base_feature.height,
        time.monotonic() - started_at,
    )


if __name__ == "__main__":
    configure_logging()
    args = parser.parse_args()
    main(args)
