import os
import argparse
import logging
import sys
import time

sys.path.append(".")
from operator_futures.util import (
    find_ohlcv_groups,
    find_ohlc_groups,
    symbol_contract_path_parts,
)
from operator_futures.data_quality import DataQualityValidator
import polars as pl
from operator_futures.commodity.config import COMMODITY_CONFIGS
from operator_futures.time_operator.multi_processing_util import (
    get_multi_window_ohlcv,
    get_multi_window_ohlc,
    get_multi_feature_window_price,
    get_risk_and_liquidity_state_features,
    _inner_join_on_timestamp,
)
from operator_futures.time_operator.time_operator_util import process_enhanced_state_features


logger = logging.getLogger(__name__)


def _csv_path(feather_path: str) -> str:
    return os.path.splitext(feather_path)[0] + ".csv"


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
    default="PREPROCESS_DATASET/binance-futures/MERGE_CONCAT/CONCAT_FEATURE/",
    help="the path of storing the data",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/TIME_FEATURE/",
    help="the path of storing the data",
)
parser.add_argument(
    "--symbols", type=str, default="BTCUSDT", help="the name of the ticker"
)
parser.add_argument("--contract", type=str, default=None, help="the contract of the ticker")
# date
parser.add_argument(
    "--start_date",
    type=str,
    default="2023-01-01",
    help="the path to save the data",
)
parser.add_argument(
    "--end_date",
    type=str,
    default="2023-02-01",
    help="the path to save the data",
)

# freq
parser.add_argument(
    "--target_freq",
    type=str,
    default="10s",
    help="the date of start",
    choices=["10s", "1min", "5min", "10min", "30min", "1H", "1D"],
)
parser.add_argument(
    "--windows",
    type=str,
    default="2,6,12,16,24,48,96,192",
    help="List of windows sizes as comma-separated values",
)
parser.add_argument(
    "--orderbook_depth",
    type=int,
    default=25,
    help="the available orderbook depth",
)


def main(args):
    started_at = time.monotonic()
    time_feature_list_all = []
    windows = list(map(int, args.windows.split(",")))
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    symbol_parts = symbol_contract_path_parts(args.symbols, args.contract)
    logger.info(
        "Starting time feature process: symbol=%s start_date=%s end_date=%s target_freq=%s windows=%s data_path=%s save_path=%s orderbook_depth=%d",
        args.symbols,
        args.start_date,
        args.end_date,
        args.target_freq,
        windows,
        args.data_path,
        args.save_path,
        args.orderbook_depth,
    )
    input_path = os.path.join(
        args.data_path,
        *symbol_parts,
        args.target_freq,
        args.start_date + "-" + args.end_date + ".feather",
    )
    logger.info("Reading time feature input: input=%s", input_path)
    original_df = pl.read_ipc(input_path)
    logger.info(
        "Loaded time feature input: rows=%d columns=%d",
        original_df.height,
        len(original_df.columns),
    )
    DataQualityValidator.validate_no_illegal_values(
        original_df,
        stage="time_feature_input",
        contract=args.contract or args.symbols,
        trading_day=args.start_date + "-" + args.end_date,
    )
    ohlcv_features, _ = find_ohlcv_groups(original_df.columns)
    ohlc_features, _ = find_ohlc_groups(original_df.columns)
    price_features = [
        *[f"bid{l+1}_price" for l in range(args.orderbook_depth)],
        *[f"ask{l+1}_price" for l in range(args.orderbook_depth)],
        "buy_spread_oe_max",
        "sell_spread_oe_max",
        "wap_1",
        "wap_2",
        "buy_wap",
        "sell_wap",
        "mark_price",
        "buy_volume_oe",
        "sell_volume_oe",
        "imblance_volume_oe",
        *[f"ask{l+1}_size_n" for l in range(args.orderbook_depth)],
        *[f"bid{l+1}_size_n" for l in range(args.orderbook_depth)],
    ]
    price_features = [
        feature for feature in price_features if feature in original_df.columns
    ]
    logger.info(
        "Detected time feature groups: price_features=%d ohlcv_groups=%d ohlc_groups=%d",
        len(price_features),
        len(ohlcv_features),
        len(ohlc_features),
    )
    df_time = get_multi_feature_window_price(original_df, windows, price_features)
    # DataQualityValidator.validate_no_illegal_values(
    #         df_time,
    #         stage="df_time",
    #         contract=args.contract or args.symbols,
    #         trading_day=args.start_date + "-" + args.end_date,
    #     )
    time_feature_list_all.append(df_time)
    enhanced_df = process_enhanced_state_features(original_df)
    if enhanced_df.width > 1:
        time_feature_list_all.append(enhanced_df)
    if "open_interest" in original_df.columns:
        risk_liq_df = get_risk_and_liquidity_state_features(
            original_df,
            windows,
            symbol=args.symbols,
            target_freq=args.target_freq,
        )
        time_feature_list_all.append(risk_liq_df)
    for key in ohlcv_features:
        ohlc_features.pop(key, None)

    for ffuixes in ohlcv_features:
        (prefix, suffix) = ffuixes
        # print("prefix",prefix,"suffix",suffix)
        after_name = prefix + suffix
        converted_strings = "_origin" if after_name == "" else after_name
        feature_names = ohlcv_features[ffuixes]
        df_ohlcv = original_df.select(["timestamp", *feature_names]).rename(
            {
                prefix + key + suffix: key
                for key in ["open", "high", "low", "close", "volume"]
            },
        )
        p_process_ohlcv = get_multi_window_ohlcv(df_ohlcv, windows)
        p_process_ohlcv = p_process_ohlcv.rename(
            {
                key: key + converted_strings
                for key in p_process_ohlcv.columns
                if key != "timestamp"
            }
        )
        time_feature_list_all.append(p_process_ohlcv)
        # DataQualityValidator.validate_no_illegal_values(
        #     p_process_ohlcv,
        #     stage="p_process_ohlcv",
        #     contract=args.contract or args.symbols,
        #     trading_day=args.start_date + "-" + args.end_date,
        # )
    for ffuixes in ohlc_features:
        (prefix, suffix) = ffuixes
        # print("prefix",prefix,"suffix",suffix)
        after_name = prefix + suffix
        converted_strings = "_origin" if after_name == "" else after_name
        logger.info("Processing OHLC feature group: suffix=%s features=%d", converted_strings, len(ohlc_features[ffuixes]))
        feature_names = ohlc_features[ffuixes]
        df_ohlc = original_df.select(["timestamp", *feature_names]).rename(
            {
                prefix + key + suffix: key
                for key in [
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            }
        )
        p_process_ohlc = get_multi_window_ohlc(df_ohlc, windows)
        p_process_ohlc = p_process_ohlc.rename(
            {
                key: key + converted_strings
                for key in p_process_ohlc.columns
                if key != "timestamp"
            }
        )
        # DataQualityValidator.validate_no_illegal_values(
        #     p_process_ohlc,
        #     stage="p_process_ohlc",
        #     contract=args.contract or args.symbols,
        #     trading_day=args.start_date + "-" + args.end_date,
        # )
        time_feature_list_all.append(p_process_ohlc)

    time_df = _inner_join_on_timestamp(time_feature_list_all)
    DataQualityValidator.validate_no_illegal_values(
        time_df,
        stage="time_feature_output",
        contract=args.contract or args.symbols,
        trading_day=args.start_date + "-" + args.end_date,
    )
    if not os.path.exists(os.path.join(args.save_path, *symbol_parts, args.target_freq)):
        os.makedirs(os.path.join(args.save_path, *symbol_parts, args.target_freq))
    output_path = os.path.join(
        args.save_path,
        *symbol_parts,
        args.target_freq,
        args.start_date + "-" + args.end_date + ".feather",
    )
    logger.info("Writing time feature output: output=%s rows=%d columns=%d", output_path, time_df.height, len(time_df.columns))
    time_df.write_ipc(output_path)
    time_df.write_csv(_csv_path(output_path))
    logger.info(
        "Finished time feature process: rows=%d columns=%d elapsed_seconds=%.2f",
        time_df.height,
        len(time_df.columns),
        time.monotonic() - started_at,
    )


if __name__ == "__main__":
    configure_logging()
    args = parser.parse_args()
    main(args)
    logger.info("Done!")
