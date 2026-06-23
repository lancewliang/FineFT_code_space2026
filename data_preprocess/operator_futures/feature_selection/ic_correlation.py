# utlize the correlation between the return of the mark price and feature.
# select the features with high correlation with markprice return and low correlation with existing features.
import numpy as np
import os
import argparse
import sys
import json
import logging
from pathlib import Path
import time

import polars as pl

sys.path.append(".")
from operator_futures.feature_selection.cor_util import select_feature


logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

# TODO add multi-labelling: the target is calculated as a list of window lengths
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
    default="PREPROCESS_DATASET/binance-futures/ALL_FEATURE/",
    help="the path of storing the data",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="PREPROCESS_DATASET/binance-futures/IC_RESULT/",
    help="the path of storing the data",
)
parser.add_argument(
    "--symbols", type=str, default="BTCUSDT", help="the name of the ticker"
)
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
#
parser.add_argument(
    "--ic_theshold",
    type=float,
    default=0.01,
    help="the date of start",
)
parser.add_argument(
    "--cor_theshold",
    type=float,
    default=0.7,
    help="the date of start",
)

parser.add_argument(
    "--windows_list",
    type=int,  # 指定每个元素应转换为浮点数
    nargs="*",  # '*' 表示接受零个或多个参数
    default=[
        1,
        6,
        12,
    ],  # 设置默认值为一个列表
    help="List of threshold values",
)
parser.add_argument(
    "--market_type",
    type=str,
    default="crypto_futures",
    choices=["crypto_futures", "commodity_futures"],
    help="the market type of the preprocessed data",
)
parser.add_argument(
    "--orderbook_depth",
    type=int,
    default=25,
    help="the available orderbook depth",
)


def calculate_cor(column, target):
    column = np.asarray(column, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = ~(np.isnan(column) | np.isnan(target))
    column = column[valid]
    target = target[valid]
    if column.size < 2 or target.size < 2 or np.std(column) == 0 or np.std(target) == 0:
        return np.nan
    return float(np.corrcoef(column, target)[0, 1])


def build_pandas_like_correlation_frame(df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    if not features:
        return pl.DataFrame({"feature": []})
    arrays = [df[feature].to_numpy() for feature in features]
    matrix = np.empty((len(features), len(features)), dtype=float)
    for row_index, left in enumerate(arrays):
        for col_index in range(row_index, len(features)):
            value = calculate_cor(left, arrays[col_index])
            matrix[row_index, col_index] = value
            matrix[col_index, row_index] = value
    return pl.DataFrame(matrix, schema=features).with_columns(
        pl.Series("feature", features)
    ).select(["feature", *features])


def remove_duplicates_preserve_order(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def calculate_target(df, reward_feature, window_length):
    # the length of the target = len(df)-1
    if isinstance(df, pl.DataFrame):
        target = df.select(
            (pl.col(reward_feature).shift(-window_length) - pl.col(reward_feature)).alias(
                reward_feature
            )
        )[reward_feature]
        target = target.slice(0, max(target.len() - window_length, 0))
        return target
    target = df[reward_feature].shift(-window_length) - df[reward_feature]
    target = target[:-window_length]
    return target


def select_reward_state_features(df, market_type="crypto_futures", orderbook_depth=25):
    if market_type == "commodity_futures":
        from operator_futures.commodity.schema import get_reward_execution_columns

        reward_features = [
            col for col in get_reward_execution_columns(orderbook_depth) if col in df.columns
        ]
    else:
        reward_features = list(df.columns[:106])
    state_feature = [col for col in df.columns if col not in reward_features]
    return reward_features, state_feature


def main(args):
    started_at = time.monotonic()
    args.data_path = os.path.join(args.root_path, args.data_path)
    args.save_path = os.path.join(args.root_path, args.save_path)
    input_path = Path(args.data_path) / args.symbols / args.target_freq / (
        "{}-{}.feather".format(args.start_date, args.end_date)
    )
    output_dir = Path(args.save_path) / args.symbols / args.target_freq / (
        "{}-{}".format(args.start_date, args.end_date)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Starting IC correlation process: symbol=%s start_date=%s end_date=%s target_freq=%s input=%s output_dir=%s ic_threshold=%s cor_threshold=%s windows=%s market_type=%s orderbook_depth=%d",
        args.symbols,
        args.start_date,
        args.end_date,
        args.target_freq,
        input_path,
        output_dir,
        args.ic_theshold,
        args.cor_theshold,
        args.windows_list,
        args.market_type,
        args.orderbook_depth,
    )
    df = pl.read_ipc(input_path)
    logger.info("Loaded IC input: rows=%d columns=%d", df.height, len(df.columns))
    reward_features, state_feature = select_reward_state_features(
        df, args.market_type, args.orderbook_depth
    )
    logger.info(
        "Selected IC feature groups: reward_features=%d candidate_state_features=%d",
        len(reward_features),
        len(state_feature),
    )

    ic_selection_key_all = []
    for window_length in args.windows_list:
        logger.info("Calculating IC window: window=%d", window_length)
        target = calculate_target(df, "mark_price", window_length)
        df_ic = df.slice(0, max(df.height - window_length, 0))
        ic_result = [
            calculate_cor(df_ic[feature].to_numpy(), target.to_numpy())
            for feature in state_feature
        ]
        sorted_pairs = sorted(
            zip(state_feature, ic_result), key=lambda x: abs(x[1]), reverse=True
        )

        cor = {feature: result for feature, result in sorted_pairs}
        with open(output_dir / "ic_window_{}.json".format(window_length), "w") as f:
            json.dump(cor, f)
        cor_abs = {}
        for key in cor.keys():
            cor_abs[key] = abs(cor[key])
        ic_selection_key = [
            key for key in cor_abs.keys() if cor_abs[key] > args.ic_theshold
        ]
        ic_selection_key_all.extend(ic_selection_key)
        logger.info(
            "Finished IC window: window=%d selected_features=%d output=%s",
            window_length,
            len(ic_selection_key),
            output_dir / "ic_window_{}.json".format(window_length),
        )
    # ic_selection_key = list(set(ic_selection_key_all))
    ic_selection_key=remove_duplicates_preserve_order(ic_selection_key_all)
    logger.info("Merged IC selected features: selected_features=%d", len(ic_selection_key))
    df_cor = build_pandas_like_correlation_frame(df, ic_selection_key)
    df_cor.write_csv(output_dir / "correlation.csv")
    selected_feature_names = select_feature(corre_df=df_cor, theshold=args.cor_theshold)
    state_feature = selected_feature_names
    out = df.select([*reward_features, *state_feature])
    logger.info(
        "Writing IC outputs: selected_state_features=%d total_columns=%d output_dir=%s",
        len(state_feature),
        len(out.columns),
        output_dir,
    )
    out.write_ipc(output_dir / "df.feather")
    np.save(output_dir / "state_features.npy", np.array(state_feature))
    logger.info(
        "Finished IC correlation process: rows=%d columns=%d elapsed_seconds=%.2f",
        out.height,
        len(out.columns),
        time.monotonic() - started_at,
    )
    return out


if __name__ == "__main__":
    configure_logging()
    args = parser.parse_args()
    main(args)
    logger.info("Done!")
