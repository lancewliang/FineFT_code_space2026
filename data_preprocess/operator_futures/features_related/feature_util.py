import pandas as pd
import polars as pl


def calculate_quotes_df(agg_data: pd.DataFrame):
    feature = agg_data["bid_price"]
    return len(feature)


def calculate_bid_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed = agg_data["bid_price"] != agg_data["bid_price"].shift(1)

    # Combine the two checks with an OR operation
    bid_change = bid_price_changed

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_ask_quotes_df(agg_data: pd.DataFrame):
    ask_price_changed = agg_data["ask_price"] != agg_data["ask_price"].shift(1)

    # Combine the two checks with an OR operation
    ask_change = ask_price_changed

    # Count the number of True values
    change_count = ask_change.sum()
    return change_count


def calculate_bid_up_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed_up = agg_data["bid_price"] > agg_data["bid_price"].shift(1)

    # Combine the two checks with an OR operation
    bid_change = bid_price_changed_up

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_bid_down_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed_up = agg_data["bid_price"] < agg_data["bid_price"].shift(1)

    # Combine the two checks with an OR operation
    bid_change = bid_price_changed_up

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_ask_up_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed_up = agg_data["ask_price"] > agg_data["ask_price"].shift(1)

    # Combine the two checks with an OR operation
    bid_change = bid_price_changed_up

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_ask_down_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed_up = agg_data["ask_price"] < agg_data["ask_price"].shift(1)

    # Combine the two checks with an OR operation
    bid_change = bid_price_changed_up

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_bidsize_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed = agg_data["bid_amount"] != agg_data["bid_amount"].shift(1)

    # Combine the two checks with an OR operation
    bid_change = bid_price_changed

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_asksize_quotes_df(agg_data: pd.DataFrame):
    ask_price_changed = agg_data["ask_amount"] != agg_data["ask_amount"].shift(1)

    # Combine the two checks with an OR operation
    ask_change = ask_price_changed

    # Count the number of True values
    change_count = ask_change.sum()
    return change_count


def calculate_bidsize_up_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed_up = agg_data["bid_amount"] > agg_data["bid_amount"].shift(1)

    # Combine the two checks with an OR operation
    bid_change = bid_price_changed_up

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_bidsize_down_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed_up = agg_data["bid_amount"] < agg_data["bid_amount"].shift(1)

    # Combine the two checks with an OR operation
    bid_change = bid_price_changed_up

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_asksize_up_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed_up = agg_data["ask_amount"] > agg_data["ask_amount"].shift(1)

    # Combine the two checks with an OR operation
    bid_change = bid_price_changed_up

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_asksize_down_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed_up = agg_data["ask_amount"] < agg_data["ask_amount"].shift(1)

    # Combine the two checks with an OR operation
    bid_change = bid_price_changed_up

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_bid_askflat_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed = agg_data["bid_price"] != agg_data["bid_price"].shift(1)
    askflat = agg_data["ask_price"] == agg_data["ask_price"].shift(1)
    # Combine the two checks with an OR operation
    bid_change = askflat & bid_price_changed

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_bidup_askflat_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed = agg_data["bid_price"] > agg_data["bid_price"].shift(1)
    askflat = agg_data["ask_price"] == agg_data["ask_price"].shift(1)
    # Combine the two checks with an OR operation
    bid_change = askflat & bid_price_changed

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_biddown_askflat_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed = agg_data["bid_price"] < agg_data["bid_price"].shift(1)
    askflat = agg_data["ask_price"] == agg_data["ask_price"].shift(1)
    # Combine the two checks with an OR operation
    bid_change = askflat & bid_price_changed

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_ask_bidflat_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed = agg_data["ask_price"] != agg_data["ask_price"].shift(1)
    askflat = agg_data["bid_price"] == agg_data["bid_price"].shift(1)
    # Combine the two checks with an OR operation
    bid_change = askflat & bid_price_changed

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_askup_bidflat_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed = agg_data["ask_price"] > agg_data["ask_price"].shift(1)
    askflat = agg_data["bid_price"] == agg_data["bid_price"].shift(1)
    # Combine the two checks with an OR operation
    bid_change = askflat & bid_price_changed

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def calculate_askdown_bidflat_quotes_df(agg_data: pd.DataFrame):
    bid_price_changed = agg_data["ask_price"] < agg_data["ask_price"].shift(1)
    askflat = agg_data["bid_price"] == agg_data["bid_price"].shift(1)
    # Combine the two checks with an OR operation
    bid_change = askflat & bid_price_changed

    # Count the number of True values
    change_count = bid_change.sum()
    return change_count


def create_quotes_feature(quotes, target_freq):
    if isinstance(quotes, pl.DataFrame):
        prepared = quotes.sort("timestamp").with_columns(
            (pl.col("bid_price") != pl.col("bid_price").shift(1))
            .fill_null(False)
            .alias("_bid_change"),
            (pl.col("ask_price") != pl.col("ask_price").shift(1))
            .fill_null(False)
            .alias("_ask_change"),
            (pl.col("bid_price") > pl.col("bid_price").shift(1))
            .fill_null(False)
            .alias("_bid_up"),
            (pl.col("bid_price") < pl.col("bid_price").shift(1))
            .fill_null(False)
            .alias("_bid_down"),
            (pl.col("ask_price") > pl.col("ask_price").shift(1))
            .fill_null(False)
            .alias("_ask_up"),
            (pl.col("ask_price") < pl.col("ask_price").shift(1))
            .fill_null(False)
            .alias("_ask_down"),
            (pl.col("bid_amount") != pl.col("bid_amount").shift(1))
            .fill_null(False)
            .alias("_bidsize_change"),
            (pl.col("ask_amount") != pl.col("ask_amount").shift(1))
            .fill_null(False)
            .alias("_asksize_change"),
            (pl.col("bid_amount") > pl.col("bid_amount").shift(1))
            .fill_null(False)
            .alias("_bidsize_up"),
            (pl.col("bid_amount") < pl.col("bid_amount").shift(1))
            .fill_null(False)
            .alias("_bidsize_down"),
            (pl.col("ask_amount") > pl.col("ask_amount").shift(1))
            .fill_null(False)
            .alias("_asksize_up"),
            (pl.col("ask_amount") < pl.col("ask_amount").shift(1))
            .fill_null(False)
            .alias("_asksize_down"),
            (pl.col("ask_price") == pl.col("ask_price").shift(1))
            .fill_null(False)
            .alias("_ask_flat"),
            (pl.col("bid_price") == pl.col("bid_price").shift(1))
            .fill_null(False)
            .alias("_bid_flat"),
        )
        return (
            prepared.group_by_dynamic(
                "timestamp", every=target_freq, closed="left", label="left"
            )
            .agg(
                pl.len().alias("nquote"),
                pl.col("_bid_change").sum().alias("nquote_bid"),
                pl.col("_ask_change").sum().alias("nquote_ask"),
                pl.col("_bid_up").sum().alias("nquote_bid_up"),
                pl.col("_bid_down").sum().alias("nquote_bid_down"),
                pl.col("_ask_up").sum().alias("nquote_ask_up"),
                pl.col("_ask_down").sum().alias("nquote_ask_down"),
                pl.col("_bidsize_change").sum().alias("nquote_bidsize"),
                pl.col("_asksize_change").sum().alias("nquote_asksize"),
                pl.col("_bidsize_up").sum().alias("nquote_bidsize_up"),
                pl.col("_bidsize_down").sum().alias("nquote_bidsize_down"),
                pl.col("_asksize_up").sum().alias("nquote_asksize_up"),
                pl.col("_asksize_down").sum().alias("nquote_asksize_down"),
                (pl.col("_ask_flat") & pl.col("_bid_change"))
                .sum()
                .alias("nquote_bid_askflat"),
                (pl.col("_ask_flat") & pl.col("_bid_up"))
                .sum()
                .alias("nquote_bidup_askflat"),
                (pl.col("_ask_flat") & pl.col("_bid_down"))
                .sum()
                .alias("nquote_biddown_askflat"),
                (pl.col("_bid_flat") & pl.col("_ask_change"))
                .sum()
                .alias("nquote_ask_bidflat"),
                (pl.col("_bid_flat") & pl.col("_ask_up"))
                .sum()
                .alias("nquote_askup_bidflat"),
                (pl.col("_bid_flat") & pl.col("_ask_down"))
                .sum()
                .alias("nquote_askdown_bidflat"),
            )
            .sort("timestamp")
        )

    function_name_list = [
        "nquote",
        "nquote_bid",
        "nquote_ask",
        "nquote_bid_up",
        "nquote_bid_down",
        "nquote_ask_up",
        "nquote_ask_down",
        "nquote_bidsize",
        "nquote_asksize",
        "nquote_bidsize_up",
        "nquote_bidsize_down",
        "nquote_asksize_up",
        "nquote_asksize_down",
        "nquote_bid_askflat",
        "nquote_bidup_askflat",
        "nquote_biddown_askflat",
        "nquote_ask_bidflat",
        "nquote_askup_bidflat",
        "nquote_askdown_bidflat",
    ]
    quotes_indicators = []
    for fc, name in zip(
        [
            calculate_quotes_df,
            calculate_bid_quotes_df,
            calculate_ask_quotes_df,
            calculate_bid_up_quotes_df,
            calculate_bid_down_quotes_df,
            calculate_ask_up_quotes_df,
            calculate_ask_down_quotes_df,
            calculate_bidsize_quotes_df,
            calculate_asksize_quotes_df,
            calculate_bidsize_up_quotes_df,
            calculate_bidsize_down_quotes_df,
            calculate_asksize_up_quotes_df,
            calculate_asksize_down_quotes_df,
            calculate_bid_askflat_quotes_df,
            calculate_bidup_askflat_quotes_df,
            calculate_biddown_askflat_quotes_df,
            calculate_ask_bidflat_quotes_df,
            calculate_askup_bidflat_quotes_df,
            calculate_askdown_bidflat_quotes_df,
        ],
        function_name_list,
    ):
        indicator = quotes.resample(target_freq).apply(fc)
        indicator.name = name
        quotes_indicators.append(indicator)
        quotes_df = pd.concat(quotes_indicators, axis=1, names=function_name_list)
    return quotes_df


def create_ohlc_quotes_feature(quotes, target_freq):
    if isinstance(quotes, pl.DataFrame):
        indicators = quotes.with_columns(
            (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
            ((pl.col("ask_price") + pl.col("bid_price")) / 2).alias("mid"),
            (
                (pl.col("bid_amount") - pl.col("ask_amount"))
                / (pl.col("bid_amount") + pl.col("ask_amount"))
            ).alias("imblance_volume"),
            (
                (
                    pl.col("ask_amount") * pl.col("ask_price")
                    + pl.col("bid_amount") * pl.col("bid_price")
                )
                / (pl.col("ask_amount") + pl.col("bid_amount"))
            ).alias("makav_rev"),
            (
                (
                    pl.col("ask_amount") * pl.col("bid_price")
                    + pl.col("bid_amount") * pl.col("ask_price")
                )
                / (pl.col("ask_amount") + pl.col("bid_amount"))
            ).alias("makav_ori"),
            pl.col("bid_price").alias("bid"),
            pl.col("bid_amount").alias("bidsize"),
            pl.col("ask_price").alias("ask"),
            pl.col("ask_amount").alias("asksize"),
        )
        indicator_names = [
            "spread",
            "mid",
            "imblance_volume",
            "makav_rev",
            "makav_ori",
            "bid",
            "bidsize",
            "ask",
            "asksize",
        ]
        aggregations = []
        for indicator_name in indicator_names:
            aggregations.extend(
                [
                    pl.col(indicator_name).first().alias(f"open_{indicator_name}"),
                    pl.col(indicator_name).max().alias(f"high_{indicator_name}"),
                    pl.col(indicator_name).min().alias(f"low_{indicator_name}"),
                    pl.col(indicator_name).last().alias(f"close_{indicator_name}"),
                    pl.col(indicator_name).mean().alias(f"twap_{indicator_name}"),
                    pl.col(indicator_name).mean().alias(f"awap_{indicator_name}"),
                ]
            )
        return (
            indicators.sort("timestamp")
            .group_by_dynamic("timestamp", every=target_freq, closed="left", label="left")
            .agg(aggregations)
            .fill_null(strategy="forward")
        )

    book_price_indicators = {
        "spread": quotes["ask_price"] - quotes["bid_price"],
        "mid": (quotes["ask_price"] + quotes["bid_price"]) / 2,
        "imblance_volume": (quotes["bid_amount"] - quotes["ask_amount"])
        / (quotes["bid_amount"] + quotes["ask_amount"]),
        "makav_rev": (
            quotes["ask_amount"] * quotes["ask_price"]
            + quotes["bid_amount"] * quotes["bid_price"]
        )
        / (quotes["ask_amount"] + quotes["bid_amount"]),
        "makav_ori": (
            quotes["ask_amount"] * quotes["bid_price"]
            + quotes["bid_amount"] * quotes["ask_price"]
        )
        / (quotes["ask_amount"] + quotes["bid_amount"]),
        "bid": quotes["bid_price"],
        "bidsize": quotes["bid_amount"],
        "ask": quotes["ask_price"],
        "asksize": quotes["ask_amount"],
    }
    quotes_df_ = pd.DataFrame(index=quotes.resample(target_freq).asfreq().index)

    for indicator_name, indicator in book_price_indicators.items():
        resample_OHLC, time_weighted_indicator, average_weighted_indicator = (
            calculate_agg(quotes, indicator, target_freq)
        )

        # 添加 OHLC 特征到结果 DataFrame
        for ohlc_name in ["open", "high", "low", "close"]:
            col_name = "{}_{}".format(ohlc_name, indicator_name)
            quotes_df_[col_name] = resample_OHLC[ohlc_name]

        # 添加 time-weighted 和 volume-weighted 特征到结果 DataFrame
        quotes_df_["twap_{}".format(indicator_name)] = time_weighted_indicator
        quotes_df_["awap_{}".format(indicator_name)] = average_weighted_indicator

    quotes_df_.ffill(inplace=True)
    return quotes_df_


def calculate_agg(df: pd.DataFrame, indicator: pd.DataFrame, AGG_FREQUENCY):
    resample_OHLC = indicator.resample(AGG_FREQUENCY).ohlc()
    time_diff = df.index.to_series().diff().dt.total_seconds()
    timemutiplyer = indicator * time_diff
    time_weighted_indicator = (
        timemutiplyer.resample(AGG_FREQUENCY).sum()
        / time_diff.resample(AGG_FREQUENCY).sum()
    )
    average_weighted_indicator = indicator.resample(AGG_FREQUENCY).mean()
    return resample_OHLC, time_weighted_indicator, average_weighted_indicator


def count_up(x):
    return (x > 0).sum()


def count_down(x):
    return (x < 0).sum()


def count_flat(x):
    return (x == 0).sum()


def time_diff_weighted(x):
    time_diff_weights = x.index.to_series().diff().dt.total_seconds()
    return (x * time_diff_weights).sum() / time_diff_weights.sum()


def intial_process_trades(trades: pd.DataFrame, target_freq):
    if isinstance(trades, pl.DataFrame):
        trades = trades.sort("timestamp").with_columns(
            (pl.col("amount") * pl.col("price")).alias("tradeval"),
            pl.col("price").diff().alias("price_diff"),
        )
        trades_df = (
            trades.group_by_dynamic(
                "timestamp", every=target_freq, closed="left", label="left"
            )
            .agg(
                pl.col("price").first().alias("open"),
                pl.col("price").max().alias("high"),
                pl.col("price").min().alias("low"),
                pl.col("price").last().alias("close"),
                pl.col("amount").sum().alias("volume"),
                pl.col("tradeval").sum().alias("tradeval"),
                pl.col("id").count().alias("ntrade"),
                pl.col("price").mean().alias("awap"),
                (pl.col("price_diff") > 0).sum().alias("ntrade_up"),
                (pl.col("price_diff") < 0).sum().alias("ntrade_down"),
                (pl.col("price_diff") == 0).sum().alias("ntrade_flat"),
            )
            .with_columns(
                (pl.col("tradeval") / pl.col("volume")).alias("vwap"),
                pl.col("awap").alias("twap"),
            )
            .select(
                [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "tradeval",
                    "ntrade",
                    "vwap",
                    "awap",
                    "twap",
                    "ntrade_up",
                    "ntrade_down",
                    "ntrade_flat",
                ]
            )
        )
        return trades_df, trades

    trades_df = trades["price"].resample(target_freq).ohlc()
    trades_df["volume"] = trades["amount"].resample(target_freq).sum()
    trades["tradeval"] = trades["amount"] * trades["price"]
    trades_df["tradeval"] = trades["tradeval"].resample(target_freq).sum()
    trades_df["ntrade"] = trades["id"].resample(target_freq).count()
    trades_df["vwap"] = trades_df["tradeval"] / trades_df["volume"]
    trades_df["awap"] = trades["price"].resample(target_freq).mean()
    time_diff = trades.index.to_series().diff().dt.total_seconds()
    trades_df["twap"] = (trades["price"] * time_diff).resample(
        target_freq
    ).sum() / time_diff.resample(target_freq).sum()
    trades["price_diff"] = trades["price"].diff()
    trades_df["ntrade_up"] = trades["price_diff"].resample(target_freq).apply(count_up)
    trades_df["ntrade_down"] = (
        trades["price_diff"].resample(target_freq).apply(count_down)
    )
    trades_df["ntrade_flat"] = (
        trades["price_diff"].resample(target_freq).apply(count_flat)
    )
    return trades_df, trades

    # Define functions to count the up, down, and flat trades


def side_group_trades(trades: pd.DataFrame, target_freq: str):
    if isinstance(trades, pl.DataFrame):
        side_frames = []
        for side in trades.select("side").unique().to_series().to_list():
            side_df = (
                trades.filter(pl.col("side") == side)
                .sort("timestamp")
                .group_by_dynamic(
                    "timestamp", every=target_freq, closed="left", label="left"
                )
                .agg(
                    pl.col("price").first().alias(f"open_{side}"),
                    pl.col("price").max().alias(f"high_{side}"),
                    pl.col("price").min().alias(f"low_{side}"),
                    pl.col("price").last().alias(f"close_{side}"),
                    pl.col("amount").sum().alias(f"volume_{side}"),
                    pl.col("amount").sum().alias(f"{side}_volume"),
                    pl.col("tradeval").sum().alias(f"tradeval_{side}"),
                    pl.col("id").count().alias(f"ntrade_{side}"),
                    (pl.col("price_diff") > 0).sum().alias(f"ntrade_up_{side}"),
                    (pl.col("price_diff") < 0).sum().alias(f"ntrade_down_{side}"),
                    (pl.col("price_diff") == 0).sum().alias(f"ntrade_flat_{side}"),
                    pl.col("price").mean().alias(f"awap_{side}"),
                    pl.col("price").mean().alias(f"twap_{side}"),
                )
                .with_columns(
                    (pl.col(f"tradeval_{side}") / pl.col(f"volume_{side}")).alias(
                        f"vwap_{side}"
                    )
                )
            )
            side_frames.append(side_df)

        if not side_frames:
            return pl.DataFrame({"timestamp": []})

        result = side_frames[0]
        for side_df in side_frames[1:]:
            result = result.join(side_df, on="timestamp", how="full", coalesce=True)
        return result.sort("timestamp")

    buy_sell_dfs = []
    buy_sell_dfs.append(
        trades.groupby(["side"])["price"]
        .resample(target_freq)
        .ohlc()
        .reset_index()
        .pivot(
            index="timestamp", columns="side", values=["open", "high", "low", "close"]
        )
    )
    buy_sell_dfs.append(
        trades.groupby(["side"])["amount"]
        .resample(target_freq)
        .sum()
        .reset_index()
        .rename(columns={"amount": "volume"})
        .pivot(index="timestamp", columns="side", values=["volume"])
    )
    buy_sell_dfs.append(
        trades.groupby(["side"])["tradeval"]
        .resample(target_freq)
        .sum()
        .reset_index()
        .pivot(index="timestamp", columns="side", values=["tradeval"])
    )
    buy_sell_dfs.append(
        trades.groupby(["side"])["id"]
        .resample(target_freq)
        .count()
        .reset_index()
        .rename(columns={"id": "ntrade"})
        .pivot(index="timestamp", columns="side", values=["ntrade"])
    )
    buy_sell_dfs.append(
        trades.groupby(["side"])["price_diff"]
        .resample(target_freq)
        .apply(count_up)
        .reset_index()
        .rename(columns={"price_diff": "ntrade_up"})
        .pivot(index="timestamp", columns="side", values=["ntrade_up"])
    )
    buy_sell_dfs.append(
        trades.groupby(["side"])["price_diff"]
        .resample(target_freq)
        .apply(count_down)
        .reset_index()
        .rename(columns={"price_diff": "ntrade_down"})
        .pivot(index="timestamp", columns="side", values=["ntrade_down"])
    )
    buy_sell_dfs.append(
        trades.groupby(["side"])["price_diff"]
        .resample(target_freq)
        .apply(count_flat)
        .reset_index()
        .rename(columns={"price_diff": "ntrade_flat"})
        .pivot(index="timestamp", columns="side", values=["ntrade_flat"])
    )
    buy_sell_dfs.append(
        trades.groupby(["side"])["price"]
        .resample(target_freq)
        .mean()
        .reset_index()
        .rename(columns={"price": "awap"})
        .pivot(index="timestamp", columns="side", values=["awap"])
    )
    buy_sell_dfs.append(
        trades.groupby(["side", pd.Grouper(freq=target_freq)])["price"]
        .apply(time_diff_weighted)
        .reset_index()
        .rename(columns={"price": "twap"})
        .pivot(index="timestamp", columns="side", values=["twap"])
    )
    buy_sell_df = pd.concat(buy_sell_dfs, axis=1)
    buy_sell_df.columns = ["_".join(col).strip() for col in buy_sell_df.columns]
    buy_sell_df["vwap_buy"] = buy_sell_df["tradeval_buy"] / buy_sell_df["volume_buy"]
    buy_sell_df["vwap_sell"] = buy_sell_df["tradeval_sell"] / buy_sell_df["volume_sell"]
    return buy_sell_df


def move_column_in_position(df: pd.DataFrame, column, position):
    if isinstance(df, pl.DataFrame):
        cols = df.columns
        cols.insert(position, cols.pop(cols.index(column)))
        return df.select(cols)

    cols = df.columns.tolist()
    cols.insert(position, cols.pop(cols.index(column)))
    return df[cols]


def preprocess_trades(df: pd.DataFrame):
    if isinstance(df, pl.DataFrame):
        drop_columns = [column for column in ["local_timestamp"] if column in df.columns]
        return (
            df.drop(drop_columns)
            .with_columns(pl.from_epoch("timestamp", time_unit="us").alias("timestamp"))
            .sort("timestamp")
            .with_row_index("id")
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us")
    df.drop(columns=["local_timestamp"], inplace=True)
    df["id"] = range(len(df))
    df.set_index("timestamp", inplace=True)
    return df


def preprocess_quotes(df: pd.DataFrame):
    if isinstance(df, pl.DataFrame):
        sort_column = "local_timestamp" if "local_timestamp" in df.columns else "timestamp"
        drop_columns = [column for column in ["local_timestamp"] if column in df.columns]
        return (
            df.sort(sort_column)
            .with_columns(pl.from_epoch("timestamp", time_unit="us").alias("timestamp"))
            .drop(drop_columns)
            .unique(subset=["timestamp"], keep="first", maintain_order=True)
            .sort("timestamp")
        )

    df.sort_values("local_timestamp", inplace=True)
    df['timestamp'] += df.groupby('timestamp').cumcount()

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us")
    df.drop(columns=["local_timestamp"], inplace=True)
    df.set_index("timestamp", inplace=True)
    # 发生的并不常见
    if df.index.duplicated().any():
        # 处理重复的索引
        df = df[~df.index.duplicated(keep="first")]
        print("Alert, there are multi snapshot corresponding to the same timestamp")
    return df


if __name__ == "__main__":
    trades = pd.read_csv(
        "./DOWNLOAD_DATASET/binance-futures/BTCUSDT/trades/binance-futures_trades_2023-01-01_BTCUSDT.csv"
    )
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], unit="us")
    trades.set_index("timestamp", inplace=True)
    target_freq = "1min"
    print(side_group_trades(trades, target_freq))
