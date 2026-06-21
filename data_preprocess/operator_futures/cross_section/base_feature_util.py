import pandas as pd
import numpy as np
import polars as pl
import re

minium = 1e-15


def normalize_feature_cross_section(df: pd.DataFrame, features: list, method: str):
    pd_new = pd.DataFrame(index=df.index)
    assert method in ["buy_sell", "up_down_flat", "bid_ask"]
    if method == "buy_sell":
        assert len(features) == 3
        array = df[features].values
        all, buy, sell = array[:, 0], array[:, 1], array[:, 2]
        # print(all.shape, buy.shape, sell.shape)
        # if np.any(all == 0):
        #     index = np.where(all == 0)
        #     print(index)
        #     print("buy index", buy[index])
        #     print("sell index", sell[index])
        #     print("all index", all[index])
        buy_norm = buy / (all + minium)
        sell_norm = sell / (all + minium)
        imbalance = (buy - sell) / (all + minium)
        pd_new["{}_buy_bsnorm".format(features[0])] = buy_norm
        pd_new["{}_sell_bsnorm".format(features[0])] = sell_norm
        pd_new["{}_buysell_imbalance_bsnorm".format(features[0])] = imbalance

    elif method == "up_down_flat":
        assert len(features) in [3, 4]
        if len(features) == 3:
            array = df[features].values
            all, up, down = array[:, 0], array[:, 1], array[:, 2]

            up_norm = up / (all + minium)
            down_norm = down / (all + minium)
            imbalance_norm = (up - down) / (all + minium)
            pd_new["{}_up_udnorm".format(features[0])] = up_norm
            pd_new["{}_down_udnorm".format(features[0])] = down_norm
            pd_new["{}_updown_imbalance_udnorm".format(features[0])] = imbalance_norm
        else:
            array = df[features].values
            all, up, down, flat = array[:, 0], array[:, 1], array[:, 2], array[:, 3]

            up_norm = up / (all + minium)
            down_norm = down / (all + minium)
            flat_norm = flat / (all + minium)
            imbalance_norm = (up - down) / (all + minium)
            vol_norm = (up + down - flat) / (all + minium)
            pd_new["{}_up_udnorm".format(features[0])] = up_norm
            pd_new["{}_down_udnorm".format(features[0])] = down_norm
            pd_new["{}_flat_udnorm".format(features[0])] = flat_norm
            pd_new["{}_updown_imbalance_udnorm".format(features[0])] = imbalance_norm
            pd_new["{}_updownflat_vol_udnorm".format(features[0])] = vol_norm
    elif method == "bid_ask":
        assert len(features) in [2, 3]
        array = df[features].values
        if len(features) == 2:
            bid, ask = array[:, 0], array[:, 1]
            bid_norm = bid / (bid + ask + minium)
            ask_norm = ask / (bid + ask + minium)
            imbalance_norm = (bid - ask) / (bid + ask + minium)
            pd_new["{}_abnorm".format(features[0])] = bid_norm
            pd_new["{}_abnorm".format(features[1])] = ask_norm
            pd_new["{}_bid_imbalance_abnorm".format(features[1])] = imbalance_norm
        else:
            all, bid, ask = array[:, 0], array[:, 1], array[:, 2]
            bid_norm = bid / (all + minium)
            ask_norm = ask / (all + minium)
            imbalance_norm = (bid - ask) / (all + minium)

            pd_new["{}_bid_abnorm".format(features[0])] = bid_norm
            pd_new["{}_ask_abnorm".format(features[0])] = ask_norm
            pd_new["{}_askbid_imbalance_abnorm".format(features[0])] = imbalance_norm
    return pd_new


def find_ohlc_groups(features):
    # Initialize an empty dictionary to store the groups
    pattern = re.compile(r"^(.*?)(open|high|low|close|twap|awap|vwap)(.*)$")

    # Initialize a dictionary to hold the matched groups
    groups = {}
    matched_features = set()

    # Iterate over the features to match and group them
    for feature in features:
        match = pattern.match(feature)
        if match:
            # Extract the base (prefix and suffix), keyword, and suffix
            prefix, keyword, suffix = match.groups()
            # Use the combination of prefix and suffix as the key
            key = (prefix, suffix)
            # Add the feature to the corresponding group in the dictionary
            if key not in groups:
                groups[key] = []
            groups[key].append(feature)
            matched_features.add(feature)
    grouped_features = {k: v for k, v in groups.items()}
    unmatched_features = [
        feature for feature in features if feature not in matched_features
    ]

    return grouped_features, unmatched_features


def find_nquotes_groups(features):
    # Initialize an empty dictionary to store the groups
    pattern_1 = re.compile(r"^(.*?)(sell|buy)(.*)$")
    pattern_2 = re.compile(r"^(.*?)(up|down|flat)(.*)$")
    pattern_3 = re.compile(r"^(.*?)(bid|ask)(.*)$")
    # Initialize a dictionary to hold the matched groups
    groups_1 = {}
    groups_2 = {}
    groups_3 = {}
    matched_features = set()

    # Iterate over the features to match and group them
    for feature in features:
        match = pattern_1.match(feature)
        if match:
            # Extract the base (prefix and suffix), keyword, and suffix
            prefix, keyword, suffix = match.groups()
            # Use the combination of prefix and suffix as the key
            key = (prefix, suffix)
            # Add the feature to the corresponding group in the dictionary
            if key not in groups_1:
                groups_1[key] = []
            groups_1[key].append(feature)
            matched_features.add(feature)
    grouped_features_1 = {k: v for k, v in groups_1.items()}
    for grouped_feature in grouped_features_1:
        for name in features:
            if (name not in grouped_feature) and (name == grouped_feature[0][:-1]):
                grouped_features_1[grouped_feature].insert(0, name)
                matched_features.add(name)

    for feature in features:
        match = pattern_2.match(feature)
        if match:
            # Extract the base (prefix and suffix), keyword, and suffix
            prefix, keyword, suffix = match.groups()
            # Use the combination of prefix and suffix as the key
            key = (prefix, suffix)
            # Add the feature to the corresponding group in the dictionary
            if key not in groups_2:
                groups_2[key] = []
            groups_2[key].append(feature)
            matched_features.add(feature)
    grouped_features_2 = {k: v for k, v in groups_2.items()}
    for grouped_feature in grouped_features_2:
        prefix = (
            grouped_feature[0][:-1]
            if grouped_feature[0].endswith("_")
            else grouped_feature[0]
        )
        suffix = grouped_feature[1]
        for name in features:
            if (name not in grouped_feature) and (name == (prefix + suffix)):
                grouped_features_2[grouped_feature].insert(0, name)
                matched_features.add(name)
    keys_to_delete = [
        key for key, value in grouped_features_2.items() if len(value) == 1
    ]
    # 删除这些键
    for key in keys_to_delete:
        del grouped_features_2[key]

    for feature in features:
        match = pattern_3.match(feature)
        if match:
            # Extract the base (prefix and suffix), keyword, and suffix
            prefix, keyword, suffix = match.groups()
            # Use the combination of prefix and suffix as the key
            key = (prefix, suffix)
            # Add the feature to the corresponding group in the dictionary
            if key not in groups_3:
                groups_3[key] = []
            groups_3[key].append(feature)
            matched_features.add(feature)
    grouped_features_3 = {k: v for k, v in groups_3.items()}
    for grouped_feature in grouped_features_3:
        prefix = (
            grouped_feature[0][:-1]
            if grouped_feature[0].endswith("_")
            else grouped_feature[0]
        )
        suffix = grouped_feature[1]
        for name in features:
            if (name not in grouped_feature) and (name == (prefix + suffix)):
                grouped_features_3[grouped_feature].insert(0, name)
                matched_features.add(name)
    keys_to_delete = [
        key for key, value in grouped_features_3.items() if len(value) == 1
    ]
    # 删除这些键
    for key in keys_to_delete:
        del grouped_features_3[key]

    unmatched_features = [
        feature for feature in features if feature not in matched_features
    ]

    return (
        grouped_features_1,
        grouped_features_2,
        grouped_features_3,
        unmatched_features,
    )


def process_k_line_feature(df: pd.DataFrame):
    columns = df.columns
    grouped_features, _ = find_ohlc_groups(columns)
    df_generation_list = []
    for k, v in grouped_features.items():
        single_generation_df = pd.DataFrame(index=df.index)
        prefix, suffix = k
        vwap_judger = "{}vwap{}".format(prefix, suffix) in columns

        (
            open_feature,
            high_feature,
            low_name_feature,
            close_feature,
            twap_feature,
            awap_feature,
        ) = (
            "{}open{}".format(prefix, suffix),
            "{}high{}".format(prefix, suffix),
            "{}low{}".format(prefix, suffix),
            "{}close{}".format(prefix, suffix),
            "{}twap{}".format(prefix, suffix),
            "{}awap{}".format(prefix, suffix),
        )
        if vwap_judger:
            vwap_feature = "{}vwap{}".format(prefix, suffix)
        open, high, low, close, twap, awap = (
            df[open_feature].values,
            df[high_feature].values,
            df[low_name_feature].values,
            df[close_feature].values,
            df[twap_feature].values,
            df[awap_feature].values,
        )
        if vwap_judger:
            vwap = df[vwap_feature].values
        """
        "KMID","($close-$open)/$open",
        "KLEN","($high-$low)/$open",
        "KMID2","($close-$open)/($high-$low+1e-12)",
        "KUP","($high-Greater($open, $close))/$open",
        "KUP2","($high-Greater($open, $close))/($high-$low+1e-12)",
        "KLOW","(Less($open, $close)-$low)/$open",
        "KLOW2","(Less($open, $close)-$low)/($high-$low+1e-12)",
        "KSFT","(2*$close-$high-$low)/$open",
        "KSFT2","(2*$close-$high-$low)/($high-$low+1e-12)",
        ADDING NEW FEATURES WITH TWAP, AWAP AND VWAP(IF APPLIABLE)
        """
        # zero_open_mask = open == 0
        # zero_close_mask = close == 0

        # if zero_open_mask.any():
        #     print("分母为零的情况：")
        #     for i in np.where(zero_open_mask)[0]:
        #         print(f"索引: {i}, 分母 = {open[i]}")
        #         print(
        #             "{}open{}".format(prefix, suffix),
        #         )
        # if zero_close_mask.any():
        #     print("分母为零的情况：")
        #     for i in np.where(zero_close_mask)[0]:
        #         print(f"索引: {i}, 分母 = {close[i]}")
        #         print(
        #             "{}close{}".format(prefix, suffix),
        #         )
        # 注意到quotes里面的造的imblance volume的OHLC可能都是0，所以这里要加上一个minium
        klen = (high - low) / (open + minium)
        kmid = (close - open) / (open + minium)
        kmid2 = (close - open) / (high - low + minium)
        kup = (high - np.maximum(open, close)) / (open + minium)
        kup2 = (high - np.maximum(open, close)) / (high - low + minium)
        klow = (np.minimum(open, close) - low) / (open + minium)
        klow2 = (np.minimum(open, close) - low) / (high - low + minium)
        ksft = (2 * close - high - low) / (open + minium)
        ksft2 = (2 * close - high - low) / (high - low + minium)
        kotwap = (open - twap) / (open + minium)
        kotwap2 = (open - twap) / (high - low + minium)
        kctwap = (close - twap) / (close + minium)
        kctwap2 = (close - twap) / (high - low + minium)
        koawap = (open - awap) / (open + minium)
        koawap2 = (open - awap) / (high - low + minium)
        kcawap = (close - awap) / (close + minium)
        kcawap2 = (close - awap) / (high - low + minium)
        if vwap_judger:
            kovwap = (open - vwap) / (open + minium)
            kovwap2 = (open - vwap) / (high - low + minium)
            kcvwap = (close - vwap) / (close + minium)
            kcvwap2 = (close - vwap) / (high - low + minium)

        single_generation_df["{}klen{}".format(prefix, suffix)] = klen
        single_generation_df["{}kmid{}".format(prefix, suffix)] = kmid
        single_generation_df["{}kmid2{}".format(prefix, suffix)] = kmid2
        single_generation_df["{}kup{}".format(prefix, suffix)] = kup
        single_generation_df["{}kup2{}".format(prefix, suffix)] = kup2
        single_generation_df["{}klow{}".format(prefix, suffix)] = klow
        single_generation_df["{}klow2{}".format(prefix, suffix)] = klow2
        single_generation_df["{}ksft{}".format(prefix, suffix)] = ksft
        single_generation_df["{}ksft2{}".format(prefix, suffix)] = ksft2
        single_generation_df["{}kotwap{}".format(prefix, suffix)] = kotwap
        single_generation_df["{}kotwap2{}".format(prefix, suffix)] = kotwap2
        single_generation_df["{}kctwap{}".format(prefix, suffix)] = kctwap
        single_generation_df["{}kctwap2{}".format(prefix, suffix)] = kctwap2
        single_generation_df["{}koawap{}".format(prefix, suffix)] = koawap
        single_generation_df["{}koawap2{}".format(prefix, suffix)] = koawap2
        single_generation_df["{}kcawap{}".format(prefix, suffix)] = kcawap
        single_generation_df["{}kcawap2{}".format(prefix, suffix)] = kcawap2
        if vwap_judger:
            single_generation_df["{}kovwap{}".format(prefix, suffix)] = kovwap
            single_generation_df["{}kovwap2{}".format(prefix, suffix)] = kovwap2
            single_generation_df["{}kcvwap{}".format(prefix, suffix)] = kcvwap
            single_generation_df["{}kcvwap2{}".format(prefix, suffix)] = kcvwap2
        df_generation_list.append(single_generation_df)
    df_generation = pd.concat(df_generation_list, axis=1)
    return df_generation


def process_quotes_n_feature(df: pd.DataFrame):
    columns = df.columns
    _, unmatch_feature = find_ohlc_groups(columns)
    grouped_features_1, grouped_features_2, grouped_features_3, unmatched_feature = (
        find_nquotes_groups(unmatch_feature)
    )
    # print("grouped_features_1", grouped_features_1, "\n")
    # print("grouped_features_2", grouped_features_2, "\n")
    # print("grouped_features_3", grouped_features_3, "\n")
    # print("unmatched_feature", unmatched_feature, "\n")
    norm_df_list = []
    for section in grouped_features_1:
        pd_new = normalize_feature_cross_section(
            df, grouped_features_1[section], "buy_sell"
        )
        norm_df_list.append(pd_new)
    for section in grouped_features_2:
        pd_new = normalize_feature_cross_section(
            df, grouped_features_2[section], "up_down_flat"
        )
        norm_df_list.append(pd_new)
    for section in grouped_features_3:
        pd_new = normalize_feature_cross_section(
            df, grouped_features_3[section], "bid_ask"
        )
        norm_df_list.append(pd_new)
    df_norm = pd.concat(norm_df_list, axis=1)
    return df_norm


def process_snapshot_features(df: pd.DataFrame, topk=5, depth=25):
    if depth < 2:
        raise ValueError("orderbook depth must be at least 2")
    topk = min(topk, depth)
    ask_size_array = df[[f"ask{i}_size" for i in range(1, depth + 1)]].values
    bid_size_array = df[[f"bid{i}_size" for i in range(1, depth + 1)]].values
    ask_price_array = df[[f"ask{i}_price" for i in range(1, depth + 1)]].values
    bid_price_array = df[[f"bid{i}_price" for i in range(1, depth + 1)]].values
    normalized_ask_size_array = ask_size_array / (
        np.sum(ask_size_array, axis=1).reshape(-1, 1)
    )
    normalized_bid_size_array = bid_size_array / (
        np.sum(bid_size_array, axis=1).reshape(-1, 1)
    )

    best_ask_size_array = df["ask{}_size".format(1)].values
    best_ask_price_array = df["ask{}_price".format(1)].values
    best_bid_size_array = df["bid{}_size".format(1)].values
    best_bid_price_array = df["bid{}_price".format(1)].values

    ask_indices = np.argsort(ask_size_array, axis=1)[:, -topk:]
    bid_indices = np.argsort(bid_size_array, axis=1)[:, -topk:]
    ask_price_topk_size = np.take_along_axis(ask_price_array, ask_indices, axis=1)
    bid_price_topk_size = np.take_along_axis(bid_price_array, bid_indices, axis=1)
    ask_size_topk_size = np.take_along_axis(ask_size_array, ask_indices, axis=1)
    bid_size_topk_size = np.take_along_axis(bid_size_array, bid_indices, axis=1)

    price_related_df = pd.DataFrame(index=df.index)
    volume_related_df = pd.DataFrame(index=df.index)
    # price related features
    price_related_df["midprice"] = (df["ask1_price"] + df["bid1_price"]) / 2
    price_related_df["wap_1"] = (
        best_ask_size_array * best_bid_price_array
        + best_bid_size_array * best_ask_price_array
    ) / (best_ask_size_array + best_bid_size_array)

    price_related_df["wap_2"] = (
        ask_size_array[:, 1] * bid_price_array[:, 1]
        + bid_size_array[:, 1] * ask_price_array[:, 1]
    ) / (bid_size_array[:, 1] + ask_size_array[:, 1])
    price_related_df["wap_balance"] = (
        price_related_df["wap_1"] - price_related_df["wap_2"]
    )
    price_related_df["sell_wap"] = np.sum(
        normalized_ask_size_array * ask_price_array, axis=1
    )
    price_related_df["buy_wap"] = np.sum(
        normalized_bid_size_array * bid_price_array, axis=1
    )
    price_related_df["buy_sell_wap_spread"] = (
        price_related_df["buy_wap"] - price_related_df["sell_wap"]
    )
    price_related_df["buy_spread_oe_max"] = np.abs(
        df["bid1_price"] - df[f"bid{depth}_price"]
    )
    price_related_df["sell_spread_oe_max"] = np.abs(
        df["ask1_price"] - df[f"ask{depth}_price"]
    )
    topk_related_df = pd.DataFrame(
        columns=["ask_price_topk_size_{}_increments".format(i + 1) for i in range(topk)]
        + ["bid_price_topk_size_{}_increments".format(i + 1) for i in range(topk)]
        + ["ask_size_topk_size_{}_increments".format(i + 1) for i in range(topk)]
        + ["bid_size_topk_size_{}_increments".format(i + 1) for i in range(topk)],index=df.index
    )
    for i in range(topk):
        topk_related_df["ask_price_topk_size_{}_increments".format(i + 1)] = (
            ask_price_topk_size[:, i] - best_ask_price_array
        )
        topk_related_df["bid_price_topk_size_{}_increments".format(i + 1)] = (
            bid_price_topk_size[:, i] - best_bid_price_array
        )
        topk_related_df["ask_size_topk_size_{}_increments".format(i + 1)] = (
            ask_size_topk_size[:, i] - best_ask_size_array
        )
        topk_related_df["bid_size_topk_size_{}_increments".format(i + 1)] = (
            bid_size_topk_size[:, i] - best_bid_size_array
        )
    price_related_df = pd.concat([price_related_df, topk_related_df], axis=1)

    # volume related features
    volume_related_df["buy_volume_oe"] = np.sum(bid_size_array, axis=1)
    volume_related_df["sell_volume_oe"] = np.sum(ask_size_array, axis=1)
    volume_related_df["imblance_volume_oe"] = (
        volume_related_df["buy_volume_oe"] - volume_related_df["sell_volume_oe"]
    ) / (
        volume_related_df["buy_volume_oe"]
        + volume_related_df["sell_volume_oe"]
        + minium
    )
    all_normalized_size_df_list = []

    for i in range(1, depth + 1):
        single_normalized_size_df = pd.DataFrame(index=df.index)
        single_normalized_size_df["ask{}_size_n".format(i)] = (
            ask_size_array[:, i - 1] / volume_related_df["sell_volume_oe"]
        )
        single_normalized_size_df["bid{}_size_n".format(i)] = (
            bid_size_array[:, i - 1] / volume_related_df["buy_volume_oe"]
        )
        all_normalized_size_df_list.append(single_normalized_size_df)
    all_normalized_size_df = pd.concat(all_normalized_size_df_list, axis=1)

    volume_related_df = pd.concat([volume_related_df, all_normalized_size_df], axis=1)



    df = pd.concat([price_related_df, volume_related_df], axis=1)
    return df


def _as_polars_frame(df) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        out = df.clone()
    else:
        out = pl.from_pandas(df.reset_index())
    if "timestamp" not in out.columns and "index" in out.columns:
        out = out.rename({"index": "timestamp"})
    return out


def _timestamp_expr(df: pl.DataFrame) -> list[pl.Expr]:
    return [pl.col("timestamp")] if "timestamp" in df.columns else []


def normalize_feature_cross_section(df, features: list, method: str) -> pl.DataFrame:
    df = _as_polars_frame(df)
    assert method in ["buy_sell", "up_down_flat", "bid_ask"]
    expressions = _timestamp_expr(df)
    if method == "buy_sell":
        assert len(features) == 3
        all_feature, buy, sell = features
        expressions.extend(
            [
                (pl.col(buy) / (pl.col(all_feature) + minium)).alias(
                    f"{all_feature}_buy_bsnorm"
                ),
                (pl.col(sell) / (pl.col(all_feature) + minium)).alias(
                    f"{all_feature}_sell_bsnorm"
                ),
                ((pl.col(buy) - pl.col(sell)) / (pl.col(all_feature) + minium)).alias(
                    f"{all_feature}_buysell_imbalance_bsnorm"
                ),
            ]
        )
    elif method == "up_down_flat":
        assert len(features) in [3, 4]
        all_feature, up, down = features[:3]
        expressions.extend(
            [
                (pl.col(up) / (pl.col(all_feature) + minium)).alias(
                    f"{all_feature}_up_udnorm"
                ),
                (pl.col(down) / (pl.col(all_feature) + minium)).alias(
                    f"{all_feature}_down_udnorm"
                ),
                ((pl.col(up) - pl.col(down)) / (pl.col(all_feature) + minium)).alias(
                    f"{all_feature}_updown_imbalance_udnorm"
                ),
            ]
        )
        if len(features) == 4:
            flat = features[3]
            expressions.extend(
                [
                    (pl.col(flat) / (pl.col(all_feature) + minium)).alias(
                        f"{all_feature}_flat_udnorm"
                    ),
                    (
                        (pl.col(up) + pl.col(down) - pl.col(flat))
                        / (pl.col(all_feature) + minium)
                    ).alias(f"{all_feature}_updownflat_vol_udnorm"),
                ]
            )
    elif method == "bid_ask":
        assert len(features) in [2, 3]
        if len(features) == 2:
            bid, ask = features
            denom = pl.col(bid) + pl.col(ask) + minium
            expressions.extend(
                [
                    (pl.col(bid) / denom).alias(f"{bid}_abnorm"),
                    (pl.col(ask) / denom).alias(f"{ask}_abnorm"),
                    ((pl.col(bid) - pl.col(ask)) / denom).alias(
                        f"{ask}_bid_imbalance_abnorm"
                    ),
                ]
            )
        else:
            all_feature, bid, ask = features
            expressions.extend(
                [
                    (pl.col(bid) / (pl.col(all_feature) + minium)).alias(
                        f"{all_feature}_bid_abnorm"
                    ),
                    (pl.col(ask) / (pl.col(all_feature) + minium)).alias(
                        f"{all_feature}_ask_abnorm"
                    ),
                    ((pl.col(bid) - pl.col(ask)) / (pl.col(all_feature) + minium)).alias(
                        f"{all_feature}_askbid_imbalance_abnorm"
                    ),
                ]
            )
    return df.select(expressions)


def process_k_line_feature(df) -> pl.DataFrame:
    df = _as_polars_frame(df)
    columns = df.columns
    grouped_features, _ = find_ohlc_groups(columns)
    expressions = _timestamp_expr(df)
    for prefix, suffix in grouped_features:
        open_feature = f"{prefix}open{suffix}"
        high_feature = f"{prefix}high{suffix}"
        low_feature = f"{prefix}low{suffix}"
        close_feature = f"{prefix}close{suffix}"
        twap_feature = f"{prefix}twap{suffix}"
        awap_feature = f"{prefix}awap{suffix}"
        high_low = pl.col(high_feature) - pl.col(low_feature)
        expressions.extend(
            [
                (high_low / (pl.col(open_feature) + minium)).alias(f"{prefix}klen{suffix}"),
                ((pl.col(close_feature) - pl.col(open_feature)) / (pl.col(open_feature) + minium)).alias(f"{prefix}kmid{suffix}"),
                ((pl.col(close_feature) - pl.col(open_feature)) / (high_low + minium)).alias(f"{prefix}kmid2{suffix}"),
                ((pl.col(high_feature) - pl.max_horizontal(open_feature, close_feature)) / (pl.col(open_feature) + minium)).alias(f"{prefix}kup{suffix}"),
                ((pl.col(high_feature) - pl.max_horizontal(open_feature, close_feature)) / (high_low + minium)).alias(f"{prefix}kup2{suffix}"),
                ((pl.min_horizontal(open_feature, close_feature) - pl.col(low_feature)) / (pl.col(open_feature) + minium)).alias(f"{prefix}klow{suffix}"),
                ((pl.min_horizontal(open_feature, close_feature) - pl.col(low_feature)) / (high_low + minium)).alias(f"{prefix}klow2{suffix}"),
                ((2 * pl.col(close_feature) - pl.col(high_feature) - pl.col(low_feature)) / (pl.col(open_feature) + minium)).alias(f"{prefix}ksft{suffix}"),
                ((2 * pl.col(close_feature) - pl.col(high_feature) - pl.col(low_feature)) / (high_low + minium)).alias(f"{prefix}ksft2{suffix}"),
                ((pl.col(open_feature) - pl.col(twap_feature)) / (pl.col(open_feature) + minium)).alias(f"{prefix}kotwap{suffix}"),
                ((pl.col(open_feature) - pl.col(twap_feature)) / (high_low + minium)).alias(f"{prefix}kotwap2{suffix}"),
                ((pl.col(close_feature) - pl.col(twap_feature)) / (pl.col(close_feature) + minium)).alias(f"{prefix}kctwap{suffix}"),
                ((pl.col(close_feature) - pl.col(twap_feature)) / (high_low + minium)).alias(f"{prefix}kctwap2{suffix}"),
                ((pl.col(open_feature) - pl.col(awap_feature)) / (pl.col(open_feature) + minium)).alias(f"{prefix}koawap{suffix}"),
                ((pl.col(open_feature) - pl.col(awap_feature)) / (high_low + minium)).alias(f"{prefix}koawap2{suffix}"),
                ((pl.col(close_feature) - pl.col(awap_feature)) / (pl.col(close_feature) + minium)).alias(f"{prefix}kcawap{suffix}"),
                ((pl.col(close_feature) - pl.col(awap_feature)) / (high_low + minium)).alias(f"{prefix}kcawap2{suffix}"),
            ]
        )
        vwap_feature = f"{prefix}vwap{suffix}"
        if vwap_feature in columns:
            expressions.extend(
                [
                    ((pl.col(open_feature) - pl.col(vwap_feature)) / (pl.col(open_feature) + minium)).alias(f"{prefix}kovwap{suffix}"),
                    ((pl.col(open_feature) - pl.col(vwap_feature)) / (high_low + minium)).alias(f"{prefix}kovwap2{suffix}"),
                    ((pl.col(close_feature) - pl.col(vwap_feature)) / (pl.col(close_feature) + minium)).alias(f"{prefix}kcvwap{suffix}"),
                    ((pl.col(close_feature) - pl.col(vwap_feature)) / (high_low + minium)).alias(f"{prefix}kcvwap2{suffix}"),
                ]
            )
    return df.select(expressions)


def process_quotes_n_feature(df) -> pl.DataFrame:
    df = _as_polars_frame(df)
    _, unmatch_feature = find_ohlc_groups(df.columns)
    grouped_features_1, grouped_features_2, grouped_features_3, _ = find_nquotes_groups(
        unmatch_feature
    )
    frames = []
    for section in grouped_features_1:
        frames.append(normalize_feature_cross_section(df, grouped_features_1[section], "buy_sell"))
    for section in grouped_features_2:
        frames.append(normalize_feature_cross_section(df, grouped_features_2[section], "up_down_flat"))
    for section in grouped_features_3:
        frames.append(normalize_feature_cross_section(df, grouped_features_3[section], "bid_ask"))
    if not frames:
        return df.select(_timestamp_expr(df))
    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on="timestamp", how="inner") if "timestamp" in result.columns else pl.concat([result, frame], how="horizontal")
    return result


def process_snapshot_features(df, topk=5, depth=25) -> pl.DataFrame:
    df = _as_polars_frame(df)
    if depth < 2:
        raise ValueError("orderbook depth must be at least 2")
    topk = min(topk, depth)
    ask_size_array = df.select([f"ask{i}_size" for i in range(1, depth + 1)]).to_numpy()
    bid_size_array = df.select([f"bid{i}_size" for i in range(1, depth + 1)]).to_numpy()
    ask_price_array = df.select([f"ask{i}_price" for i in range(1, depth + 1)]).to_numpy()
    bid_price_array = df.select([f"bid{i}_price" for i in range(1, depth + 1)]).to_numpy()

    normalized_ask_size_array = ask_size_array / np.sum(ask_size_array, axis=1).reshape(-1, 1)
    normalized_bid_size_array = bid_size_array / np.sum(bid_size_array, axis=1).reshape(-1, 1)
    best_ask_size_array = ask_size_array[:, 0]
    best_ask_price_array = ask_price_array[:, 0]
    best_bid_size_array = bid_size_array[:, 0]
    best_bid_price_array = bid_price_array[:, 0]

    ask_indices = np.argsort(ask_size_array, axis=1)[:, -topk:]
    bid_indices = np.argsort(bid_size_array, axis=1)[:, -topk:]
    ask_price_topk_size = np.take_along_axis(ask_price_array, ask_indices, axis=1)
    bid_price_topk_size = np.take_along_axis(bid_price_array, bid_indices, axis=1)
    ask_size_topk_size = np.take_along_axis(ask_size_array, ask_indices, axis=1)
    bid_size_topk_size = np.take_along_axis(bid_size_array, bid_indices, axis=1)

    data = {}
    if "timestamp" in df.columns:
        data["timestamp"] = df["timestamp"].to_list()
    data["midprice"] = ((df["ask1_price"] + df["bid1_price"]) / 2).to_list()
    data["wap_1"] = (
        (best_ask_size_array * best_bid_price_array + best_bid_size_array * best_ask_price_array)
        / (best_ask_size_array + best_bid_size_array)
    )
    data["wap_2"] = (
        (ask_size_array[:, 1] * bid_price_array[:, 1] + bid_size_array[:, 1] * ask_price_array[:, 1])
        / (bid_size_array[:, 1] + ask_size_array[:, 1])
    )
    data["wap_balance"] = data["wap_1"] - data["wap_2"]
    data["sell_wap"] = np.sum(normalized_ask_size_array * ask_price_array, axis=1)
    data["buy_wap"] = np.sum(normalized_bid_size_array * bid_price_array, axis=1)
    data["buy_sell_wap_spread"] = data["buy_wap"] - data["sell_wap"]
    data["buy_spread_oe_max"] = np.abs(df["bid1_price"].to_numpy() - df[f"bid{depth}_price"].to_numpy())
    data["sell_spread_oe_max"] = np.abs(df["ask1_price"].to_numpy() - df[f"ask{depth}_price"].to_numpy())
    for i in range(topk):
        data[f"ask_price_topk_size_{i + 1}_increments"] = ask_price_topk_size[:, i] - best_ask_price_array
        data[f"bid_price_topk_size_{i + 1}_increments"] = bid_price_topk_size[:, i] - best_bid_price_array
        data[f"ask_size_topk_size_{i + 1}_increments"] = ask_size_topk_size[:, i] - best_ask_size_array
        data[f"bid_size_topk_size_{i + 1}_increments"] = bid_size_topk_size[:, i] - best_bid_size_array
    data["buy_volume_oe"] = np.sum(bid_size_array, axis=1)
    data["sell_volume_oe"] = np.sum(ask_size_array, axis=1)
    data["imblance_volume_oe"] = (
        (data["buy_volume_oe"] - data["sell_volume_oe"])
        / (data["buy_volume_oe"] + data["sell_volume_oe"] + minium)
    )
    for i in range(1, depth + 1):
        data[f"ask{i}_size_n"] = ask_size_array[:, i - 1] / data["sell_volume_oe"]
        data[f"bid{i}_size_n"] = bid_size_array[:, i - 1] / data["buy_volume_oe"]
    return pl.DataFrame(data)


if __name__ == "__main__":
    df = pd.read_feather(
        "./PREPROCESS_DATASET/binance-futures/DOWNSCALE_ORDERBOOK_25/BTCUSDT/10s/2023-01-01.feather"
    )
    df_norm = process_snapshot_features(df)
    print(df_norm)
    print(df_norm.columns)
