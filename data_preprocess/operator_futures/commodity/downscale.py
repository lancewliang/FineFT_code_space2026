from typing import List

import numpy as np
import pandas as pd

from .main_contract import normalize_timestamp
from .schema import resample_kwargs


def validate_best_quotes(df: pd.DataFrame, contract: str) -> None:
    invalid = (
        df["BidPrice1"].isna()
        | df["AskPrice1"].isna()
        | (df["BidPrice1"] <= 0)
        | (df["AskPrice1"] <= 0)
        | (df["BidPrice1"] >= df["AskPrice1"])
    )
    if invalid.any():
        first = df.loc[invalid].iloc[0]
        raise ValueError(
            "Invalid best quote for "
            f"{contract}: fields=['BidPrice1', 'AskPrice1'], "
            f"TradingDay={first.get('TradingDay')}, UpdateTime={first.get('UpdateTime')}"
        )


def create_second_level_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    copied = df.copy()
    contract = (
        str(copied["InstrumentID"].iloc[0])
        if "InstrumentID" in copied.columns and len(copied)
        else "unknown"
    )
    validate_best_quotes(copied, contract)
    copied["timestamp"] = copied.apply(normalize_timestamp, axis=1)
    copied = copied.sort_values("timestamp")
    copied["second"] = copied["timestamp"].dt.floor("s")
    second = copied.groupby("second", as_index=False).tail(1).copy()
    second = second.set_index("second").sort_index()
    second.index.name = "timestamp"
    return second


def _reference_price(df: pd.DataFrame) -> pd.Series:
    mid = (df["BidPrice1"] + df["AskPrice1"]) / 2
    valid = df["LastPrice"].notna() & (df["LastPrice"] > 0)
    if "UpperLimitPrice" in df.columns:
        valid &= df["LastPrice"] <= df["UpperLimitPrice"]
    if "LowerLimitPrice" in df.columns:
        valid &= df["LastPrice"] >= df["LowerLimitPrice"]
    return df["LastPrice"].where(valid, mid)


def downscale_derivative_reference(
    second_df: pd.DataFrame, target_freq: str, symbol: str
) -> pd.DataFrame:
    price = _reference_price(second_df)
    frame = pd.DataFrame(
        {
            "symbol": symbol,
            "funding_timestamp": second_df.index,
            "funding_rate": 0.0,
            "index_price": price,
            "mark_price": price,
        },
        index=second_df.index,
    )
    result = frame.resample(target_freq, **resample_kwargs()).first()
    return result.dropna(subset=["mark_price"]).reset_index()


def downscale_orderbook(
    second_df: pd.DataFrame, target_freq: str, depth: int = 5
) -> pd.DataFrame:
    renamed = pd.DataFrame(index=second_df.index)
    for level in range(1, depth + 1):
        renamed[f"ask{level}_price"] = second_df[f"AskPrice{level}"]
        renamed[f"ask{level}_size"] = second_df[f"AskVolume{level}"]
        renamed[f"bid{level}_price"] = second_df[f"BidPrice{level}"]
        renamed[f"bid{level}_size"] = second_df[f"BidVolume{level}"]
    result = renamed.resample(target_freq, **resample_kwargs()).last()
    return result.dropna(how="all").reset_index()


def _second_trade_frame(second_df: pd.DataFrame) -> pd.DataFrame:
    frame = second_df.copy()
    frame["second_volume"] = pd.to_numeric(frame["Volume"], errors="coerce").diff()
    frame["second_tradeval"] = pd.to_numeric(frame["Turnover"], errors="coerce").diff()
    invalid = (frame["second_volume"] > 0) & (
        frame["second_tradeval"].isna() | (frame["second_tradeval"] <= 0)
    )
    if invalid.any():
        row = frame.loc[invalid].iloc[0]
        raise ValueError(
            "Invalid turnover delta with positive volume: "
            f"timestamp={row.name}, contract={row.get('InstrumentID')}, "
            f"second_volume={row['second_volume']}, "
            f"second_tradeval={row['second_tradeval']}"
        )

    frame["second_avg_price"] = frame["second_tradeval"] / frame["second_volume"]
    frame.loc[frame["second_volume"] <= 0, "second_avg_price"] = np.nan
    valid_price = frame["second_avg_price"].dropna()
    direction = pd.Series("none", index=frame.index, dtype=object)
    diff = valid_price.diff()
    direction.loc[diff[diff > 0].index] = "buy_estimated"
    direction.loc[diff[diff < 0].index] = "sell_estimated"
    direction.loc[diff[diff == 0].index] = "flat"
    frame["direction_estimated"] = direction
    return frame


def downscale_base_features(second_df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    frame = _second_trade_frame(second_df)
    price = frame["second_avg_price"].copy()
    price = price.fillna(_reference_price(frame))

    out = pd.DataFrame(index=frame.index)
    out["price"] = price
    out["volume"] = frame["second_volume"].clip(lower=0).fillna(0)
    out["tradeval"] = (
        frame["second_tradeval"].where(frame["second_volume"] > 0, 0).fillna(0)
    )
    ohlc = out["price"].resample(target_freq, **resample_kwargs()).ohlc()
    grouped = pd.DataFrame(index=ohlc.index)
    grouped[["open", "high", "low", "close"]] = ohlc[
        ["open", "high", "low", "close"]
    ]
    grouped["volume"] = out["volume"].resample(
        target_freq, **resample_kwargs()
    ).sum()
    grouped["tradeval"] = out["tradeval"].resample(
        target_freq, **resample_kwargs()
    ).sum()
    grouped["vwap"] = grouped["tradeval"] / grouped["volume"]
    grouped["vwap"] = grouped["vwap"].where(grouped["volume"] > 0, grouped["close"])
    grouped["awap"] = out["price"].resample(target_freq, **resample_kwargs()).mean()
    grouped["twap"] = grouped["awap"]
    grouped["ntrade_estimated"] = (frame["second_volume"] > 0).resample(
        target_freq, **resample_kwargs()
    ).sum()
    grouped["ntrade_up_estimated"] = (
        frame["direction_estimated"] == "buy_estimated"
    ).resample(target_freq, **resample_kwargs()).sum()
    grouped["ntrade_down_estimated"] = (
        frame["direction_estimated"] == "sell_estimated"
    ).resample(target_freq, **resample_kwargs()).sum()
    grouped["ntrade_flat_estimated"] = (
        frame["direction_estimated"] == "flat"
    ).resample(target_freq, **resample_kwargs()).sum()
    return grouped.dropna(subset=["open"]).reset_index()


def _change_count(series: pd.Series, direction: str | None = None) -> pd.Series:
    diff = series.diff()
    if direction == "up":
        return diff > 0
    if direction == "down":
        return diff < 0
    return diff.ne(0)


def downscale_quote_features(second_df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    if second_df.empty:
        raise ValueError("Target window has no quote snapshots")

    quote = pd.DataFrame(index=second_df.index)
    quote["bid_price"] = second_df["BidPrice1"]
    quote["ask_price"] = second_df["AskPrice1"]
    quote["bid_amount"] = second_df["BidVolume1"]
    quote["ask_amount"] = second_df["AskVolume1"]
    quote["spread"] = quote["ask_price"] - quote["bid_price"]
    quote["mid"] = (quote["ask_price"] + quote["bid_price"]) / 2
    quote["imbalance_volume"] = (quote["bid_amount"] - quote["ask_amount"]) / (
        quote["bid_amount"] + quote["ask_amount"]
    )
    quote["bid"] = quote["bid_price"]
    quote["ask"] = quote["ask_price"]
    quote["bidsize"] = quote["bid_amount"]
    quote["asksize"] = quote["ask_amount"]

    result = pd.DataFrame(index=quote.resample(target_freq, **resample_kwargs()).last().index)
    result["nquote"] = quote["bid_price"].resample(
        target_freq, **resample_kwargs()
    ).count()
    result["nquote_bid"] = _change_count(quote["bid_price"]).resample(
        target_freq, **resample_kwargs()
    ).sum()
    result["nquote_ask"] = _change_count(quote["ask_price"]).resample(
        target_freq, **resample_kwargs()
    ).sum()
    result["nquote_bid_up"] = _change_count(quote["bid_price"], "up").resample(
        target_freq, **resample_kwargs()
    ).sum()
    result["nquote_bid_down"] = _change_count(quote["bid_price"], "down").resample(
        target_freq, **resample_kwargs()
    ).sum()
    result["nquote_ask_up"] = _change_count(quote["ask_price"], "up").resample(
        target_freq, **resample_kwargs()
    ).sum()
    result["nquote_ask_down"] = _change_count(quote["ask_price"], "down").resample(
        target_freq, **resample_kwargs()
    ).sum()

    for name in ["spread", "mid", "imbalance_volume", "bid", "ask", "bidsize", "asksize"]:
        ohlc = quote[name].resample(target_freq, **resample_kwargs()).ohlc()
        for field in ["open", "high", "low", "close"]:
            result[f"{field}_{name}"] = ohlc[field]
        result[f"awap_{name}"] = quote[name].resample(
            target_freq, **resample_kwargs()
        ).mean()
        result[f"twap_{name}"] = result[f"awap_{name}"]

    empty_windows = result["nquote"] == 0
    if empty_windows.any():
        first = result.loc[empty_windows].index[0]
        raise ValueError(f"Target window has no quote snapshots: {first}")
    return result.reset_index()
