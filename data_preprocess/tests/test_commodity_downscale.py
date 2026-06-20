import pandas as pd
import pytest

from operator_futures.commodity.downscale import (
    create_second_level_snapshots,
    downscale_base_features,
    downscale_derivative_reference,
    downscale_orderbook,
    downscale_quote_features,
    validate_best_quotes,
)


SAMPLE_PATH = "docs/上海商品交易所/fu2302.csv"


def test_sample_file_can_create_depth_five_outputs():
    raw = pd.read_csv(SAMPLE_PATH).head(20)
    second = create_second_level_snapshots(raw)
    orderbook = downscale_orderbook(second, "5min", depth=5)
    derivative = downscale_derivative_reference(second, "5min", "fu")
    base = downscale_base_features(second, "5min")
    quote = downscale_quote_features(second, "5min")

    assert "ask5_price" in orderbook.columns
    assert "ask6_price" not in orderbook.columns
    assert "mark_price" in derivative.columns
    assert "ntrade_estimated" in base.columns
    assert "nquote" in quote.columns


def test_invalid_best_quote_fails_fast():
    raw = pd.read_csv(SAMPLE_PATH).head(2)
    raw.loc[0, "BidPrice1"] = raw.loc[0, "AskPrice1"]

    with pytest.raises(ValueError, match="BidPrice1"):
        validate_best_quotes(raw, "fu2302")


def test_second_level_uses_last_snapshot_per_second():
    raw = pd.read_csv(SAMPLE_PATH).head(4)
    raw.loc[2, "UpdateTime"] = raw.loc[1, "UpdateTime"]
    raw.loc[2, "BidPrice1"] = 2600.0

    second = create_second_level_snapshots(raw)

    timestamp = pd.Timestamp(
        f"{raw.loc[2, 'ActionDay']} {raw.loc[2, 'UpdateTime']}"
    )
    assert second.loc[timestamp.floor("s"), "BidPrice1"] == 2600.0


def test_derivative_reference_falls_back_to_midprice_for_invalid_lastprice():
    raw = pd.read_csv(SAMPLE_PATH).head(3)
    second = create_second_level_snapshots(raw)
    first_index = second.index[0]
    second.loc[first_index, "LastPrice"] = 0

    derivative = downscale_derivative_reference(second, "5min", "fu")

    expected_mid = (second.loc[first_index, "BidPrice1"] + second.loc[first_index, "AskPrice1"]) / 2
    assert derivative.iloc[0]["mark_price"] == expected_mid
    assert derivative.iloc[0]["funding_rate"] == 0


def test_empty_quote_window_fails_fast():
    raw = pd.read_csv(SAMPLE_PATH).head(2)
    second = create_second_level_snapshots(raw)

    with pytest.raises(ValueError, match="no quote snapshots"):
        downscale_quote_features(second.iloc[0:0], "5min")
