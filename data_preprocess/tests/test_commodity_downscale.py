from datetime import datetime
import logging
from types import SimpleNamespace

import polars as pl
import pytest

from operator_futures.commodity.main_contract import MainContractSummary
from operator_futures.data_quality import DataQualityValidator
import operator_futures.commodity.downscale_continuous_by_trading_day as continuous_downscale
from operator_futures.commodity.downscale_continuous_by_trading_day import (
    _write_downscaled_day,
    iter_summary_trading_days,
)
from operator_futures.commodity.downscale import (
    create_second_level_snapshots,
    downscale_base_features,
    downscale_derivative_reference,
    downscale_orderbook,
    downscale_quote_features,
    downscale_quote_ofi_features,
    validate_best_quotes,
)


SAMPLE_PATH = "docs/上海商品交易所/fu2302.csv"


def _five_depth_quote_frame(row_overrides: list[dict]) -> pl.DataFrame:
    rows = []
    for index, overrides in enumerate(row_overrides):
        row = {"timestamp": datetime(2026, 2, 2, 9, 0, index)}
        for level in range(1, 6):
            row[f"BidPrice{level}"] = 100.0 - level
            row[f"AskPrice{level}"] = 100.0 + level
            row[f"BidVolume{level}"] = float(level * 10)
            row[f"AskVolume{level}"] = float(level * 20)
        row.update(overrides)
        rows.append(row)
    return pl.DataFrame(rows)


def test_downscale_quote_ofi_features_computes_five_depth_direction_math():
    frame = _five_depth_quote_frame(
        [
            {},
            {
                "BidPrice1": 100.0,
                "BidVolume1": 11.0,
                "BidVolume2": 25.0,
                "BidPrice3": 96.0,
                "BidVolume3": 35.0,
                "BidVolume4": 35.0,
                "AskPrice1": 100.0,
                "AskVolume1": 21.0,
                "AskVolume2": 42.0,
                "AskPrice3": 104.0,
                "AskVolume3": 65.0,
                "AskVolume4": 70.0,
            },
        ]
    )

    result = downscale_quote_ofi_features(frame, window_rows=12)
    row = result.row(0, named=True)

    assert row["timestamp"] == datetime(2026, 2, 2, 9, 0, 1)
    assert row["nquote"] == 2
    assert row["ofi_bid1"] == 11.0
    assert row["ofi_bid2"] == 5.0
    assert row["ofi_bid3"] == -30.0
    assert row["ofi_bid4"] == -5.0
    assert row["ofi_bid5"] == 0.0
    assert row["ofi_ask1"] == -21.0
    assert row["ofi_ask2"] == -2.0
    assert row["ofi_ask3"] == 60.0
    assert row["ofi_ask4"] == 10.0
    assert row["ofi_ask5"] == 0.0
    assert row["ofi_bid"] == -19.0
    assert row["ofi_ask"] == 47.0
    assert row["ofi"] == 28.0


def test_downscale_quote_ofi_features_aggregates_every_twelve_rows_and_keeps_tail():
    frame = _five_depth_quote_frame(
        [{"BidVolume1": float(10 + index)} for index in range(13)]
    )

    result = downscale_quote_ofi_features(frame)

    assert result["timestamp"].to_list() == [
        datetime(2026, 2, 2, 9, 0, 11),
        datetime(2026, 2, 2, 9, 0, 12),
    ]
    assert result["nquote"].to_list() == [12, 1]
    assert result["ofi_bid1"].to_list() == [11.0, 1.0]
    assert result["ofi_bid"].to_list() == [11.0, 1.0]
    assert result["ofi_ask"].to_list() == [0.0, 0.0]
    assert result["ofi"].to_list() == [11.0, 1.0]


def test_downscale_quote_ofi_features_compares_across_row_window_boundary():
    frame = _five_depth_quote_frame(
        [{} for _ in range(12)] + [{"BidPrice1": 100.0, "BidVolume1": 77.0}]
    )

    result = downscale_quote_ofi_features(frame)
    boundary_row = result.row(1, named=True)

    assert boundary_row["timestamp"] == datetime(2026, 2, 2, 9, 0, 12)
    assert boundary_row["nquote"] == 1
    assert boundary_row["ofi_bid1"] == 77.0
    assert boundary_row["ofi_bid"] == 77.0
    assert boundary_row["ofi"] == 77.0


def test_downscale_quote_ofi_features_rejects_empty_input():
    with pytest.raises(ValueError, match="OFI input has no quote snapshots"):
        downscale_quote_ofi_features(pl.DataFrame())


def test_downscale_quote_ofi_features_rejects_missing_depth_columns():
    frame = _five_depth_quote_frame([{}]).drop("BidPrice5")

    with pytest.raises(ValueError, match="Missing OFI columns: BidPrice5"):
        downscale_quote_ofi_features(frame)


def test_downscale_quote_ofi_features_rejects_null_depth_values():
    frame = _five_depth_quote_frame([{"AskVolume4": None}])

    with pytest.raises(ValueError, match="OFI columns contain null values: AskVolume4"):
        downscale_quote_ofi_features(frame)


def test_downscale_quote_ofi_features_rejects_non_finite_depth_values():
    frame = _five_depth_quote_frame(
        [{"BidVolume2": float("nan"), "AskPrice3": float("inf")}]
    )

    with pytest.raises(
        ValueError,
        match="OFI columns contain non-finite values: BidVolume2, AskPrice3",
    ):
        downscale_quote_ofi_features(frame)


def test_downscale_quote_ofi_features_rejects_invalid_window_rows():
    frame = _five_depth_quote_frame([{}])

    with pytest.raises(ValueError, match="window_rows must be positive"):
        downscale_quote_ofi_features(frame, window_rows=0)


def test_downscale_quote_ofi_features_outputs_normalized_ofi():
    frame = _five_depth_quote_frame(
        [
            {},
            {
                "BidPrice1": 100.0,
                "BidVolume1": 11.0,
                "BidVolume2": 25.0,
                "BidPrice3": 96.0,
                "BidVolume3": 35.0,
                "BidVolume4": 35.0,
                "AskPrice1": 100.0,
                "AskVolume1": 21.0,
                "AskVolume2": 42.0,
                "AskPrice3": 104.0,
                "AskVolume3": 65.0,
                "AskVolume4": 70.0,
            },
        ]
    )

    result = downscale_quote_ofi_features(frame, window_rows=12)
    row = result.row(0, named=True)

    assert row["ofi_bid_norm"] == pytest.approx(-19.0 / 306.0)
    assert row["ofi_ask_norm"] == pytest.approx(47.0 / 598.0)
    assert row["ofi_norm"] == pytest.approx(28.0 / 904.0)


def test_downscale_quote_ofi_features_zeroes_normalized_ofi_when_denominator_is_zero():
    frame = _five_depth_quote_frame(
        [
            {
                "BidVolume1": 0.0,
                "BidVolume2": 0.0,
                "BidVolume3": 0.0,
                "BidVolume4": 0.0,
                "BidVolume5": 0.0,
                "AskVolume1": 0.0,
                "AskVolume2": 0.0,
                "AskVolume3": 0.0,
                "AskVolume4": 0.0,
                "AskVolume5": 0.0,
            }
        ]
    )

    result = downscale_quote_ofi_features(frame, window_rows=12)
    row = result.row(0, named=True)

    assert row["ofi_bid_norm"] == 0.0
    assert row["ofi_ask_norm"] == 0.0
    assert row["ofi_norm"] == 0.0


def test_iter_summary_trading_days_accepts_summary_model(tmp_path):
    source_file = tmp_path / "fu2601.csv"
    source_file.write_text("placeholder\n", encoding="utf-8")
    summary = MainContractSummary.from_dict(
        {
            "symbol": "fu",
            "commodity_name": "燃料油",
            "start_date": "2026-01-05",
            "end_date": "2026-01-06",
            "selection_rule": "monthly_top_2_by_sum_daily_volume_delta",
            "contracts": [
                {
                    "contract": "fu2601",
                    "start_trading_day": "20260105",
                    "end_trading_day": "20260105",
                    "trading_day_count": 1,
                    "selected_months": ["2026-01"],
                    "trading_days": [
                        {
                            "trading_day": "20260105",
                            "date": "2026-01-05",
                            "source_file": str(source_file),
                            "daily_volume": 100.0,
                        }
                    ],
                }
            ],
        }
    )

    days = list(iter_summary_trading_days(summary))

    assert days[0].contract == "fu2601"
    assert days[0].date == "2026-01-05"
    assert days[0].source_file == source_file


def test_downscale_continuous_terminates_process_pool_on_worker_error(
    tmp_path, monkeypatch
):
    source_file_1 = tmp_path / "fu2601_20260105.csv"
    source_file_2 = tmp_path / "fu2601_20260106.csv"
    source_file_1.write_text("placeholder\n", encoding="utf-8")
    source_file_2.write_text("placeholder\n", encoding="utf-8")
    summary = MainContractSummary.from_dict(
        {
            "symbol": "fu",
            "commodity_name": "燃料油",
            "start_date": "2026-01-05",
            "end_date": "2026-01-07",
            "selection_rule": "monthly_top_2_by_sum_daily_volume_delta",
            "contracts": [
                {
                    "contract": "fu2601",
                    "start_trading_day": "20260105",
                    "end_trading_day": "20260106",
                    "trading_day_count": 2,
                    "selected_months": ["2026-01"],
                    "trading_days": [
                        {
                            "trading_day": "20260105",
                            "date": "2026-01-05",
                            "source_file": str(source_file_1),
                            "daily_volume": 100.0,
                        },
                        {
                            "trading_day": "20260106",
                            "date": "2026-01-06",
                            "source_file": str(source_file_2),
                            "daily_volume": 90.0,
                        },
                    ],
                }
            ],
        }
    )

    class FailingPool:
        def __init__(self):
            self.closed = False
            self.initializer = None
            self.joined = False
            self.processes = None
            self.tasks = None
            self.terminated = False

        def imap_unordered(self, worker, tasks):
            self.tasks = list(tasks)
            yield ("fu2601", "20260105")
            raise RuntimeError("worker failed")

        def close(self):
            self.closed = True

        def terminate(self):
            self.terminated = True

        def join(self):
            self.joined = True

    pool = FailingPool()

    def build_pool(processes, initializer):
        pool.processes = processes
        pool.initializer = initializer
        return pool

    monkeypatch.setattr(
        continuous_downscale,
        "mp",
        SimpleNamespace(get_context=lambda method: SimpleNamespace(Pool=build_pool)),
        raising=False,
    )
    monkeypatch.setattr(
        continuous_downscale,
        "load_main_contract_summary",
        lambda summary_path: summary,
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        continuous_downscale.downscale_continuous_by_trading_day(
            summary_path=tmp_path / "summary.json",
            output_root=tmp_path,
            target_freq="5min",
            symbol="fu",
            max_workers=2,
        )

    assert pool.processes == 2
    assert pool.initializer is continuous_downscale.configure_logging
    assert pool.tasks is not None
    assert len(pool.tasks) == 2
    assert pool.terminated
    assert pool.joined
    assert not pool.closed


def test_sample_file_can_create_depth_five_outputs():
    raw = pl.read_csv(SAMPLE_PATH).head(20)
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


def test_downscale_orderbook_preserves_price_limit_columns():
    second = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 2, 2, 9, 0, 1),
                datetime(2026, 2, 2, 9, 4, 59),
            ],
            "LowerLimitPrice": [2500.0, 2501.0],
            "UpperLimitPrice": [3100.0, 3101.0],
            "AskPrice1": [3001.0, 3002.0],
            "AskVolume1": [10, 11],
            "BidPrice1": [2999.0, 3000.0],
            "BidVolume1": [20, 21],
            "AskPrice2": [3003.0, 3004.0],
            "AskVolume2": [12, 13],
            "BidPrice2": [2998.0, 2999.0],
            "BidVolume2": [22, 23],
            "AskPrice3": [3005.0, 3006.0],
            "AskVolume3": [14, 15],
            "BidPrice3": [2997.0, 2998.0],
            "BidVolume3": [24, 25],
            "AskPrice4": [3007.0, 3008.0],
            "AskVolume4": [16, 17],
            "BidPrice4": [2996.0, 2997.0],
            "BidVolume4": [26, 27],
            "AskPrice5": [3009.0, 3010.0],
            "AskVolume5": [18, 19],
            "BidPrice5": [2995.0, 2996.0],
            "BidVolume5": [28, 29],
        }
    )

    out = downscale_orderbook(second, "5min", depth=5)
    row = out.filter(pl.col("timestamp") == datetime(2026, 2, 2, 9, 5, 0))

    assert row.item(0, "LowerLimitPrice") == 2501.0
    assert row.item(0, "UpperLimitPrice") == 3101.0


def test_invalid_best_quote_fails_fast():
    raw = pl.read_csv(SAMPLE_PATH).head(2)
    ask_price = raw.item(0, "AskPrice1")
    raw = raw.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(ask_price))
        .otherwise(pl.col("BidPrice1"))
        .alias("BidPrice1")
    )

    with pytest.raises(ValueError) as exc_info:
        validate_best_quotes(raw, "fu2302")

    message = str(exc_info.value)
    assert f"BidPrice1={ask_price}" in message
    assert f"AskPrice1={ask_price}" in message
    assert "reason=BidPrice1 >= AskPrice1" in message
    assert "row={" in message
    assert "'InstrumentID': 'fu2302'" in message
    assert "'LastPrice':" in message


def test_illegal_value_validation_reports_null_nan_and_infinite(caplog):
    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 5, 9, 0, 0), None],
            "nan_feature": [1.0, float("nan")],
            "null_feature": [None, 2.0],
            "inf_feature": [float("inf"), 3.0],
        }
    )

    with pytest.raises(ValueError) as exc_info:
        DataQualityValidator.validate_no_illegal_values(
            frame,
            stage="test_stage",
            feature_name="TEST_FEATURE",
            contract="fu2601",
            trading_day="20260105",
        )

    message = str(exc_info.value)
    assert "stage=test_stage" in message
    assert "feature=TEST_FEATURE" in message
    assert "nan_feature:nan=1" in message
    assert "null_feature:null=1" in message
    assert "inf_feature:infinite=1" in message
    assert "timestamp:null=1" in message
    assert "Illegal data detected" in caplog.text


def test_illegal_value_validation_checks_only_requested_columns():
    frame = pl.DataFrame(
        {
            "checked": [1.0, 2.0],
            "ignored": [None, float("nan")],
        }
    )

    DataQualityValidator.validate_no_illegal_values(
        frame,
        stage="test_stage",
        feature_name="TEST_FEATURE",
        contract="fu2601",
        trading_day="20260105",
        columns=["checked"],
    )


def test_continuous_downscale_ignores_unused_second_level_null_columns(tmp_path):
    raw = pl.read_csv(SAMPLE_PATH).head(2).with_columns(
        pl.col("LastPrice").alias("LowPrice"),
        pl.col("LastPrice").alias("HighPrice"),
    )

    trading_day = _write_downscaled_day(raw, tmp_path, "5min", "fu", "fu2302", depth=5)

    assert trading_day == "20230104"
    assert list(tmp_path.rglob("*.feather"))


def test_continuous_downscale_rejects_illegal_second_level_values(tmp_path, caplog):
    raw = pl.read_csv(SAMPLE_PATH).head(2).with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit(float("nan")))
        .otherwise(pl.col("LastPrice"))
        .alias("LastPrice")
    )

    with pytest.raises(ValueError) as exc_info:
        _write_downscaled_day(raw, tmp_path, "5min", "fu", "fu2302", depth=5)

    message = str(exc_info.value)
    assert "stage=second_level_snapshots" in message
    assert "contract=fu2302" in message
    assert "LastPrice:nan=1" in message
    assert "Illegal data detected" in caplog.text
    assert not list(tmp_path.rglob("*.feather"))


def test_continuous_downscale_rejects_illegal_feature_outputs(tmp_path, caplog):
    raw = pl.read_csv(SAMPLE_PATH).head(2).fill_null(0).with_columns(
        pl.lit(0.0).alias("ClosePrice"),
        pl.lit(0.0).alias("SettlementPrice"),
        pl.lit(0.0).alias("PreDelta"),
        pl.lit(0.0).alias("CurrDelta"),
        pl.lit(0.0).alias("AveragePrice"),
        pl.lit(0).alias("BidVolume1"),
        pl.lit(0).alias("AskVolume1"),
    )

    with pytest.raises(ValueError) as exc_info:
        _write_downscaled_day(raw, tmp_path, "5min", "fu", "fu2302", depth=5)

    message = str(exc_info.value)
    assert "stage=feature_output" in message
    assert "feature=COMMODITY_QUOTE_FEATURE" in message
    assert "imbalance_volume" in message
    assert "nan=1" in message
    assert "Illegal data detected" in caplog.text
    assert not list(tmp_path.rglob("*.feather"))


def test_second_level_snapshots_handle_string_quote_columns():
    raw = pl.read_csv(SAMPLE_PATH).head(1).with_columns(
        pl.col("AskPrice1").cast(pl.Utf8)
    )

    second = create_second_level_snapshots(raw)

    assert second.height == 1
    assert second.item(0, "BidPrice1") == 2593.0
    assert second.item(0, "AskPrice1") == 2638.0


def test_second_level_fills_empty_ohlc_with_last_price_when_no_trade():
    raw = pl.read_csv(SAMPLE_PATH).head(1).with_columns(
        pl.lit(None).alias("OpenPrice"),
        pl.lit(None).alias("HighPrice"),
        pl.lit(None).alias("LowPrice"),
        pl.lit(0).alias("Volume"),
        pl.lit(0.0).alias("Turnover"),
    )

    second = create_second_level_snapshots(raw)

    assert second.item(0, "OpenPrice") == second.item(0, "LastPrice")
    assert second.item(0, "HighPrice") == second.item(0, "LastPrice")
    assert second.item(0, "LowPrice") == second.item(0, "LastPrice")


def test_second_level_gap_fill_logs_source_rule_and_line(caplog):
    raw = pl.read_csv(SAMPLE_PATH).head(2).with_columns(
        pl.lit(None).alias("OpenPrice"),
        pl.lit(None).alias("HighPrice"),
        pl.lit(None).alias("LowPrice"),
        pl.lit(0).alias("Volume"),
        pl.lit(0.0).alias("Turnover"),
    )

    with caplog.at_level(logging.INFO, logger="operator_futures.commodity.downscale"):
        create_second_level_snapshots(raw, source_file="input.csv")

    fill_messages = [
        record.getMessage()
        for record in caplog.records
        if "Second-level gap filled" in record.getMessage()
    ]
    messages = "\n".join(fill_messages)
    assert len(fill_messages) == 2
    assert "Second-level gap filled" in messages
    assert "source_file=input.csv" in messages
    assert "source_line=2" in messages
    assert "source_line=3" in messages
    assert "rule=empty_ohlc_no_trade" in messages
    assert "columns=OpenPrice,HighPrice,LowPrice" in messages
    assert "old_values={'OpenPrice': None, 'HighPrice': None, 'LowPrice': None}" in messages
    assert "new_value=" in messages


def test_second_level_fills_empty_ask_volumes_from_previous_level():
    raw = pl.read_csv(SAMPLE_PATH).head(1).with_columns(
        pl.lit(12).alias("AskVolume1"),
        pl.lit(None).alias("AskPrice2"),
        pl.lit(0).alias("AskVolume2"),
        pl.lit(None).alias("AskPrice3"),
        pl.lit(0).alias("AskVolume3"),
        pl.lit(None).alias("AskPrice4"),
        pl.lit(0).alias("AskVolume4"),
        pl.lit(None).alias("AskPrice5"),
        pl.lit(0).alias("AskVolume5"),
    )

    second = create_second_level_snapshots(raw)

    for level in range(2, 6):
        assert second.item(0, f"AskVolume{level}") == 12


def test_second_level_fills_empty_ask_price_and_volume_depth_gaps():
    raw = pl.read_csv(SAMPLE_PATH).head(1).with_columns(
        pl.lit(12345.0).alias("AskPrice3"),
        pl.lit(7).alias("AskVolume3"),
        pl.lit(None).alias("AskPrice4"),
        pl.lit(0).alias("AskVolume4"),
        pl.lit(None).alias("AskPrice5"),
        pl.lit(0).alias("AskVolume5"),
    )

    second = create_second_level_snapshots(raw)

    for level in (4, 5):
        assert second.item(0, f"AskPrice{level}") == 12345.0
        assert second.item(0, f"AskVolume{level}") == 7


def test_second_level_fills_empty_bid_price_and_volume_depth_gaps():
    raw = pl.read_csv(SAMPLE_PATH).head(1).with_columns(
        pl.lit(3089.0).alias("LastPrice"),
        pl.lit(3089.0).alias("LowPrice"),
        pl.lit(3088.0).alias("LowerLimitPrice"),
        pl.lit(3089.0).alias("BidPrice1"),
        pl.lit(15).alias("BidVolume1"),
        pl.lit(3088.0).alias("BidPrice2"),
        pl.lit(61).alias("BidVolume2"),
        pl.lit(None).alias("BidPrice3"),
        pl.lit(0).alias("BidVolume3"),
        pl.lit(None).alias("BidPrice4"),
        pl.lit(0).alias("BidVolume4"),
        pl.lit(None).alias("BidPrice5"),
        pl.lit(0).alias("BidVolume5"),
        pl.lit(3100.0).alias("AskPrice1"),
    )

    second = create_second_level_snapshots(raw)

    for level in (3, 4, 5):
        assert second.item(0, f"BidPrice{level}") == 3088.0
        assert second.item(0, f"BidVolume{level}") == 61


def test_second_level_fills_limit_up_empty_ask_prices():
    raw = pl.read_csv(SAMPLE_PATH).head(1)
    upper_limit = raw.item(0, "UpperLimitPrice")
    raw = raw.with_columns(
        [pl.lit(upper_limit).alias("LastPrice")]
        + [
            pl.lit(None).alias(f"AskPrice{level}")
            for level in range(1, 6)
        ]
        + [
            pl.lit(0).alias(f"AskVolume{level}")
            for level in range(1, 6)
        ]
    )

    second = create_second_level_snapshots(raw)

    for level in range(1, 6):
        assert second.item(0, f"AskPrice{level}") == upper_limit


def test_second_level_fills_low_price_empty_ask_prices():
    raw = pl.read_csv(SAMPLE_PATH).filter(pl.col("LowPrice").is_not_null()).head(1)
    low_price = raw.item(0, "LowPrice")
    upper_limit = raw.item(0, "UpperLimitPrice")
    raw = raw.with_columns(
        [pl.lit(low_price).alias("LastPrice")]
        + [
            pl.lit(None).alias(f"AskPrice{level}")
            for level in range(1, 6)
        ]
        + [
            pl.lit(0).alias(f"AskVolume{level}")
            for level in range(1, 6)
        ]
    )

    second = create_second_level_snapshots(raw)

    for level in range(1, 6):
        assert second.item(0, f"AskPrice{level}") == upper_limit


def test_second_level_fills_limit_down_empty_bid_prices():
    raw = pl.read_csv(SAMPLE_PATH).head(1)
    lower_limit = raw.item(0, "LowerLimitPrice")
    raw = raw.with_columns(
        [pl.lit(lower_limit).alias("LastPrice")]
        + [
            pl.lit(None).alias(f"BidPrice{level}")
            for level in range(1, 6)
        ]
        + [
            pl.lit(0).alias(f"BidVolume{level}")
            for level in range(1, 6)
        ]
    )

    second = create_second_level_snapshots(raw)

    for level in range(1, 6):
        assert second.item(0, f"BidPrice{level}") == lower_limit


def test_second_level_drops_rows_with_all_depth_prices_null():
    raw = pl.read_csv(SAMPLE_PATH).head(2)
    dropped_timestamp = datetime.strptime(
        f"{raw.item(0, 'ActionDay')} {raw.item(0, 'UpdateTime')}",
        "%Y%m%d %H:%M:%S.%f",
    ).replace(microsecond=0)
    depth_price_columns = [
        f"{side}Price{level}"
        for side in ("Bid", "Ask")
        for level in range(1, 6)
    ]
    raw = raw.with_columns(
        [
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit(None))
            .otherwise(pl.col(column))
            .alias(column)
            for column in depth_price_columns
        ]
    )

    second = create_second_level_snapshots(raw)

    assert second.height == 1
    assert not second["timestamp"].eq(dropped_timestamp).any()


def test_limit_down_single_sided_book_is_allowed():
    raw = pl.read_csv(SAMPLE_PATH).head(1)
    lower_limit = raw.item(0, "LowerLimitPrice")
    raw = raw.with_columns(
        [pl.lit(lower_limit).alias("LastPrice")]
        + [
            pl.lit(None).alias(f"BidPrice{level}")
            for level in range(1, 6)
        ]
        + [
            pl.lit(0).alias(f"BidVolume{level}")
            for level in range(1, 6)
        ]
    )

    validate_best_quotes(raw, "fu2302")


def test_touched_limit_down_single_sided_book_is_allowed():
    raw = pl.read_csv(SAMPLE_PATH).head(1)
    lower_limit = raw.item(0, "LowerLimitPrice")
    raw = raw.with_columns(
        [
            pl.lit(lower_limit + 1).alias("LastPrice"),
            pl.lit(lower_limit).alias("LowPrice"),
        ]
        + [
            pl.lit(None).alias(f"BidPrice{level}")
            for level in range(1, 6)
        ]
        + [
            pl.lit(0).alias(f"BidVolume{level}")
            for level in range(1, 6)
        ]
    )

    validate_best_quotes(raw, "fu2302")


def test_limit_up_single_sided_book_is_allowed():
    raw = pl.read_csv(SAMPLE_PATH).head(1)
    upper_limit = raw.item(0, "UpperLimitPrice")
    raw = raw.with_columns(
        [pl.lit(upper_limit).alias("LastPrice")]
        + [
            pl.lit(None).alias(f"AskPrice{level}")
            for level in range(1, 6)
        ]
        + [
            pl.lit(0).alias(f"AskVolume{level}")
            for level in range(1, 6)
        ]
    )

    validate_best_quotes(raw, "fu2302")


def test_touched_limit_up_single_sided_book_is_allowed():
    raw = pl.read_csv(SAMPLE_PATH).head(1)
    upper_limit = raw.item(0, "UpperLimitPrice")
    raw = raw.with_columns(
        [
            pl.lit(upper_limit - 1).alias("LastPrice"),
            pl.lit(upper_limit).alias("HighPrice"),
        ]
        + [
            pl.lit(None).alias(f"AskPrice{level}")
            for level in range(1, 6)
        ]
        + [
            pl.lit(0).alias(f"AskVolume{level}")
            for level in range(1, 6)
        ]
    )

    validate_best_quotes(raw, "fu2302")


def test_non_limit_single_sided_book_still_fails():
    raw = pl.read_csv(SAMPLE_PATH).head(1).with_columns(
        [
            pl.lit(None).alias(f"BidPrice{level}")
            for level in range(1, 6)
        ]
        + [
            pl.lit(0).alias(f"BidVolume{level}")
            for level in range(1, 6)
        ]
    )

    with pytest.raises(ValueError, match="BidPrice1 is null"):
        validate_best_quotes(raw, "fu2302")


def test_second_level_uses_last_snapshot_per_second():
    raw = pl.read_csv(SAMPLE_PATH).head(4)
    update_time = raw.item(1, "UpdateTime")
    raw = raw.with_columns(
        pl.when(pl.int_range(pl.len()) == 2)
        .then(pl.lit(update_time))
        .otherwise(pl.col("UpdateTime"))
        .alias("UpdateTime"),
        pl.when(pl.int_range(pl.len()) == 2)
        .then(pl.lit(2600.0))
        .otherwise(pl.col("BidPrice1"))
        .alias("BidPrice1"),
    )

    second = create_second_level_snapshots(raw)

    timestamp = datetime.strptime(
        f"{raw.item(2, 'ActionDay')} {raw.item(2, 'UpdateTime')}",
        "%Y%m%d %H:%M:%S.%f",
    ).replace(microsecond=0)
    assert (
        second.filter(pl.col("timestamp") == timestamp).item(0, "BidPrice1")
        == 2600.0
    )


def test_derivative_reference_falls_back_to_midprice_for_invalid_lastprice():
    raw = pl.read_csv(SAMPLE_PATH).head(3)
    second = create_second_level_snapshots(raw)
    first_timestamp = second.item(0, "timestamp")
    second = second.with_columns(
        pl.when(pl.col("timestamp") == first_timestamp)
        .then(pl.lit(0))
        .otherwise(pl.col("LastPrice"))
        .alias("LastPrice")
    )

    derivative = downscale_derivative_reference(second, "5min", "fu")

    expected_mid = (second.item(0, "BidPrice1") + second.item(0, "AskPrice1")) / 2
    assert derivative.item(0, "mark_price") == expected_mid
    assert derivative.item(0, "funding_rate") == 0


def test_base_features_use_contract_unit_for_prices_but_keep_raw_tradeval():
    second = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 3, 9, 0, 0),
                datetime(2023, 1, 3, 9, 0, 1),
                datetime(2023, 1, 3, 9, 0, 2),
            ],
            "InstrumentID": ["fu2302", "fu2302", "fu2302"],
            "BidPrice1": [2599.0, 2599.0, 2600.0],
            "AskPrice1": [2601.0, 2601.0, 2602.0],
            "LastPrice": [2600.0, 2600.0, 2601.0],
            "Volume": [0, 1, 2],
            "Turnover": [0.0, 26000.0, 52010.0],
        }
    )

    base = downscale_base_features(second, "5min", "fu").filter(pl.col("volume") > 0)

    assert base.item(0, "open") == 2600.0
    assert base.item(0, "close") == 2601.0
    assert base.item(0, "volume") == 2
    assert base.item(0, "tradeval") == 52010.0
    assert base.item(0, "vwap") == 2600.5


def test_empty_quote_window_fails_fast():
    raw = pl.read_csv(SAMPLE_PATH).head(2)
    second = create_second_level_snapshots(raw)

    with pytest.raises(ValueError, match="no quote snapshots"):
        downscale_quote_features(second.head(0), "5min")


def test_limit_down_single_sided_quote_window_counts_as_quote():
    second = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 2, 2, 13, 50, 1),
                datetime(2026, 2, 2, 13, 52, 0),
                datetime(2026, 2, 2, 13, 54, 59),
            ],
            "LastPrice": [2679.0, 2679.0, 2679.0],
            "LowPrice": [2679.0, 2679.0, 2679.0],
            "LowerLimitPrice": [2679.0, 2679.0, 2679.0],
            "BidPrice1": [None, None, None],
            "BidVolume1": [0, 0, 0],
            "AskPrice1": [2679.0, 2679.0, 2679.0],
            "AskVolume1": [601, 900, 1383],
        }
    )

    result = downscale_quote_features(second, "5min")
    window = result.filter(pl.col("timestamp") == datetime(2026, 2, 2, 13, 55, 0))

    assert window.item(0, "nquote") == 3
    assert window.item(0, "open_bid") == 2679.0
    assert window.item(0, "close_bid") == 2679.0
    assert window.item(0, "open_bidsize") == 0
    assert window.item(0, "close_bidsize") == 0


def test_limit_up_single_sided_quote_window_counts_as_quote():
    second = pl.DataFrame(
        {
            "timestamp": [
                datetime(2026, 2, 2, 13, 50, 1),
                datetime(2026, 2, 2, 13, 52, 0),
                datetime(2026, 2, 2, 13, 54, 59),
            ],
            "LastPrice": [2905.0, 2905.0, 2905.0],
            "HighPrice": [2905.0, 2905.0, 2905.0],
            "UpperLimitPrice": [2905.0, 2905.0, 2905.0],
            "BidPrice1": [2905.0, 2905.0, 2905.0],
            "BidVolume1": [601, 900, 1383],
            "AskPrice1": [None, None, None],
            "AskVolume1": [0, 0, 0],
        }
    )

    result = downscale_quote_features(second, "5min")
    window = result.filter(pl.col("timestamp") == datetime(2026, 2, 2, 13, 55, 0))

    assert window.item(0, "nquote") == 3
    assert window.item(0, "open_ask") == 2905.0
    assert window.item(0, "close_ask") == 2905.0
    assert window.item(0, "open_asksize") == 0
    assert window.item(0, "close_asksize") == 0


def test_cross_session_quote_gap_does_not_fail():
    second = pl.DataFrame(
        {
            "timestamp": [
                datetime(2025, 10, 31, 23, 0, 0),
                datetime(2025, 11, 3, 9, 0, 0),
            ],
            "BidPrice1": [2600.0, 2601.0],
            "AskPrice1": [2602.0, 2603.0],
            "BidVolume1": [1.0, 1.0],
            "AskVolume1": [1.0, 1.0],
        }
    )

    result = downscale_quote_features(second, "5min")

    assert result["timestamp"].to_list() == [
        datetime(2025, 10, 31, 23, 0, 0),
        datetime(2025, 11, 3, 9, 0, 0),
    ]


def test_intermediate_empty_quote_window_in_same_session_fails_fast():
    second = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 3, 9, 0, 0),
                datetime(2023, 1, 3, 9, 10, 0),
            ],
            "BidPrice1": [2600.0, 2601.0],
            "AskPrice1": [2602.0, 2603.0],
            "BidVolume1": [1.0, 1.0],
            "AskVolume1": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="2023-01-03 09:05:00"):
        downscale_quote_features(second, "5min")
