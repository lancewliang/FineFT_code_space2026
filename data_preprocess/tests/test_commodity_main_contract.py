import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from operator_futures.commodity.main_contract import (
    build_main_contract_daily_frames_for_date_range,
    build_main_contract_continuous_frame_for_date_range,
    calculate_contract_volume,
    infer_years_for_date_range,
    iter_contract_files,
    normalize_timestamp,
    select_main_contract_for_day,
    stitch_main_contract_frames,
    write_main_contract_daily_files_for_date_range,
)


def _frame(contract: str, trading_day: str, action_day: str, volumes):
    rows = []
    for idx, volume in enumerate(volumes):
        rows.append(
            {
                "InstrumentID": contract,
                "TradingDay": trading_day,
                "ActionDay": action_day,
                "UpdateTime": f"21:00:0{idx}.500",
                "LastPrice": 2600 + idx,
                "Volume": volume,
                "Turnover": volume * (2600 + idx),
                "BidPrice1": 2599,
                "BidVolume1": 1,
                "AskPrice1": 2601,
                "AskVolume1": 1,
            }
        )
    return pl.DataFrame(rows)


def _write_contract_file(
    root: Path,
    commodity_name: str,
    year: str,
    month: str,
    trading_day: str,
    contract: str,
    action_day: str,
    volumes,
) -> Path:
    path = root / commodity_name / year / month / trading_day / f"{contract}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame(contract, trading_day, action_day, volumes).write_csv(path)
    return path


def test_infer_years_for_left_closed_right_open_date_range():
    assert infer_years_for_date_range("2023-01-01", "2026-03-01") == [
        "2023",
        "2024",
        "2025",
        "2026",
    ]
    assert infer_years_for_date_range("2023-01-01", "2024-01-01") == ["2023"]
    assert infer_years_for_date_range("2026-02-28", "2026-03-01") == ["2026"]


def test_normalize_timestamp_uses_action_day():
    row = {"ActionDay": "20230103", "UpdateTime": "21:00:00.500"}
    assert normalize_timestamp(row) == datetime(2023, 1, 3, 21, 0, 0, 500000)


def test_calculate_contract_volume_uses_cumulative_volume_delta():
    df = _frame("fu2302", "20230104", "20230103", [10, 12, 18])
    assert calculate_contract_volume(df) == 8


def test_iter_contract_files_scans_raw_download_layout(tmp_path):
    nested_file = (
        tmp_path
        / "data"
        / "原始下载"
        / "燃料油"
        / "2026"
        / "01"
        / "20260105"
        / "fu2602.csv"
    )
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text(
        "InstrumentID,TradingDay\nfu2602,20260105\n", encoding="utf-8"
    )
    flat_file = (
        tmp_path
        / "data"
        / "原始下载"
        / "燃料油"
        / "2026"
        / "fu2602-2026-01-06.csv"
    )
    flat_file.write_text(
        "InstrumentID,TradingDay\nfu2602,20260106\n", encoding="utf-8"
    )

    files = list(iter_contract_files(tmp_path / "data" / "原始下载", "燃料油", "2026"))

    assert files == [nested_file, flat_file]


def test_select_main_contract_uses_previous_day_volume():
    previous = {
        "fu2302": _frame("fu2302", "20230103", "20230102", [10, 12]),
        "fu2303": _frame("fu2303", "20230103", "20230102", [7, 40]),
    }
    current = {
        "fu2302": _frame("fu2302", "20230104", "20230103", [0, 1]),
        "fu2303": _frame("fu2303", "20230104", "20230103", [0, 2]),
    }
    selected, reason = select_main_contract_for_day(previous, current, "fu")
    assert selected == "fu2303"
    assert reason == "previous_trading_day_volume"


def test_select_main_contract_falls_back_to_current_day_volume():
    previous = {"fu2302": _frame("fu2302", "20230103", "20230102", [10, 30])}
    current = {
        "fu2303": _frame("fu2303", "20230104", "20230103", [0, 11]),
        "fu2304": _frame("fu2304", "20230104", "20230103", [0, 5]),
    }
    selected, reason = select_main_contract_for_day(previous, current, "fu")
    assert selected == "fu2303"
    assert reason == "current_trading_day_fallback"


def test_stitch_main_contract_frames_keeps_metadata_and_no_back_adjustment():
    day1 = _frame("fu2302", "20230104", "20230103", [0, 1])
    day2 = _frame("fu2303", "20230105", "20230104", [0, 2])
    stitched = stitch_main_contract_frames(
        [
            ("20230104", "fu2302", day1, Path("fu2302.csv")),
            ("20230105", "fu2303", day2, Path("fu2303.csv")),
        ]
    )
    assert stitched["main_contract"].to_list() == [
        "fu2302",
        "fu2302",
        "fu2303",
        "fu2303",
    ]
    assert stitched["source_contract"].to_list() == [
        "fu2302",
        "fu2302",
        "fu2303",
        "fu2303",
    ]
    assert stitched["source_file"].str.ends_with(".csv").all()
    assert stitched.item(1, "LastPrice") == 2601
    assert stitched.item(2, "LastPrice") == 2600


def test_build_main_contract_for_date_range_keeps_previous_frames_across_years(
    tmp_path,
):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract_file(
        raw_root, "燃料油", "2023", "12", "20231229", "fu2401", "20231228", [0, 50]
    )
    _write_contract_file(
        raw_root, "燃料油", "2023", "12", "20231229", "fu2402", "20231228", [0, 10]
    )
    _write_contract_file(
        raw_root, "燃料油", "2024", "01", "20240102", "fu2401", "20240101", [0, 2]
    )
    _write_contract_file(
        raw_root, "燃料油", "2024", "01", "20240102", "fu2402", "20240101", [0, 20]
    )

    stitched = build_main_contract_continuous_frame_for_date_range(
        raw_root,
        "燃料油",
        "2023-12-29",
        "2024-01-03",
        "fu",
    )

    assert stitched["main_contract_trading_day"].cast(pl.Utf8).to_list() == [
        "20231229",
        "20231229",
        "20240102",
        "20240102",
    ]
    assert stitched["main_contract"].to_list() == [
        "fu2401",
        "fu2401",
        "fu2401",
        "fu2401",
    ]
    assert stitched["main_contract_selection_reason"].to_list() == [
        "current_trading_day_fallback",
        "current_trading_day_fallback",
        "previous_trading_day_volume",
        "previous_trading_day_volume",
    ]


def test_build_main_contract_logs_selected_file_contract_and_reason(
    tmp_path,
    caplog,
):
    raw_root = tmp_path / "data" / "原始下载"
    expected_file = _write_contract_file(
        raw_root, "燃料油", "2024", "01", "20240102", "fu2402", "20240101", [0, 20]
    )
    _write_contract_file(
        raw_root, "燃料油", "2024", "01", "20240102", "fu2401", "20240101", [0, 2]
    )

    with caplog.at_level(
        logging.INFO, logger="operator_futures.commodity.main_contract"
    ):
        build_main_contract_continuous_frame_for_date_range(
            raw_root,
            "燃料油",
            "2024-01-02",
            "2024-01-03",
            "fu",
        )

    selected_logs = [
        record
        for record in caplog.records
        if record.message.startswith("Selected commodity main-contract file:")
    ]
    assert len(selected_logs) == 1
    assert "trading_day=20240102" in selected_logs[0].message
    assert "contract=fu2402" in selected_logs[0].message
    assert "reason=current_trading_day_fallback" in selected_logs[0].message
    assert "file_name=fu2402.csv" in selected_logs[0].message
    assert f"source_file={expected_file}" in selected_logs[0].message


def test_build_main_contract_daily_frames_skips_missing_calendar_dates(
    tmp_path, caplog
):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260105", "fu2602", "20260104", [0, 30]
    )
    _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260107", "fu2602", "20260106", [0, 4]
    )

    with caplog.at_level(
        logging.INFO, logger="operator_futures.commodity.main_contract"
    ):
        daily = build_main_contract_daily_frames_for_date_range(
            raw_root,
            "燃料油",
            "2026-01-05",
            "2026-01-08",
            "fu",
        )

    assert sorted(daily) == ["2026-01-05", "2026-01-07"]
    assert daily["2026-01-05"]["main_contract_trading_day"].cast(
        pl.Utf8
    ).to_list() == [
        "20260105",
        "20260105",
    ]
    assert "Skipped commodity main-contract source dates" in caplog.text
    assert "2026-01-06" in caplog.text


def test_write_main_contract_daily_files_overwrites_existing_file(tmp_path, caplog):
    raw_root = tmp_path / "data" / "原始下载"
    output_dir = (
        tmp_path / "PREPROCESS_DATASET" / "commodity-futures" / "CONTINUOUS_RAW" / "fu"
    )
    _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260105", "fu2602", "20260104", [0, 30]
    )
    output_dir.mkdir(parents=True)
    existing = output_dir / "2026-01-05.csv"
    existing.write_text("old\n", encoding="utf-8")

    with caplog.at_level(
        logging.INFO, logger="operator_futures.commodity.main_contract"
    ):
        written = write_main_contract_daily_files_for_date_range(
            raw_root=raw_root,
            commodity_name="燃料油",
            output_dir=output_dir,
            start_date="2026-01-05",
            end_date="2026-01-06",
            symbol="fu",
        )

    assert written == [existing]
    result = pl.read_csv(existing)
    assert result["main_contract"].to_list() == ["fu2602", "fu2602"]
    assert "Overwriting commodity main-contract daily file" in caplog.text
    assert str(existing) in caplog.text
