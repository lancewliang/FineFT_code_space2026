from pathlib import Path

import pandas as pd

from operator_futures.commodity.main_contract import (
    calculate_contract_volume,
    iter_contract_files,
    normalize_timestamp,
    select_main_contract_for_day,
    stitch_main_contract_frames,
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
    return pd.DataFrame(rows)


def test_normalize_timestamp_uses_action_day():
    row = pd.Series({"ActionDay": "20230103", "UpdateTime": "21:00:00.500"})
    assert normalize_timestamp(row) == pd.Timestamp("2023-01-03 21:00:00.500")


def test_calculate_contract_volume_uses_cumulative_volume_delta():
    df = _frame("fu2302", "20230104", "20230103", [10, 12, 18])
    assert calculate_contract_volume(df) == 8


def test_iter_contract_files_scans_raw_download_layout(tmp_path):
    contract_file = (
        tmp_path
        / "data"
        / "原始下载"
        / "燃料油"
        / "2026"
        / "01"
        / "20260105"
        / "fu2602.csv"
    )
    contract_file.parent.mkdir(parents=True)
    contract_file.write_text(
        "InstrumentID,TradingDay\nfu2602,20260105\n", encoding="utf-8"
    )

    files = list(iter_contract_files(tmp_path / "data" / "原始下载", "燃料油", "2026"))

    assert files == [contract_file]


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
    assert stitched["main_contract"].tolist() == [
        "fu2302",
        "fu2302",
        "fu2303",
        "fu2303",
    ]
    assert stitched["source_contract"].tolist() == [
        "fu2302",
        "fu2302",
        "fu2303",
        "fu2303",
    ]
    assert stitched["source_file"].str.endswith(".csv").all()
    assert stitched.loc[1, "LastPrice"] == 2601
    assert stitched.loc[2, "LastPrice"] == 2600
