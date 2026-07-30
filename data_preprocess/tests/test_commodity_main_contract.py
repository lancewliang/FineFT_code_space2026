import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from operator_futures.commodity.config import get_commodity_config
from operator_futures.commodity.main_contract import (
    ContractSourceFile,
    MAIN_CONTRACT_SELECTION_RULE,
    MainContractSummary,
    MainContractSummaryContract,
    MainContractSummaryTradingDay,
    build_main_contract_summary_for_date_range,
    calculate_contract_volume,
    infer_years_for_date_range,
    iter_contract_files,
    load_main_contract_summary,
    load_contract_files_by_trading_day_for_years,
    normalize_timestamp,
    write_main_contract_summary_for_date_range,
)


def _frame(
    contract: str,
    trading_day: str,
    action_day: str,
    volumes,
    open_interests=None,
):
    if open_interests is None:
        open_interests = [100 + idx for idx, _ in enumerate(volumes)]
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
                "OpenInterest": open_interests[idx],
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
    open_interests=None,
) -> Path:
    path = root / commodity_name / year / month / trading_day / f"{contract}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame(contract, trading_day, action_day, volumes, open_interests).write_csv(path)
    return path


def test_main_contract_summary_model_serializes_to_json_contract(tmp_path):
    source_file = tmp_path / "fu2601.csv"
    summary = MainContractSummary(
        symbol="fu",
        commodity_name="燃料油",
        start_date="2026-01-05",
        end_date="2026-01-06",
        contracts=[
            MainContractSummaryContract(
                contract="fu2601",
                last_trading_day="20260115",
                total_trading_day_count=10,
                selected_months=["2026-01"],
                trading_days=[
                    MainContractSummaryTradingDay(
                        trading_day="20260105",
                        date="2026-01-05",
                        source_file=str(source_file),
                        daily_volume=100.0,
                    )
                ],
            )
        ],
    )

    assert summary.to_dict() == {
        "symbol": "fu",
        "commodity_name": "燃料油",
        "start_date": "2026-01-05",
        "end_date": "2026-01-06",
        "selection_rule": MAIN_CONTRACT_SELECTION_RULE,
        "contracts": [
            {
                "contract": "fu2601",
                "start_trading_day": "20260105",
                "end_trading_day": "20260105",
                "trading_day_count": 1,
                "last_trading_day": "20260105",
                "total_trading_day_count": 1,
                "last_trading_day": "20260115",
                "total_trading_day_count": 10,
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


def test_main_contract_summary_model_deserializes_from_json_contract(tmp_path):
    source_file = tmp_path / "fu2601.csv"
    payload = {
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
                "last_trading_day": "20260105",
                "total_trading_day_count": 1,
                "last_trading_day": "20260115",
                "total_trading_day_count": 10,
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

    summary = MainContractSummary.from_dict(payload)

    assert summary.symbol == "fu"
    assert summary.contracts[0].contract == "fu2601"
    assert summary.contracts[0].trading_days[0].source_file == str(source_file)
    assert summary.contracts[0].trading_days[0].daily_volume == 100.0
    assert summary.to_dict() == payload


def test_load_main_contract_summary_reads_model(tmp_path):
    source_file = tmp_path / "fu2601.csv"
    summary_path = tmp_path / "main_contract_summary.json"
    payload = {
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
                "last_trading_day": "20260105",
                "total_trading_day_count": 1,
                "last_trading_day": "20260115",
                "total_trading_day_count": 10,
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
    summary_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    summary = load_main_contract_summary(summary_path)

    assert isinstance(summary, MainContractSummary)
    assert summary.contracts[0].contract == "fu2601"
    assert summary.contracts[0].trading_days[0].daily_volume == 100.0


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


def test_fu_config_defines_main_contract_daily_volume_threshold():
    config = get_commodity_config("fu")

    assert config.main_contract_daily_volume_threshold == 15000


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


def test_load_contract_files_by_trading_day_for_years_returns_paths(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    first = _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260105", "fu2602", "20260104", [0, 30]
    )
    second = _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260106", "fu2603", "20260105", [0, 20]
    )

    days = load_contract_files_by_trading_day_for_years(
        raw_root, "燃料油", ["2026"]
    )

    assert isinstance(days[0].contract_files[0], ContractSourceFile)
    assert days[0].trading_day == "20260105"
    assert days[0].contract_files[0].contract == "fu2602"
    assert days[0].contract_files[0].source_file == first
    assert days[1].trading_day == "20260106"
    assert days[1].contract_files[0].contract == "fu2603"
    assert days[1].contract_files[0].source_file == second


def test_build_main_contract_summary_selects_monthly_top_two_contracts(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    for day in range(1, 12):
        trading_day = f"202601{day:02d}"
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2601",
            trading_day,
            [0, 100 + day],
        )
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2602",
            trading_day,
            [0, 80 + day],
        )
    for day in range(1, 12):
        trading_day = f"202602{day:02d}"
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "02",
            trading_day,
            "fu2603",
            trading_day,
            [0, 90 + day],
        )
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "02",
            trading_day,
            "fu2604",
            trading_day,
            [0, 70 + day],
        )

    summary = build_main_contract_summary_for_date_range(
        raw_root=raw_root,
        commodity_name="燃料油",
        start_date="2026-01-01",
        end_date="2026-03-01",
        symbol="fu",
    )

    assert isinstance(summary, MainContractSummary)
    contracts = {item.contract: item for item in summary.contracts}
    assert summary.selection_rule == MAIN_CONTRACT_SELECTION_RULE
    assert list(contracts) == ["fu2601", "fu2602", "fu2603", "fu2604"]
    assert contracts["fu2601"].selected_months == ["2026-01"]
    assert contracts["fu2602"].selected_months == ["2026-01"]
    assert contracts["fu2603"].selected_months == ["2026-02"]
    assert contracts["fu2604"].selected_months == ["2026-02"]
    assert contracts["fu2601"].start_trading_day == "20260101"
    assert contracts["fu2601"].end_trading_day == "20260101"
    assert contracts["fu2601"].trading_day_count == 1
    assert contracts["fu2603"].start_trading_day == "20260201"
    assert contracts["fu2603"].end_trading_day == "20260201"
    assert contracts["fu2603"].trading_day_count == 1


def test_build_main_contract_summary_records_daily_main_sub_other_roles(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    for day in range(1, 12):
        trading_day = f"202601{day:02d}"
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2601",
            trading_day,
            [0, 300 + day],
        )
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2602",
            trading_day,
            [0, 200 + day],
        )
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2603",
            trading_day,
            [0, 100 + day],
        )

    summary = build_main_contract_summary_for_date_range(
        raw_root, "燃料油", "2026-01-01", "2026-02-01", "fu"
    )

    assert summary.main_sub_roles["20260101"] == {
        "fu2601": "main",
        "fu2602": "sub",
        "fu2603": "other",
    }
    assert summary.to_dict()["main_sub_roles"]["20260101"] == {
        "fu2601": "main",
        "fu2602": "sub",
        "fu2603": "other",
    }


def test_build_main_contract_summary_uses_open_interest_to_break_daily_volume_ties(
    tmp_path,
):
    raw_root = tmp_path / "data" / "原始下载"
    for day in range(1, 12):
        trading_day = f"202601{day:02d}"
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2601",
            trading_day,
            [0, 100],
            [100, 100],
        )
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2602",
            trading_day,
            [0, 100],
            [200, 200],
        )
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2603",
            trading_day,
            [0, 90],
            [300, 300],
        )

    summary = build_main_contract_summary_for_date_range(
        raw_root, "燃料油", "2026-01-01", "2026-02-01", "fu"
    )

    assert summary.main_sub_roles["20260101"] == {
        "fu2602": "main",
        "fu2601": "sub",
        "fu2603": "other",
    }


def test_build_main_contract_summary_selects_contract_with_ten_high_volume_days(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    for day in range(1, 13):
        trading_day = f"202601{day:02d}"
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2601",
            trading_day,
            [0, 30000],
        )
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2602",
            trading_day,
            [0, 25000],
        )
        volume = 15001 if day <= 10 else 100
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2603",
            trading_day,
            [0, volume],
        )

    summary = build_main_contract_summary_for_date_range(
        raw_root, "燃料油", "2026-01-01", "2026-02-01", "fu"
    )

    assert isinstance(summary, MainContractSummary)
    contracts = {item.contract: item for item in summary.contracts}
    assert list(contracts) == ["fu2601", "fu2602", "fu2603"]
    assert contracts["fu2603"].selected_months == ["2026-01"]


def test_build_main_contract_summary_high_volume_rule_requires_strictly_greater_than_threshold(
    tmp_path,
):
    raw_root = tmp_path / "data" / "原始下载"
    for day in range(1, 13):
        trading_day = f"202601{day:02d}"
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2601",
            trading_day,
            [0, 30000],
        )
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2602",
            trading_day,
            [0, 25000],
        )
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2603",
            trading_day,
            [0, 15000],
        )

    summary = build_main_contract_summary_for_date_range(
        raw_root, "燃料油", "2026-01-01", "2026-02-01", "fu"
    )

    assert [item.contract for item in summary.contracts] == [
        "fu2601",
        "fu2602",
    ]


def test_build_main_contract_summary_clips_selected_contract_trading_window(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260105", "fu2601", "20260105", [0, 10]
    )
    for day in range(1, 12):
        trading_day = f"202601{day:02d}"
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2602",
            trading_day,
            [0, 300 + day],
        )
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2603",
            trading_day,
            [0, 250 + day],
        )
    for day in range(2, 16):
        trading_day = f"202602{day:02d}"
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "02",
            trading_day,
            "fu2601",
            trading_day,
            [0, 100 + day],
        )

    summary = build_main_contract_summary_for_date_range(
        raw_root, "燃料油", "2026-01-01", "2026-03-01", "fu"
    )

    assert isinstance(summary, MainContractSummary)
    contract = {item.contract: item for item in summary.contracts}["fu2601"]
    assert contract.selected_months == ["2026-02"]
    assert contract.start_trading_day == "20260202"
    assert contract.end_trading_day == "20260205"
    assert [day.trading_day for day in contract.trading_days] == [
        "20260202",
        "20260203",
        "20260204",
        "20260205",
    ]


def test_build_main_contract_summary_rejects_empty_clipped_window(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260105", "fu2602", "20260105", [0, 300]
    )
    _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260105", "fu2603", "20260105", [0, 250]
    )
    _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260105", "fu2601", "20260105", [0, 10]
    )
    for day in range(1, 11):
        trading_day = f"202602{day:02d}"
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "02",
            trading_day,
            "fu2601",
            trading_day,
            [0, 100 + day],
        )

    with pytest.raises(ValueError, match="No retained trading days for contract fu2601"):
        build_main_contract_summary_for_date_range(
            raw_root, "燃料油", "2026-01-01", "2026-03-01", "fu"
        )


def test_build_main_contract_summary_ties_sort_by_contract_name(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    for day in range(1, 12):
        trading_day = f"202601{day:02d}"
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2603",
            trading_day,
            [0, 90],
        )
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2601",
            trading_day,
            [0, 90],
        )
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2602",
            trading_day,
            [0, 90],
        )

    summary = build_main_contract_summary_for_date_range(
        raw_root, "燃料油", "2026-01-01", "2026-02-01", "fu"
    )

    assert [item.contract for item in summary.contracts] == ["fu2601", "fu2602"]


def test_build_main_contract_summary_rejects_zero_volume_candidates(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260105", "fu2601", "20260105", [10, 10]
    )

    with pytest.raises(ValueError, match="No monthly top-2 contracts found"):
        build_main_contract_summary_for_date_range(
            raw_root, "燃料油", "2026-01-01", "2026-02-01", "fu"
        )


def test_build_main_contract_summary_rejects_duplicate_contract_day(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    original = _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260105", "fu2601", "20260105", [0, 10]
    )
    duplicate = raw_root / "燃料油" / "2026" / "fu2601-copy.csv"
    _frame("fu2601", "20260105", "20260105", [0, 11]).write_csv(duplicate)

    with pytest.raises(ValueError) as excinfo:
        build_main_contract_summary_for_date_range(
            raw_root, "燃料油", "2026-01-01", "2026-02-01", "fu"
        )

    message = str(excinfo.value)
    assert "Duplicate contract data for TradingDay 20260105" in message
    assert "contract fu2601" in message
    assert str(original) in message
    assert str(duplicate) in message


def test_load_contract_frames_by_trading_day_for_years_reports_duplicate_paths(
    tmp_path,
):
    raw_root = tmp_path / "data" / "原始下载"
    first = _write_contract_file(
        raw_root, "燃料油", "2025", "12", "20260105", "fu2601", "20260105", [0, 10]
    )
    second = _write_contract_file(
        raw_root, "燃料油", "2026", "01", "20260105", "fu2601", "20260105", [0, 11]
    )

    with pytest.raises(ValueError) as excinfo:
        load_contract_files_by_trading_day_for_years(
            raw_root, "燃料油", ["2025", "2026"]
        )

    message = str(excinfo.value)
    assert "Duplicate contract data for TradingDay 20260105" in message
    assert "contract fu2601" in message
    assert str(first) in message
    assert str(second) in message


def test_build_main_contract_summary_requires_volume_column(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    path = raw_root / "燃料油" / "2026" / "01" / "20260105" / "fu2601.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame("fu2601", "20260105", "20260105", [0, 10]).drop("Volume").write_csv(path)

    with pytest.raises(ValueError, match="missing required columns: .*Volume"):
        build_main_contract_summary_for_date_range(
            raw_root, "燃料油", "2026-01-01", "2026-02-01", "fu"
        )


def test_write_main_contract_summary_for_date_range_writes_json(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    output_dir = tmp_path / "continuous" / "fu"
    for day in range(1, 12):
        trading_day = f"202601{day:02d}"
        _write_contract_file(
            raw_root,
            "燃料油",
            "2026",
            "01",
            trading_day,
            "fu2601",
            trading_day,
            [0, 10 + day],
        )

    path = write_main_contract_summary_for_date_range(
        raw_root=raw_root,
        commodity_name="燃料油",
        output_dir=output_dir,
        start_date="2026-01-01",
        end_date="2026-02-01",
        symbol="fu",
    )

    assert path == output_dir / "main_contract_summary.json"
    assert path.exists()
    expected = build_main_contract_summary_for_date_range(
        raw_root=raw_root,
        commodity_name="燃料油",
        start_date="2026-01-01",
        end_date="2026-02-01",
        symbol="fu",
    )
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary == expected.to_dict()
