import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from operator_futures.dataset_split.dataset_split import (
    calculate_split_boundaries,
    generate_trading_calendar,
    run_dataset_split,
)
try:
    from operator_futures.feature_audit.calendar_adapter import CalendarAdapter
except ImportError:
    from data_preprocess.operator_futures.feature_audit.calendar_adapter import CalendarAdapter


def _contract(name, dates):
    return {
        "contract": name,
        "trading_days": [
            {
                "trading_day": date.replace("-", ""),
                "date": date,
                "source_file": f"/raw/{name}/{date}.csv",
                "daily_volume": 100.0,
            }
            for date in dates
        ],
    }


def _write_summary(path: Path, contracts):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"symbol": "fu", "contracts": contracts}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_scale_file(root: Path, contract: str, dates):
    output = (
        root
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "SCALE_SAVE"
        / "fu"
        / contract
        / "5min"
        / "2026-01-01-2026-04-01"
        / "df.feather"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "timestamp": [
                f"{date} 09:{index:02d}:00" for index, date in enumerate(dates)
            ],
            "trading_day": dates,
            "symbol": [contract] * len(dates),
            "feature_a": list(range(len(dates))),
            "feature_b": [float(value) + 0.5 for value in range(len(dates))],
        }
    ).with_columns(pl.col("timestamp").str.strptime(pl.Datetime))
    frame.write_ipc(output)
    return output


def _write_all_feature_file(root: Path, contract: str, dates):
    output = (
        root
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "ALL_FEATURE"
        / "fu"
        / contract
        / "10min"
        / "2023-01-01-2026-03-01.feather"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "timestamp": [
                f"{date} 09:{index:02d}:00" for index, date in enumerate(dates)
            ],
            "trading_day": dates,
            "symbol": [contract] * len(dates),
            "feature_a": list(range(len(dates))),
        }
    ).with_columns(pl.col("timestamp").str.strptime(pl.Datetime))
    frame.write_ipc(output)
    return output


def test_calculate_split_boundaries_uses_union_trading_days_5_3_2():
    summary = {
        "contracts": [
            _contract(
                "fu2601",
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                ],
            ),
            _contract(
                "fu2605",
                [
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-09",
                    "2026-01-10",
                ],
            ),
        ]
    }

    assert calculate_split_boundaries(summary) == {
        "start": "2026-01-01",
        "a": "2026-01-06",
        "b": "2026-01-09",
        "c": "2026-01-11",
    }


def test_run_dataset_split_writes_contract_and_merged_outputs_with_all_columns(
    tmp_path,
):
    summary_path = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "CONTINUOUS_RAW"
        / "fu"
        / "main_contract_summary.json"
    )
    _write_summary(
        summary_path,
        [
            _contract(
                "fu2601",
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                ],
            ),
            _contract(
                "fu2605",
                [
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-09",
                    "2026-01-10",
                ],
            ),
        ],
    )
    _write_scale_file(
        tmp_path,
        "fu2601",
        [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-04",
            "2026-01-05",
            "2026-01-06",
        ],
    )
    _write_scale_file(
        tmp_path,
        "fu2605",
        [
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
            "2026-01-09",
            "2026-01-10",
        ],
    )

    manifest = run_dataset_split(
        summary_path=summary_path,
        input_root=tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE",
        output_root=tmp_path / "dataset/5min",
        symbol="fu",
        target_freq="5min",
        start_date="2026-01-01",
        end_date="2026-04-01",
    )

    dataset_root = tmp_path / "dataset" / "5min" / "fu"
    assert (dataset_root / "dataset_split_manifest.json").exists()
    assert (dataset_root / "train" / "fu2601.feather").exists()
    assert (dataset_root / "train" / "fu2605.feather").exists()
    assert (dataset_root / "valid" / "fu2601.feather").exists()
    assert (dataset_root / "valid" / "fu2605.feather").exists()
    assert (dataset_root / "test" / "fu2605.feather").exists()
    assert (dataset_root / "train.feather").exists()
    assert (dataset_root / "valid.feather").exists()
    assert (dataset_root / "test.feather").exists()

    train = pl.read_ipc(dataset_root / "train.feather")
    valid = pl.read_ipc(dataset_root / "valid.feather")
    test = pl.read_ipc(dataset_root / "test.feather")
    assert train.columns == [
        "timestamp",
        "trading_day",
        "symbol",
        "feature_a",
        "feature_b",
    ]
    assert valid.columns == train.columns
    assert test.columns == train.columns
    assert train.height == 6
    assert valid.height == 4
    assert test.height == 2
    assert set(train.get_column("symbol").to_list()) == {"fu2601", "fu2605"}
    assert test.get_column("symbol").to_list() == ["fu2605", "fu2605"]

    assert manifest["sets"]["train"]["contracts_total_count"] == 6
    assert manifest["sets"]["valid"]["contracts_total_count"] == 4
    assert manifest["sets"]["test"]["contracts_total_count"] == 2
    assert manifest["sets"]["test"]["skipped_contracts"] == [
        {"contract": "fu2601", "reason": "no trading days in test range"}
    ]
    assert manifest["sets"]["train"]["merged_output_path"].endswith(
        "dataset/5min/fu/train.feather"
    )


def test_run_dataset_split_reads_all_feature_date_range_file(tmp_path):
    summary_path = tmp_path / "summary.json"
    dates = [f"2023-01-{day:02d}" for day in range(1, 11)]
    _write_summary(summary_path, [_contract("fu2305", dates)])
    input_file = _write_all_feature_file(tmp_path, "fu2305", dates)

    manifest = run_dataset_split(
        summary_path=summary_path,
        input_root=tmp_path / "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE",
        output_root=tmp_path / "dataset/10min",
        symbol="fu",
        target_freq="10min",
        start_date="2023-01-01",
        end_date="2026-03-01",
    )

    assert manifest["sets"]["train"]["contracts"][0]["input_path"] == str(input_file)
    assert (tmp_path / "dataset/10min/fu/train/fu2305.feather").exists()


def test_run_dataset_split_fails_when_planned_input_file_is_missing(tmp_path):
    summary_path = tmp_path / "summary.json"
    _write_summary(
        summary_path,
        [_contract("fu2601", [f"2026-01-{day:02d}" for day in range(1, 11)])],
    )

    with pytest.raises(FileNotFoundError, match="fu2601"):
        run_dataset_split(
            summary_path=summary_path,
            input_root=tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE",
            output_root=tmp_path / "dataset/5min",
            symbol="fu",
            target_freq="5min",
            start_date="2026-01-01",
            end_date="2026-04-01",
        )


def test_run_dataset_split_requires_non_empty_train_valid_test_sets(tmp_path):
    summary_path = tmp_path / "summary.json"
    _write_summary(summary_path, [_contract("fu2601", ["2026-01-01", "2026-01-02"])])

    with pytest.raises(ValueError, match="start < a < b < c"):
        run_dataset_split(
            summary_path=summary_path,
            input_root=tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE",
            output_root=tmp_path / "dataset/5min",
            symbol="fu",
            target_freq="5min",
            start_date="2026-01-01",
            end_date="2026-04-01",
        )


def test_generate_trading_calendar_dynamic_activation_day_only_and_night():
    trading_days = ["2023-01-03", "2023-01-04", "2023-01-05"]
    # 2023-01-03 has NO preceding night (holiday / early period)
    # 2023-01-04 has preceding night on 2023-01-03 21:00
    # 2023-01-05 has NO preceding night (holiday eve)
    timestamps = {
        datetime(2023, 1, 3, 9, 10),
        datetime(2023, 1, 3, 15, 0),
        datetime(2023, 1, 3, 21, 0),  # Night session for 2023-01-04
        datetime(2023, 1, 3, 22, 50),
        datetime(2023, 1, 4, 9, 10),
        datetime(2023, 1, 4, 15, 0),
        datetime(2023, 1, 5, 9, 10),
        datetime(2023, 1, 5, 15, 0),
    }
    cal = generate_trading_calendar("fu", trading_days, timestamps)
    assert cal["symbol"] == "fu"
    days = cal["trading_days"]
    assert len(days) == 3

    # 2023-01-03: day only
    assert days[0]["trading_day"] == "20230103"
    assert [s["name"] for s in days[0]["sessions"]] == ["m1", "m2", "aft"]
    assert days[0]["sessions"][0]["start"] == "2023-01-03 09:00:00"
    assert days[0]["sessions"][-1]["end"] == "2023-01-03 15:00:00"

    # 2023-01-04: night + day
    assert days[1]["trading_day"] == "20230104"
    assert [s["name"] for s in days[1]["sessions"]] == ["night", "m1", "m2", "aft"]
    assert days[1]["sessions"][0]["start"] == "2023-01-03 21:00:00"
    assert days[1]["sessions"][0]["end"] == "2023-01-03 23:00:00"
    assert days[1]["sessions"][-1]["end"] == "2023-01-04 15:00:00"

    # 2023-01-05: day only
    assert days[2]["trading_day"] == "20230105"
    assert [s["name"] for s in days[2]["sessions"]] == ["m1", "m2", "aft"]


def test_generate_trading_calendar_monday_with_friday_night():
    # 2023-01-06 is Friday, 2023-01-09 is Monday
    trading_days = ["2023-01-06", "2023-01-09"]
    timestamps = {
        datetime(2023, 1, 6, 9, 10),
        datetime(2023, 1, 6, 15, 0),
        datetime(2023, 1, 6, 21, 0),  # Friday night -> belongs to Monday 2023-01-09
        datetime(2023, 1, 6, 22, 50),
        datetime(2023, 1, 9, 9, 10),
        datetime(2023, 1, 9, 15, 0),
    }
    cal = generate_trading_calendar("fu", trading_days, timestamps)
    monday_cal = cal["trading_days"][1]
    assert monday_cal["trading_day"] == "20230109"
    assert monday_cal["sessions"][0]["name"] == "night"
    assert monday_cal["sessions"][0]["start"] == "2023-01-06 21:00:00"
    assert monday_cal["sessions"][0]["end"] == "2023-01-06 23:00:00"
    assert monday_cal["sessions"][-1]["end"] == "2023-01-09 15:00:00"


def test_run_dataset_split_creates_per_set_calendar_manifests(tmp_path):
    summary_path = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "CONTINUOUS_RAW"
        / "fu"
        / "main_contract_summary.json"
    )
    days = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"]
    _write_summary(summary_path, [_contract("fu2601", days)])
    _write_scale_file(tmp_path, "fu2601", days)

    manifest = run_dataset_split(
        summary_path=summary_path,
        input_root=tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE",
        output_root=tmp_path / "dataset/5min",
        symbol="fu",
        target_freq="5min",
        start_date="2026-01-01",
        end_date="2026-04-01",
    )

    dataset_root = tmp_path / "dataset" / "5min" / "fu"
    # Verify per-set calendar manifests exist
    train_cal_file = dataset_root / "train" / "calendar_manifest.json"
    valid_cal_file = dataset_root / "valid" / "calendar_manifest.json"
    test_cal_file = dataset_root / "test" / "calendar_manifest.json"

    assert train_cal_file.exists()
    assert valid_cal_file.exists()
    assert test_cal_file.exists()

    # Verify registered in manifest
    assert manifest["sets"]["train"]["calendar_manifest_path"] == str(train_cal_file)
    assert manifest["sets"]["valid"]["calendar_manifest_path"] == str(valid_cal_file)
    assert manifest["sets"]["test"]["calendar_manifest_path"] == str(test_cal_file)

    # Verify calendar file can be cleanly parsed by CalendarAdapter
    adapter = CalendarAdapter.from_manifest(train_cal_file)
    assert len(adapter.trading_days) > 0
    for td in adapter.trading_days:
        close_dt, ret_td = adapter.get_trading_day_close(td)
        assert close_dt is not None
        assert close_dt.strftime("%H:%M:%S") == "15:00:00"
