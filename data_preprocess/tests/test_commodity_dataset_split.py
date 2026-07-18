import json
from pathlib import Path

import polars as pl
import pytest

from operator_futures.dataset_split.dataset_split import (
    calculate_split_boundaries,
    run_dataset_split,
)


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
