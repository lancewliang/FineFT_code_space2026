import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from FineFT.datahandler.commodity_contract_dataset import (
    build_dataset_manifest,
    calculate_split_boundaries,
    run_dataset_generation,
    write_stage_datasets,
    write_train_slices,
)


def _contract(name, dates):
    return {
        "contract": name,
        "trading_days": [
            {
                "trading_day": date.replace("-", ""),
                "date": date,
                "source_file": f"/raw/{name}/{date}.csv",
                "daily_volume": 100,
            }
            for date in dates
        ],
    }


def test_calculate_split_boundaries_uses_union_trading_days_5_3_2():
    summary = {
        "symbol": "fu",
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
        ],
    }

    boundaries = calculate_split_boundaries(
        summary, train_ratio=5, valid_ratio=3, test_ratio=2
    )

    assert boundaries == {
        "start": "2026-01-01",
        "a": "2026-01-06",
        "b": "2026-01-09",
        "c": "2026-01-11",
    }


def test_calculate_split_boundaries_requires_non_empty_sets():
    summary = {
        "symbol": "fu",
        "contracts": [_contract("fu2601", ["2026-01-01", "2026-01-02"])],
    }

    with pytest.raises(ValueError, match="start < a < b < c"):
        calculate_split_boundaries(summary, train_ratio=5, valid_ratio=3, test_ratio=2)


def test_build_dataset_manifest_records_contract_intersections_and_slice_plan(tmp_path):
    summary = {
        "symbol": "fu",
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
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-09",
                    "2026-01-10",
                ],
            ),
        ],
    }
    boundaries = {
        "start": "2026-01-01",
        "a": "2026-01-06",
        "b": "2026-01-09",
        "c": "2026-01-11",
    }

    manifest = build_dataset_manifest(
        summary=summary,
        boundaries=boundaries,
        symbol="fu",
        target_freq="5min",
        start_date="2026-01-01",
        end_date="2026-04-01",
        input_root=tmp_path / "SCALE_SAVE",
        feature_union_path=tmp_path / "FEATURE_UNION" / "state_features.npy",
        output_root=tmp_path / "dataset",
        chunk_length=2,
        early_stop=1,
    )

    assert manifest["split_ratio"] == {"train": 5, "valid": 3, "test": 2}
    assert manifest["boundaries"] == boundaries
    train_contracts = {
        item["contract"]: item for item in manifest["sets"]["train"]["contracts"]
    }
    valid_contracts = {
        item["contract"]: item for item in manifest["sets"]["valid"]["contracts"]
    }
    test_contracts = {
        item["contract"]: item for item in manifest["sets"]["test"]["contracts"]
    }
    assert train_contracts["fu2601"]["trading_days"] == [
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "2026-01-04",
        "2026-01-05",
    ]
    assert train_contracts["fu2605"]["trading_days"] == ["2026-01-05"]
    assert valid_contracts["fu2601"]["trading_days"] == ["2026-01-06"]
    assert valid_contracts["fu2605"]["trading_days"] == [
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
    ]
    assert test_contracts["fu2605"]["trading_days"] == [
        "2026-01-09",
        "2026-01-10",
    ]
    assert train_contracts["fu2601"]["output_path"].endswith(
        "dataset/fu/train/df_fu2601.feather"
    )
    assert train_contracts["fu2601"]["slice_outputs"][0]["path"].endswith(
        "dataset/fu/train/slice/df_0.feather"
    )


def test_write_stage_datasets_filters_contract_files_and_omits_legacy_files(tmp_path):
    input_file = (
        tmp_path
        / "SCALE_SAVE"
        / "fu"
        / "fu2601"
        / "5min"
        / "2026-01-01-2026-04-01"
        / "df.feather"
    )
    input_file.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-06", "2026-01-09"]
            ),
            "symbol": ["fu2601"] * 4,
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "bid1_price": [10.0, 11.0, 12.0, 13.0],
        }
    ).to_feather(input_file)
    feature_union = tmp_path / "FEATURE_UNION" / "state_features.npy"
    feature_union.parent.mkdir(parents=True)
    np.save(feature_union, np.array(["feature_a"]))
    manifest = {
        "symbol": "fu",
        "state_features_path": str(tmp_path / "dataset" / "fu" / "state_features.npy"),
        "feature_union_path": str(feature_union),
        "sets": {
            "train": {
                "contracts": [
                    {
                        "contract": "fu2601",
                        "trading_days": ["2026-01-01", "2026-01-02"],
                        "input_path": str(input_file),
                        "output_path": str(
                            tmp_path / "dataset" / "fu" / "train" / "df_fu2601.feather"
                        ),
                    }
                ],
                "skipped_contracts": [],
            },
            "valid": {
                "contracts": [
                    {
                        "contract": "fu2601",
                        "trading_days": ["2026-01-06"],
                        "input_path": str(input_file),
                        "output_path": str(
                            tmp_path / "dataset" / "fu" / "valid" / "df_fu2601.feather"
                        ),
                    }
                ],
                "skipped_contracts": [],
            },
            "test": {
                "contracts": [
                    {
                        "contract": "fu2601",
                        "trading_days": ["2026-01-09"],
                        "input_path": str(input_file),
                        "output_path": str(
                            tmp_path / "dataset" / "fu" / "test" / "df_fu2601.feather"
                        ),
                    }
                ],
                "skipped_contracts": [],
            },
        },
    }

    write_stage_datasets(manifest)

    assert pd.read_feather(
        tmp_path / "dataset" / "fu" / "train" / "df_fu2601.feather"
    )["feature_a"].tolist() == [1.0, 2.0]
    assert pd.read_feather(
        tmp_path / "dataset" / "fu" / "valid" / "df_fu2601.feather"
    )["feature_a"].tolist() == [3.0]
    assert pd.read_feather(
        tmp_path / "dataset" / "fu" / "test" / "df_fu2601.feather"
    )["feature_a"].tolist() == [4.0]
    assert np.load(
        tmp_path / "dataset" / "fu" / "state_features.npy", allow_pickle=True
    ).tolist() == ["feature_a"]
    assert not (tmp_path / "dataset" / "fu" / "train.feather").exists()
    assert not (tmp_path / "dataset" / "fu" / "valid.feather").exists()
    assert not (tmp_path / "dataset" / "fu" / "test.feather").exists()


def test_write_train_slices_uses_contiguous_indices_and_single_contract_files(tmp_path):
    train_dir = tmp_path / "dataset" / "fu" / "train"
    train_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="D"),
            "symbol": ["fu2601"] * 5,
            "feature_a": range(5),
        }
    ).to_feather(train_dir / "df_fu2601.feather")
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="D"),
            "symbol": ["fu2605"] * 3,
            "feature_a": range(10, 13),
        }
    ).to_feather(train_dir / "df_fu2605.feather")
    manifest = {
        "sets": {
            "train": {
                "contracts": [
                    {
                        "contract": "fu2601",
                        "output_path": str(train_dir / "df_fu2601.feather"),
                        "slice_outputs": [
                            {
                                "index": 0,
                                "path": str(train_dir / "slice" / "df_0.feather"),
                                "row_start": 0,
                                "row_end": 3,
                            },
                            {
                                "index": 1,
                                "path": str(train_dir / "slice" / "df_1.feather"),
                                "row_start": 2,
                                "row_end": 5,
                            },
                        ],
                    },
                    {
                        "contract": "fu2605",
                        "output_path": str(train_dir / "df_fu2605.feather"),
                        "slice_outputs": [
                            {
                                "index": 2,
                                "path": str(train_dir / "slice" / "df_2.feather"),
                                "row_start": 0,
                                "row_end": 3,
                            }
                        ],
                    },
                ]
            }
        }
    }

    write_train_slices(manifest)

    slice_paths = sorted((train_dir / "slice").glob("df_*.feather"))
    assert [path.name for path in slice_paths] == [
        "df_0.feather",
        "df_1.feather",
        "df_2.feather",
    ]
    assert pd.read_feather(slice_paths[0])["symbol"].unique().tolist() == ["fu2601"]
    assert pd.read_feather(slice_paths[2])["symbol"].unique().tolist() == ["fu2605"]


def test_commodity_contract_dataset_does_not_call_slice_model():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "FineFT"
        / "datahandler"
        / "commodity_contract_dataset.py"
    )
    text = module_path.read_text()

    assert "slice_model" not in text
    assert "build_valid_labeler" not in text
    assert "write_valid_dynamic_slices" not in text


def test_commodity_data_handler_scripts_use_contract_dataset_tool():
    root = Path(__file__).resolve().parents[3]
    for script_name, symbol in [
        ("commodity_data_handler_fu.sh", "fu"),
        ("commodity_data_handler_al.sh", "al"),
    ]:
        text = (root / "FineFT" / "script" / "data" / script_name).read_text()
        assert "commodity_contract_dataset.py" in text
        assert f"--symbol {symbol}" in text or '--symbol "${SYMBOL}"' in text
        assert 'for valid_file in "dataset/${TARGET_FREQ}/${SYMBOL}/valid"/df_*.feather' in text
        assert 'slice_model.py --data_path "${valid_file}" --timestamp timestamp' in text
        assert "preprocess_data.py --trading_pair" not in text
        assert "slice_model.py --data_path dataset/" not in text


def test_run_dataset_generation_writes_manifest_stage_files_and_train_slices(tmp_path):
    summary = {
        "symbol": "fu",
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
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-09",
                    "2026-01-10",
                ],
            )
        ],
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(__import__("json").dumps(summary), encoding="utf-8")
    input_file = (
        tmp_path
        / "SCALE_SAVE"
        / "fu"
        / "fu2601"
        / "5min"
        / "2026-01-01-2026-04-01"
        / "df.feather"
    )
    input_file.parent.mkdir(parents=True)
    timestamps = [
        pd.Timestamp(f"2026-01-{day:02d} {hour}")
        for day in range(1, 11)
        for hour in ["09:00:00", "09:10:00"]
    ]
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["fu2601"] * len(timestamps),
            "feature_a": range(len(timestamps)),
            "bid1_price": range(100, 100 + len(timestamps)),
            "mark_price": range(100, 100 + len(timestamps)),
        }
    ).to_feather(input_file)
    feature_union = tmp_path / "FEATURE_UNION" / "state_features.npy"
    feature_union.parent.mkdir(parents=True)
    np.save(feature_union, np.array(["feature_a"]))

    run_dataset_generation(
        summary_path=summary_path,
        input_root=tmp_path / "SCALE_SAVE",
        feature_union_path=feature_union,
        output_root=tmp_path / "dataset",
        symbol="fu",
        target_freq="5min",
        start_date="2026-01-01",
        end_date="2026-04-01",
        train_ratio=5,
        valid_ratio=3,
        test_ratio=2,
        chunk_length=4,
        early_stop=1,
    )

    dataset_root = tmp_path / "dataset" / "fu"
    assert (dataset_root / "dataset_manifest.json").exists()
    assert (dataset_root / "train" / "df_fu2601.feather").exists()
    assert (dataset_root / "valid" / "df_fu2601.feather").exists()
    assert (dataset_root / "test" / "df_fu2601.feather").exists()
    manifest = json.loads((dataset_root / "dataset_manifest.json").read_text())
    expected_counts = {"train": 10, "valid": 6, "test": 4}
    for set_name, expected_count in expected_counts.items():
        contract = manifest["sets"][set_name]["contracts"][0]
        output_path = Path(contract["output_path"])
        assert contract["output_row_count"] == len(pd.read_feather(output_path))
        assert contract["output_row_count"] == expected_count
        assert manifest["sets"][set_name]["contracts_total_count"] == expected_count
    assert sorted(path.name for path in (dataset_root / "train" / "slice").glob("df_*.feather")) == [
        "df_0.feather",
        "df_1.feather",
        "df_2.feather",
    ]
    slice_paths = sorted((dataset_root / "train" / "slice").glob("df_*.feather"))
    slice_row_counts = [len(pd.read_feather(path)) for path in slice_paths]
    assert slice_row_counts == [5, 5, 2]
    train_slices = manifest["sets"]["train"]["contracts"][0]["slice_outputs"]
    assert [item["output_row_count"] for item in train_slices] == slice_row_counts
    assert not (dataset_root / "valid" / "label_0").exists()
