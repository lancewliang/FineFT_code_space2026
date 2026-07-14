import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _load_slice_model():
    datahandler_path = Path(__file__).resolve().parents[2] / "datahandler"
    sys.path.insert(0, str(datahandler_path))
    sys.modules.setdefault(
        "market_dynamics_modeling_analysis",
        types.ModuleType("market_dynamics_modeling_analysis"),
    )
    sys.modules.setdefault("label_util", types.ModuleType("label_util"))
    return importlib.import_module("slice_model")


def _load_slice_model_with_real_label_util():
    datahandler_path = Path(__file__).resolve().parents[2] / "datahandler"
    sys.path.insert(0, str(datahandler_path))
    sys.modules.setdefault(
        "market_dynamics_modeling_analysis",
        types.ModuleType("market_dynamics_modeling_analysis"),
    )
    sys.modules.pop("label_util", None)
    sys.modules.pop("slice_model", None)
    return importlib.import_module("slice_model")


def _args():
    return SimpleNamespace(
        data_path="dataset/fu/valid.feather",
        filter_strength=1,
        dynamic_number=5,
        max_length_expectation=864,
        key_indicator="mark_price",
        timestamp="index",
        tic="symbol",
        labeling_method="slope",
        min_length_limit=288,
        merging_metric="DTW_distance",
        merging_threshold=0.0003,
        merging_dynamic_constraint=1,
    )


def test_prepare_raw_data_reports_loaded_columns_and_missing_required_columns(capsys):
    module = _load_slice_model()
    model = module.Linear_Market_Dynamics_Model(_args())
    raw_data = pd.DataFrame({"symbol": ["fu2302"], "index": [0]})

    with pytest.raises(ValueError, match="missing required columns.*bid1_price"):
        model.prepare_raw_data(raw_data)

    assert "loaded columns:" in capsys.readouterr().out


def test_prepare_raw_data_preserves_datetime_timestamp_column():
    module = _load_slice_model()
    args = _args()
    args.timestamp = "timestamp"
    model = module.Linear_Market_Dynamics_Model(args)
    timestamps = pd.Series(
        pd.to_datetime(["2024-08-21 22:00:00", "2024-08-21 22:10:00"]),
        name="timestamp",
    )
    raw_data = pd.DataFrame(
        {
            "symbol": ["fu", "fu"],
            "timestamp": timestamps,
            "bid1_price": [100.0, 101.0],
        }
    )

    prepared = model.prepare_raw_data(raw_data)

    assert str(prepared["timestamp"].dtype).startswith("datetime64")
    assert prepared["timestamp"].equals(timestamps)


def test_run_writes_contract_scoped_processed_and_label_outputs(tmp_path, monkeypatch):
    module = _load_slice_model()
    valid_dir = tmp_path / "dataset" / "10min" / "fu" / "valid"
    data_path = valid_dir / "df_fu2505.feather"
    valid_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": ["fu2505"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 09:00:00",
                    "2025-01-01 09:10:00",
                    "2025-01-01 09:20:00",
                    "2025-01-01 09:30:00",
                ]
            ),
            "bid1_price": [100.0, 101.0, 102.0, 103.0],
        }
    ).to_feather(data_path)

    class FakeWorker:
        def __init__(self, data_path, *args, **kwargs):
            self.data_path = data_path
            self.data_dict = {}

        def fit(self, *args, **kwargs):
            return None

        def label(self, work_dir):
            data = pd.read_feather(self.data_path)
            if len(data) == 4:
                data["label"] = [0, 0, 1, 1]
            else:
                data["label"] = [0] * len(data)
            self.data_dict = {data["symbol"].iloc[0]: data}

    monkeypatch.setattr(module.util, "Worker", FakeWorker, raising=False)
    args = _args()
    args.data_path = str(data_path)
    args.timestamp = "timestamp"
    args.dynamic_number = 2
    model = module.Linear_Market_Dynamics_Model(args)
    model._filter_padlen = lambda: 0

    model.run()

    assert (valid_dir / "processed" / "valid_processed_fu2505.feather").exists()
    assert not (valid_dir / "valid_processed.feather").exists()
    assert (valid_dir / "fu2505" / "label_0" / "df_0.feather").exists()
    assert (valid_dir / "fu2505" / "label_1" / "df_0.feather").exists()
    assert not (valid_dir / "valid" / "label_0" / "df_0.feather").exists()
    manifest = json.loads((valid_dir / "slice_manifest.json").read_text())
    assert sorted(manifest["contracts"].keys()) == ["fu2505"]
    assert sorted(manifest["contracts"]["fu2505"]["labels"].keys()) == [
        "label_0",
        "label_1",
    ]
    assert manifest["contracts"]["fu2505"]["file_count"] == 2
    assert manifest["contracts"]["fu2505"]["total_row_count"] == 4
    assert manifest["contracts"]["fu2505"]["labels"]["label_0"]["file_count"] == 1
    assert manifest["contracts"]["fu2505"]["labels"]["label_0"]["total_row_count"] == 2
    assert manifest["contracts"]["fu2505"]["labels"]["label_0"]["files"][0][
        "output_row_count"
    ] == 2
    assert sorted(manifest["labels"].keys()) == ["label_0", "label_1"]
    assert manifest["labels"]["label_0"]["file_count"] == 1
    assert manifest["labels"]["label_0"]["total_row_count"] == 2
    assert manifest["labels"]["label_0"]["files"][0]["contract"] == "fu2505"

    second_data_path = valid_dir / "df_fu2509.feather"
    pd.DataFrame(
        {
            "symbol": ["fu2509"] * 3,
            "timestamp": pd.to_datetime(
                [
                    "2025-01-02 09:00:00",
                    "2025-01-02 09:10:00",
                    "2025-01-02 09:20:00",
                ]
            ),
            "bid1_price": [200.0, 201.0, 202.0],
        }
    ).to_feather(second_data_path)
    args.data_path = str(second_data_path)
    model = module.Linear_Market_Dynamics_Model(args)
    model._filter_padlen = lambda: 0

    model.run()

    manifest = json.loads((valid_dir / "slice_manifest.json").read_text())
    assert sorted(manifest["contracts"].keys()) == ["fu2505", "fu2509"]
    assert sorted(manifest["contracts"]["fu2509"]["labels"].keys()) == ["label_0"]
    assert manifest["contracts"]["fu2509"]["total_row_count"] == 3
    assert manifest["labels"]["label_0"]["file_count"] == 2
    assert manifest["labels"]["label_0"]["total_row_count"] == 5
    assert sorted(item["contract"] for item in manifest["labels"]["label_0"]["files"]) == [
        "fu2505",
        "fu2509",
    ]


def test_run_skips_and_records_contract_with_insufficient_rows(tmp_path):
    module = _load_slice_model_with_real_label_util()
    valid_dir = tmp_path / "dataset" / "10min" / "fu" / "valid"
    data_path = valid_dir / "df_fu2401.feather"
    stale_output = valid_dir / "fu2401" / "label_0"
    stale_output.mkdir(parents=True)
    (stale_output / "df_0.feather").write_text("stale", encoding="utf-8")
    valid_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "symbol": ["fu2401"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 09:00:00",
                    "2024-01-01 09:10:00",
                    "2024-01-01 09:20:00",
                    "2024-01-01 09:30:00",
                ]
            ),
            "bid1_price": [100.0, 101.0, 102.0, 103.0],
        }
    ).to_feather(data_path)
    args = _args()
    args.data_path = str(data_path)
    args.timestamp = "timestamp"

    module.Linear_Market_Dynamics_Model(args).run()

    manifest = json.loads((valid_dir / "slice_manifest.json").read_text())
    assert "fu2401" not in manifest["contracts"]
    assert manifest["labels"] == {}
    assert manifest["skipped_contracts"]["fu2401"]["input_row_count"] == 4
    assert "insufficient rows" in manifest["skipped_contracts"]["fu2401"]["reason"]
    assert not (valid_dir / "fu2401").exists()
    assert (valid_dir / "processed" / "valid_processed_fu2401.feather").exists()


def test_dynamic_labeler_handles_small_slope_segment_count():
    sys.modules.pop("label_util", None)
    datahandler_path = Path(__file__).resolve().parents[2] / "datahandler"
    sys.path.insert(0, str(datahandler_path))
    label_util = importlib.import_module("label_util")

    labeler = label_util.Dynamic_labeler(
        labeling_method="slope",
        dynamic_num=5,
        normalized_coef_list=[-0.1, 0.2],
        data=pd.DataFrame({"pct_return_filtered": [0.0, 0.1, -0.1]}),
        turning_points=[0, 1, 2],
    )

    assert labeler.get(-0.1) == 0
    assert 0 <= labeler.get(0.2) < 5
