import importlib
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
