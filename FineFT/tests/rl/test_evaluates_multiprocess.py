import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from model.low_level import ensemble_Qnet
from RL.DiHFT.low_level.evaluate_sub_agents import evaluates


def _make_sample_df(base_price: float, length: int = 5) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=length, freq="1s")
    return pd.DataFrame({
        "timestamp": times,
        "funding_rate": [0.0001] * length,
        "funding_timestamp": times + pd.Timedelta(hours=1),
        "mark_price": [base_price + i for i in range(length)],
        "bid1_price": [base_price + i - 0.5 for i in range(length)],
        "ask1_price": [base_price + i + 0.5 for i in range(length)],
        "bid1_size": [10.0] * length,
        "ask1_size": [10.0] * length,
        "feature1": [0.1 * i for i in range(length)],
    })


def test_evaluates_empty_data_files():
    results = evaluates(
        logg_file_path="",
        data_file_paths=[],
        model_path="",
        tech_indicator_list_path="",
        maintenance_margin_ratio_dict_path="",
        transcation_cost=0.0001,
        max_holding_number=2,
        position_choices=3,
        N=1,
    )
    assert results == []


def test_evaluates_multiprocess_multiple_files(tmp_path: Path):
    log_file = str(tmp_path / "eval_log" / "eval.log")
    data_file1 = str(tmp_path / "df1.feather")
    data_file2 = str(tmp_path / "df2.feather")
    tech_file = str(tmp_path / "tech.npy")
    margin_file = str(tmp_path / "margin.npy")
    model_file = str(tmp_path / "model.pth")

    _make_sample_df(100.0).to_feather(data_file1)
    _make_sample_df(200.0).to_feather(data_file2)

    np.save(tech_file, np.array(["feature1"]))
    np.save(margin_file, {"1000000000000": [0.1, 0.0]})

    # N_ACTIONS = (position_choices - 1) * len(leverage_choices) + 1 = 3
    net = ensemble_Qnet(
        N_STATES=1,
        N_ACTIONS=3,
        hidden_nodes=16,
        TIME_INFO_DIM=2,
        ensemble_number=1,
    )
    torch.save(net.state_dict(), model_file)

    results = evaluates(
        logg_file_path=log_file,
        data_file_paths=[data_file1, data_file2],
        model_path=model_file,
        tech_indicator_list_path=tech_file,
        maintenance_margin_ratio_dict_path=margin_file,
        transcation_cost=0.0001,
        max_holding_number=2,
        position_choices=3,
        N=1,
        time_info_dim=2,
        hidden_nodes=16,
        leverage_choices=[1.0],
        order_book_depth=1,
    )

    assert len(results) == 6  # 3 initial actions * 1 sub-model * 2 files
    paths_in_results = {r["df_path"] for r in results}
    assert paths_in_results == {data_file1, data_file2}

    assert os.path.exists(log_file)
    with open(log_file, "r", encoding="utf-8") as f:
        log_content = f.read()

    assert "Starting multi-process evaluation on 2 data files" in log_content
    assert "All 2 evaluation subprocesses finished." in log_content
