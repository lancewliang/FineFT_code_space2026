from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_ic_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "timestamp": list(range(12)),
            "mark_price": [100.0 + i for i in range(12)],
            "feature_a": [float(i) for i in range(12)],
            "feature_b": [float(12 - i) for i in range(12)],
        }
    )
    frame.write_ipc(path)


def test_feature_selection_targets_do_not_import_pandas():
    targets = [
        REPO_ROOT / "data_preprocess/operator_futures/feature_selection/ic_correlation.py",
        REPO_ROOT / "data_preprocess/operator_futures/feature_selection/rank_ic_correlation.py",
        REPO_ROOT / "data_preprocess/operator_futures/feature_selection/cor_util.py",
        REPO_ROOT / "data_preprocess/operator_futures/feature_selection/lasso_linear.py",
        REPO_ROOT / "data_preprocess/operator_futures/feature_selection/catbooost.py",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "import pandas" not in text
        assert "from pandas" not in text


def test_select_feature_accepts_polars_correlation_matrix():
    from operator_futures.feature_selection.cor_util import select_feature

    corre_df = pl.DataFrame(
        {
            "feature": ["feature_a", "feature_b"],
            "feature_a": [1.0, 0.8],
            "feature_b": [0.8, 1.0],
        }
    )

    assert select_feature(corre_df=corre_df, theshold=0.5) == ["feature_a"]


def test_select_feature_matches_pandas_reference_removal_order():
    from operator_futures.feature_selection.cor_util import select_feature

    corre_df = pl.DataFrame(
        {
            "feature": ["feature_a", "feature_b", "feature_c", "feature_d"],
            "feature_a": [1.0, 0.9, 0.9, 0.1],
            "feature_b": [0.9, 1.0, 0.1, 0.1],
            "feature_c": [0.9, 0.1, 1.0, 0.1],
            "feature_d": [0.1, 0.1, 0.1, 1.0],
        }
    )

    assert select_feature(corre_df=corre_df, theshold=0.7) == [
        "feature_a",
        "feature_d",
    ]


def test_ic_correlation_matrix_uses_pairwise_nan_handling():
    from operator_futures.feature_selection.ic_correlation import (
        build_pandas_like_correlation_frame,
    )

    frame = pl.DataFrame(
        {
            "feature_a": [1.0, 2.0, float("nan"), 4.0],
            "feature_b": [2.0, 4.0, 6.0, float("nan")],
            "feature_c": [3.0, 3.0, 3.0, 3.0],
        }
    )

    result = build_pandas_like_correlation_frame(
        frame, ["feature_a", "feature_b", "feature_c"]
    )

    assert result["feature"].to_list() == ["feature_a", "feature_b", "feature_c"]
    assert abs(result["feature_b"].to_list()[0] - 1.0) < 1e-12
    assert np.isnan(result["feature_c"].to_list()[0])


def test_scale_helpers_ignore_nan_like_pandas():
    from operator_futures.scale_describe_save.scale_save import scale_mean, scale_std

    frame = pl.DataFrame({"feature_a": [1.0, float("nan"), 3.0]})

    scaled = scale_mean(scale_std(frame, 10), 10, 10)

    values = scaled["feature_a"].to_list()
    assert not np.isnan(values[0])
    assert np.isnan(values[1])
    assert not np.isnan(values[2])


def test_ic_correlation_cli_writes_expected_files(tmp_path):
    input_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/fu/5min"
        / "2026-01-05-2026-01-06.feather"
    )
    _write_ic_fixture(input_file)

    subprocess.run(
        [
            sys.executable,
            "data_preprocess/operator_futures/feature_selection/ic_correlation.py",
            "--root_path",
            str(tmp_path),
            "--data_path",
            "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/",
            "--save_path",
            "PREPROCESS_DATASET/commodity-futures/IC_RESULT/",
            "--symbols",
            "fu",
            "--target_freq",
            "5min",
            "--start_date",
            "2026-01-05",
            "--end_date",
            "2026-01-06",
            "--market_type",
            "commodity_futures",
            "--orderbook_depth",
            "5",
            "--windows_list",
            "1",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    output_dir = (
        tmp_path / "PREPROCESS_DATASET/commodity-futures/IC_RESULT/fu/5min/2026-01-05-2026-01-06"
    )
    assert (output_dir / "df.feather").exists()
    assert (output_dir / "state_features.npy").exists()
    assert (output_dir / "correlation.csv").exists()
    assert np.load(output_dir / "state_features.npy", allow_pickle=True).size >= 0


def _write_lasso_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {"timestamp": list(range(12)), "mark_price": [100.0 + i for i in range(12)]}
    for idx in range(104):
        columns[f"reward_{idx}"] = [float(i + idx) for i in range(12)]
    columns["feature_a"] = [float(i) for i in range(12)]
    columns["feature_b"] = [float(12 - i) for i in range(12)]
    pl.DataFrame(columns).write_ipc(path)


def test_lasso_linear_cli_writes_expected_files(tmp_path):
    input_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/fu/5min"
        / "2026-01-05-2026-01-06.feather"
    )
    _write_lasso_fixture(input_file)

    subprocess.run(
        [
            sys.executable,
            "data_preprocess/operator_futures/feature_selection/lasso_linear.py",
            "--root_path",
            str(tmp_path),
            "--data_path",
            "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/",
            "--save_path",
            "PREPROCESS_DATASET/commodity-futures/IC_RESULT/",
            "--symbols",
            "fu",
            "--target_freq",
            "5min",
            "--start_date",
            "2026-01-05",
            "--end_date",
            "2026-01-06",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    output_dir = (
        tmp_path / "PREPROCESS_DATASET/commodity-futures/IC_RESULT/fu/5min/2026-01-05-2026-01-06"
    )
    assert (output_dir / "df_lasso.feather").exists()
    assert (output_dir / "state_features_lasso.npy").exists()


def test_catboost_importance_frame_sorts_descending():
    from operator_futures.feature_selection.catbooost import build_feature_importance_frame

    frame = build_feature_importance_frame(
        ["feature_a", "feature_b"],
        [0.2, 0.8],
    )

    assert frame["Feature"].to_list() == ["feature_b", "feature_a"]


def _write_scale_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "timestamp": list(range(12)),
            "mark_price": [100.0 + i for i in range(12)],
            "bid1_price": [99.0 + i for i in range(12)],
            "ask1_price": [101.0 + i for i in range(12)],
            "feature_a": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
        }
    )
    frame.write_ipc(path)


def test_scale_save_cli_writes_expected_files(tmp_path):
    input_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/IC_RESULT/fu/5min"
        / "2026-01-05-2026-01-06/df.feather"
    )
    input_file.parent.mkdir(parents=True, exist_ok=True)
    _write_scale_fixture(input_file)
    np.save(
        input_file.parent / "state_features.npy",
        np.array(["feature_a"]),
    )

    subprocess.run(
        [
            sys.executable,
            "data_preprocess/operator_futures/scale_describe_save/scale_save.py",
            "--root_path",
            str(tmp_path),
            "--data_path",
            "PREPROCESS_DATASET/commodity-futures/IC_RESULT",
            "--save_path",
            "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/",
            "--symbols",
            "fu",
            "--target_freq",
            "5min",
            "--start_date",
            "2026-01-05",
            "--end_date",
            "2026-01-06",
            "--market_type",
            "commodity_futures",
            "--orderbook_depth",
            "5",
            "--ic_choice",
            "ic",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    output_dir = (
        tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/5min/2026-01-05-2026-01-06"
    )
    assert (output_dir / "df.feather").exists()
    assert (output_dir / "state_features.npy").exists()
    assert (output_dir / "df_describe.csv").exists()


def test_remove_duplicates_feature_targets_do_not_import_pandas():
    path = REPO_ROOT / "data_preprocess/operator_futures/feature_selection/remove_duplicates_feature.py"
    text = path.read_text(encoding="utf-8")
    assert "import pandas" not in text
    assert "from pandas" not in text
