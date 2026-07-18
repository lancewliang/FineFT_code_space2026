from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest


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


def _ic_args(tmp_path):
    return SimpleNamespace(
        root_path=str(tmp_path),
        data_path="PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/",
        save_path="PREPROCESS_DATASET/commodity-futures/IC_RESULT/",
        symbols="fu",
        contract=None,
        target_freq="5min",
        start_date="2026-01-05",
        end_date="2026-01-06",
        ic_theshold=0.01,
        cor_theshold=0.7,
        windows_list=[1],
        market_type="commodity_futures",
        orderbook_depth=5,
    )


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


def test_scale_helpers_match_reference_for_tiny_std_large_mean_adjustment():
    from operator_futures.feature_validation.pandas_reference.scale_describe_save.scale_save import (
        scale_mean as pandas_scale_mean,
        scale_std as pandas_scale_std,
    )
    from operator_futures.scale_describe_save.scale_save import scale_mean, scale_std

    pandas_frame = pd.DataFrame(
        {
            "corr_like": [
                1.0000000001,
                0.9999999999,
                1.0000000002,
                0.9999999998,
            ]
        }
    )
    polars_frame = pl.from_pandas(pandas_frame)

    expected = pandas_scale_mean(pandas_scale_std(pandas_frame, 10), 10, 10)
    actual = scale_mean(scale_std(polars_frame, 10), 10, 10).to_pandas()

    pd.testing.assert_frame_equal(actual, expected)


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
    assert (output_dir / "df.csv").exists()
    assert pl.read_csv(output_dir / "df.csv").shape == pl.read_ipc(output_dir / "df.feather").shape
    assert (output_dir / "state_features.npy").exists()
    assert (output_dir / "correlation.csv").exists()
    assert np.load(output_dir / "state_features.npy", allow_pickle=True).size >= 0


def test_ic_correlation_rejects_illegal_input_values(tmp_path):
    from operator_futures.feature_selection import ic_correlation

    input_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/ALL_FEATURE/fu/5min"
        / "2026-01-05-2026-01-06.feather"
    )
    input_file.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "timestamp": list(range(12)),
            "mark_price": [100.0 + i for i in range(12)],
            "feature_a": [
                float("inf") if i == 2 else float(i) for i in range(12)
            ],
            "feature_b": [float(12 - i) for i in range(12)],
        }
    )
    frame.write_ipc(input_file)

    with pytest.raises(ValueError) as exc_info:
        ic_correlation.main(_ic_args(tmp_path))

    message = str(exc_info.value)
    assert "stage=ic_correlation_input" in message
    assert "feature_a:infinite=1" in message


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


def _write_scale_fixture(path: Path, feature_values=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if feature_values is None:
        feature_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
    frame = pl.DataFrame(
        {
            "timestamp": list(range(12)),
            "mark_price": [100.0 + i for i in range(12)],
            "bid1_price": [99.0 + i for i in range(12)],
            "ask1_price": [101.0 + i for i in range(12)],
            "feature_a": feature_values,
        }
    )
    frame.write_ipc(path)


def _scale_input_file(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/IC_RESULT/fu/5min"
        / "2026-01-05-2026-01-06/df.feather"
    )


def _scale_output_dir(tmp_path: Path) -> Path:
    return tmp_path / "PREPROCESS_DATASET/commodity-futures/SCALE_SAVE/fu/5min/2026-01-05-2026-01-06"


def _write_scale_state_features(input_file: Path) -> None:
    np.save(
        input_file.parent / "state_features.npy",
        np.array(["feature_a"]),
    )


def _run_scale_save_cli(tmp_path: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
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
        check=check,
        text=True,
        capture_output=True,
    )


def test_scale_save_cli_writes_expected_files(tmp_path):
    input_file = _scale_input_file(tmp_path)
    _write_scale_fixture(input_file)
    _write_scale_state_features(input_file)

    _run_scale_save_cli(tmp_path)

    output_dir = _scale_output_dir(tmp_path)
    assert (output_dir / "df.feather").exists()
    assert (output_dir / "df.csv").exists()
    assert pl.read_csv(output_dir / "df.csv").shape == pl.read_ipc(output_dir / "df.feather").shape
    assert (output_dir / "state_features.npy").exists()
    assert (output_dir / "df_describe.csv").exists()
    df = pl.read_ipc(output_dir / "df.feather")
    assert "symbol" in df.columns
    assert df["symbol"].unique().to_list() == ["fu"]


def test_scale_save_cli_rejects_input_nan_before_writing_outputs(tmp_path):
    input_file = _scale_input_file(tmp_path)
    _write_scale_fixture(input_file, feature_values=[10.0, 20.0, float("nan"), 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0])
    _write_scale_state_features(input_file)

    result = _run_scale_save_cli(tmp_path, check=False)

    output_dir = _scale_output_dir(tmp_path)
    combined_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "input" in combined_output
    assert str(input_file) in combined_output
    assert "feature_a(count=1, rows=[2])" in combined_output
    assert not (output_dir / "df.feather").exists()
    assert not (output_dir / "df.csv").exists()
    assert not (output_dir / "state_features.npy").exists()
    assert not (output_dir / "df_describe.csv").exists()


def test_scale_save_cli_rejects_output_nan_before_writing_outputs(tmp_path):
    input_file = _scale_input_file(tmp_path)
    _write_scale_fixture(input_file, feature_values=[0.0 for _ in range(12)])
    _write_scale_state_features(input_file)

    result = _run_scale_save_cli(tmp_path, check=False)

    output_dir = _scale_output_dir(tmp_path)
    combined_output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "output" in combined_output
    assert str(output_dir / "df.feather") in combined_output
    assert "feature_a(count=12, rows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])" in combined_output
    assert not (output_dir / "df.feather").exists()
    assert not (output_dir / "df.csv").exists()
    assert not (output_dir / "state_features.npy").exists()
    assert not (output_dir / "df_describe.csv").exists()


def test_remove_duplicates_feature_targets_do_not_import_pandas():
    path = REPO_ROOT / "data_preprocess/operator_futures/feature_selection/remove_duplicates_feature.py"
    text = path.read_text(encoding="utf-8")
    assert "import pandas" not in text
    assert "from pandas" not in text
