from pathlib import Path
import os
import subprocess
import sys

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_concat_feature_fixture(path: Path, depth: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(20):
        row = {
            "timestamp": idx,
            "open": 2600.0 + idx,
            "high": 2601.0 + idx,
            "low": 2599.0 + idx,
            "close": 2600.5 + idx,
            "volume": 100.0 + idx,
            "mark_price": 2600.25 + idx,
            "buy_spread_oe_max": 4.0,
            "sell_spread_oe_max": 4.0,
            "wap_1": 2600.2 + idx,
            "wap_2": 2600.3 + idx,
            "buy_wap": 2600.1 + idx,
            "sell_wap": 2600.4 + idx,
            "buy_volume_oe": 20.0 + idx,
            "sell_volume_oe": 21.0 + idx,
            "imblance_volume_oe": 1.0,
        }
        for level in range(1, depth + 1):
            row[f"bid{level}_price"] = 2600.0 + idx - level
            row[f"ask{level}_price"] = 2600.0 + idx + level
            row[f"bid{level}_size_n"] = 0.1 * level
            row[f"ask{level}_size_n"] = 0.2 * level
        rows.append(row)
    pl.DataFrame(rows).write_ipc(path)


def test_time_feature_multi_processing_targets_do_not_import_pandas():
    targets = [
        REPO_ROOT
        / "data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py",
        REPO_ROOT / "data_preprocess/operator_futures/time_operator/multi_processing_util.py",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "import pandas" not in text
        assert "from pandas" not in text


def test_time_feature_cli_respects_orderbook_depth_and_output_contract(tmp_path):
    input_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/CONCAT_FEATURE/fu/5min"
        / "2026-01-05-2026-01-06.feather"
    )
    _write_concat_feature_fixture(input_file, depth=5)

    subprocess.run(
        [
            sys.executable,
            "data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py",
            "--root_path",
            str(tmp_path),
            "--data_path",
            "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT/CONCAT_FEATURE/",
            "--save_path",
            "PREPROCESS_DATASET/commodity-futures/TIME_FEATURE/",
            "--symbols",
            "fu",
            "--target_freq",
            "5min",
            "--start_date",
            "2026-01-05",
            "--end_date",
            "2026-01-06",
            "--windows",
            "2",
            "--orderbook_depth",
            "5",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    output_file = (
        tmp_path
        / "PREPROCESS_DATASET/commodity-futures/TIME_FEATURE/fu/5min"
        / "2026-01-05-2026-01-06.feather"
    )
    out = pl.read_ipc(output_file)
    assert out.columns[0] == "timestamp"
    assert out.height > 0
    assert "bid5_price_log_return_2" in out.columns
    assert "bid6_price_log_return_2" not in out.columns
