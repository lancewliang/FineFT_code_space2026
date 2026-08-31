from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pandas as pd
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]


def _copy_commodity_script_tree(tmp_path: Path) -> Path:
    source_dir = (
        REPO_ROOT
        / "data_preprocess"
        / "script_preprocess"
        / "future_upgraded"
        / "commodity"
    )
    target_dir = (
        tmp_path
        / "data_preprocess"
        / "script_preprocess"
        / "future_upgraded"
        / "commodity"
    )
    target_dir.parent.mkdir(parents=True)
    shutil.copytree(source_dir, target_dir)
    (target_dir / "main.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

ROOTPATH=${ROOTPATH:-$(pwd)}
START_DATE=${START_DATE:-2025-11-03}
END_DATE=${END_DATE:-2025-11-08}
TARGET_FREQ=${TARGET_FREQ:-5min}
SYMBOL=${SYMBOL:-fu}
COMMODITY_NAME=${COMMODITY_NAME:-燃料油}
MAX_PROCESSES=${MAX_PROCESSES:-4}

source "${ROOTPATH}/data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"

LOG_DIR="${ROOTPATH}/log_futures/ticker_result/commodity"
mkdir -p "${LOG_DIR}"

run_commodity_full_process \
    "${ROOTPATH}" \
    "${START_DATE}" \
    "${END_DATE}" \
    "${TARGET_FREQ}" \
    "${SYMBOL}" \
    "${COMMODITY_NAME}" \
    "${MAX_PROCESSES}" \
    >"${LOG_DIR}/${SYMBOL}_${TARGET_FREQ}_${START_DATE}_${END_DATE}.log" 2>&1
""",
        encoding="utf-8",
    )
    return target_dir


def _write_contract(path: Path, contract: str, trading_day: str, action_day: str, volumes):
    path.parent.mkdir(parents=True, exist_ok=True)
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
                "Turnover": volume * (2600 + idx),
                "BidPrice1": 2599,
                "BidVolume1": 1,
                "AskPrice1": 2601,
                "AskVolume1": 1,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_continuous_day(
    path: Path, contract: str, trading_day: str, action_day: str
):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, volume in enumerate([0, 1, 2]):
        bid1 = 2599 + idx
        ask1 = 2601 + idx
        rows.append(
            {
                "InstrumentID": contract,
                "TradingDay": trading_day,
                "ActionDay": action_day,
                "UpdateTime": f"09:00:0{idx}.000",
                "LastPrice": 2600 + idx,
                "Volume": volume,
                "Turnover": volume * (2600 + idx) * 10,
                "OpenInterest": 1000 + idx,
                "BidPrice1": bid1,
                "BidVolume1": 10,
                "AskPrice1": ask1,
                "AskVolume1": 10,
                "BidPrice2": bid1 - 1,
                "BidVolume2": 10,
                "AskPrice2": ask1 + 1,
                "AskVolume2": 10,
                "BidPrice3": bid1 - 2,
                "BidVolume3": 10,
                "AskPrice3": ask1 + 2,
                "AskVolume3": 10,
                "BidPrice4": bid1 - 3,
                "BidVolume4": 10,
                "AskPrice4": ask1 + 3,
                "AskVolume4": 10,
                "BidPrice5": bid1 - 4,
                "BidVolume5": 10,
                "AskPrice5": ask1 + 4,
                "AskVolume5": 10,
                "HighestPrice": 3000,
                "LowestPrice": 2000,
                "UpperLimitPrice": 3000,
                "LowerLimitPrice": 2000,
                "HighPrice": 2602 + idx,
                "LowPrice": 2598 + idx,
                "main_contract": contract,
                "source_contract": contract,
                "source_file": f"{contract}.csv",
                "main_contract_trading_day": trading_day,
                "main_contract_selection_reason": "current_trading_day_fallback",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_summary(path: Path, source_file: Path, contract: str = "fu2602"):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": "fu",
        "commodity_name": "燃料油",
        "start_date": "2026-01-05",
        "end_date": "2026-01-06",
        "selection_rule": "monthly_top_2_by_sum_daily_volume_delta",
        "contracts": [
            {
                "contract": contract,
                "start_trading_day": "20260105",
                "end_trading_day": "20260105",
                "trading_day_count": 1,
                "last_trading_day": "20260105",
                "total_trading_day_count": 1,
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
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_stitch_main_contract_cli_outputs_summary_json(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    for day in range(5, 16):
        trading_day = f"202601{day:02d}"
        _write_contract(
            raw_root / "燃料油" / "2026" / "01" / trading_day / "fu2602.csv",
            "fu2602",
            trading_day,
            trading_day,
            [0, 30],
        )
        _write_contract(
            raw_root / "燃料油" / "2026" / "01" / trading_day / "fu2603.csv",
            "fu2603",
            trading_day,
            trading_day,
            [0, 5],
        )

    output_dir = tmp_path / "continuous" / "fu"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.stitch_main_contract",
            "--raw_root",
            str(raw_root),
            "--commodity_name",
            "燃料油",
            "--start_date",
            "2026-01-05",
            "--end_date",
            "2026-01-16",
            "--symbol",
            "fu",
            "--output_dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        check=True,
        text=True,
    )

    summary_path = output_dir / "main_contract_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert [item["contract"] for item in summary["contracts"]] == ["fu2602", "fu2603"]
    assert summary["contracts"][0]["selected_months"] == ["2026-01"]
    assert summary["contracts"][0]["trading_day_count"] == 1
    assert summary["contracts"][0]["trading_days"][0]["source_file"] == str(
        raw_root / "燃料油" / "2026" / "01" / "20260105" / "fu2602.csv"
    )
    assert summary["contracts"][0]["trading_days"][0]["daily_volume"] == 30.0
    assert not (output_dir / "2026-01-05.csv").exists()
    assert not (output_dir / "2026-01-06.csv").exists()
    assert not (output_dir / "fu_2026-01-05_2026-01-16.csv").exists()
    assert "Starting commodity main-contract summary build" in result.stderr
    assert "Wrote commodity main-contract summary" in result.stderr


def test_stitch_main_contract_cli_removes_legacy_daily_csv_outputs(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    for day in range(5, 16):
        trading_day = f"202601{day:02d}"
        _write_contract(
            raw_root / "燃料油" / "2026" / "01" / trading_day / "fu2602.csv",
            "fu2602",
            trading_day,
            trading_day,
            [0, 30],
        )
    output_dir = tmp_path / "continuous" / "fu"
    output_dir.mkdir(parents=True)
    legacy_daily = output_dir / "2026-01-05.csv"
    legacy_daily.write_text("old daily artifact\n", encoding="utf-8")
    unrelated = output_dir / "notes.csv"
    unrelated.write_text("keep\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.stitch_main_contract",
            "--raw_root",
            str(raw_root),
            "--commodity_name",
            "燃料油",
            "--start_date",
            "2026-01-05",
            "--end_date",
            "2026-01-16",
            "--symbol",
            "fu",
            "--output_dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    assert (output_dir / "main_contract_summary.json").exists()
    assert not legacy_daily.exists()
    assert unrelated.exists()


def test_stitch_main_contract_cli_rejects_old_output_file_argument(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.stitch_main_contract",
            "--raw_root",
            str(tmp_path),
            "--commodity_name",
            "燃料油",
            "--year",
            "2026",
            "--symbol",
            "fu",
            "--output",
            str(tmp_path / "fu_2026.csv"),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--output_dir" in result.stderr


def test_downscale_single_day_cli_accepts_output_root_alias(tmp_path):
    output_root = tmp_path / "downscale"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.downscale_single_day",
            "--input",
            str(REPO_ROOT / "docs/上海商品交易所/fu2302.csv"),
            "--output_root",
            str(output_root),
            "--symbol",
            "fu",
            "--target_freq",
            "5min",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    assert (output_root / "derivative_reference.feather").exists()
    assert (output_root / "orderbook_5.feather").exists()
    assert (output_root / "base_feature.feather").exists()
    assert (output_root / "quote_feature.feather").exists()
    base = pl.read_ipc(output_root / "base_feature.feather")
    assert "symbol" in base.columns
    assert base["symbol"].unique().to_list() == ["fu"]


def test_cross_section_create_feature_writes_csv_outputs(tmp_path):
    output_root = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"
    base_dir = output_root / "BASE_FEATURE" / "fu" / "fu2602" / "5min"
    book_dir = output_root / "DOWNSCALE_ORDERBOOK_25" / "fu" / "fu2602" / "5min"
    base_dir.mkdir(parents=True)
    book_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "timestamp": [1, 2],
            "open": [100.0, 102.0],
            "high": [105.0, 106.0],
            "low": [99.0, 101.0],
            "close": [103.0, 104.0],
            "twap": [102.0, 103.0],
            "awap": [101.0, 102.0],
            "vwap": [102.5, 103.5],
        }
    ).write_ipc(base_dir / "2026-01-05.feather")
    rows = []
    for idx in range(2):
        row = {"timestamp": idx + 1}
        for level in range(1, 6):
            row[f"ask{level}_price"] = 101.0 + idx + level
            row[f"ask{level}_size"] = 2.0 + level
            row[f"bid{level}_price"] = 100.0 + idx - level
            row[f"bid{level}_size"] = 3.0 + level
        rows.append(row)
    pl.DataFrame(rows).write_ipc(book_dir / "2026-01-05.feather")

    subprocess.run(
        [
            sys.executable,
            "data_preprocess/operator_futures/cross_section/create_feature.py",
            "--root_path",
            str(tmp_path),
            "--data_path",
            "PREPROCESS_DATASET/commodity-futures/",
            "--save_path",
            "PREPROCESS_DATASET/commodity-futures/CROSS_SECTION",
            "--symbols",
            "fu",
            "--contract",
            "fu2602",
            "--target_freq",
            "5min",
            "--date",
            "2026-01-05",
            "--orderbook_depth",
            "5",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    kline_csv = (
        output_root
        / "CROSS_SECTION"
        / "KLINE_FEATURE"
        / "fu"
        / "fu2602"
        / "5min"
        / "2026-01-05.csv"
    )
    quotes_csv = (
        output_root
        / "CROSS_SECTION"
        / "QUOTES_FEATURE"
        / "fu"
        / "fu2602"
        / "5min"
        / "2026-01-05.csv"
    )
    snapshot_csv = (
        output_root
        / "CROSS_SECTION"
        / "SNAPSHOT_FEATURE"
        / "fu"
        / "fu2602"
        / "5min"
        / "2026-01-05.csv"
    )
    assert kline_csv.exists()
    assert quotes_csv.exists()
    assert snapshot_csv.exists()
    assert "timestamp" in pl.read_csv(snapshot_csv).columns


def test_downscale_continuous_cli_reads_summary_and_writes_contract_outputs(tmp_path):
    raw_file = tmp_path / "raw" / "fu2602.csv"
    _write_continuous_day(raw_file, "fu2602", "20260105", "20260105")
    summary = tmp_path / "continuous" / "fu" / "main_contract_summary.json"
    _write_summary(summary, raw_file)
    output_root = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.downscale_continuous_by_trading_day",
            "--summary",
            str(summary),
            "--output_root",
            str(output_root),
            "--target_freq",
            "5min",
            "--symbol",
            "fu",
            "--depth",
            "5",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    assert (
        output_root
        / "BASE_FEATURE"
        / "fu"
        / "fu2602"
        / "5min"
        / "2026-01-05.feather"
    ).exists()
    base_csv = (
        output_root
        / "BASE_FEATURE"
        / "fu"
        / "fu2602"
        / "5min"
        / "2026-01-05.csv"
    )
    assert base_csv.exists()
    assert "timestamp" in pd.read_csv(base_csv).columns
    assert (
        output_root
        / "DOWNSCALE_ORDERBOOK_25"
        / "fu"
        / "fu2602"
        / "5min"
        / "2026-01-05.feather"
    ).exists()
    assert (
        output_root
        / "DOWNSCALE_ORDERBOOK_25"
        / "fu"
        / "fu2602"
        / "5min"
        / "2026-01-05.csv"
    ).exists()


def test_downscale_continuous_cli_rejects_missing_summary_source_file(tmp_path):
    summary = tmp_path / "continuous" / "fu" / "main_contract_summary.json"
    _write_summary(summary, tmp_path / "missing.csv")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.downscale_continuous_by_trading_day",
            "--summary",
            str(summary),
            "--output_root",
            str(tmp_path / "out"),
            "--target_freq",
            "5min",
            "--symbol",
            "fu",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "source_file does not exist" in result.stderr


def test_downscale_continuous_cli_rejects_non_object_summary(tmp_path):
    summary = tmp_path / "continuous" / "fu" / "main_contract_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("[]", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.downscale_continuous_by_trading_day",
            "--summary",
            str(summary),
            "--output_root",
            str(tmp_path / "out"),
            "--target_freq",
            "5min",
            "--symbol",
            "fu",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "main contract summary must be a JSON object" in result.stderr


def test_downscale_continuous_cli_filters_contract(tmp_path):
    first = tmp_path / "raw" / "fu2602.csv"
    second = tmp_path / "raw" / "fu2603.csv"
    _write_continuous_day(first, "fu2602", "20260105", "20260105")
    _write_continuous_day(second, "fu2603", "20260105", "20260105")
    summary = tmp_path / "continuous" / "fu" / "main_contract_summary.json"
    payload = {
        "symbol": "fu",
        "commodity_name": "燃料油",
        "start_date": "2026-01-05",
        "end_date": "2026-01-06",
        "selection_rule": "monthly_top_2_by_sum_daily_volume_delta",
        "contracts": [
            {
                "contract": "fu2602",
                "start_trading_day": "20260105",
                "end_trading_day": "20260105",
                "trading_day_count": 1,
                "last_trading_day": "20260105",
                "total_trading_day_count": 1,
                "selected_months": ["2026-01"],
                "trading_days": [
                    {
                        "trading_day": "20260105",
                        "date": "2026-01-05",
                        "source_file": str(first),
                        "daily_volume": 100.0,
                    }
                ],
            },
            {
                "contract": "fu2603",
                "start_trading_day": "20260105",
                "end_trading_day": "20260105",
                "trading_day_count": 1,
                "last_trading_day": "20260105",
                "total_trading_day_count": 1,
                "selected_months": ["2026-01"],
                "trading_days": [
                    {
                        "trading_day": "20260105",
                        "date": "2026-01-05",
                        "source_file": str(second),
                        "daily_volume": 100.0,
                    }
                ],
            },
        ],
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    output_root = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.downscale_continuous_by_trading_day",
            "--summary",
            str(summary),
            "--contract",
            "fu2603",
            "--output_root",
            str(output_root),
            "--target_freq",
            "5min",
            "--symbol",
            "fu",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    assert not (output_root / "BASE_FEATURE" / "fu" / "fu2602").exists()
    assert (
        output_root
        / "BASE_FEATURE"
        / "fu"
        / "fu2603"
        / "5min"
        / "2026-01-05.feather"
    ).exists()


def test_downscale_continuous_cli_ignores_missing_source_file_for_unselected_contract(
    tmp_path,
):
    selected = tmp_path / "raw" / "fu2603.csv"
    _write_continuous_day(selected, "fu2603", "20260105", "20260105")
    missing = tmp_path / "raw" / "fu2602.csv"
    summary = tmp_path / "continuous" / "fu" / "main_contract_summary.json"
    payload = {
        "symbol": "fu",
        "commodity_name": "燃料油",
        "start_date": "2026-01-05",
        "end_date": "2026-01-06",
        "selection_rule": "monthly_top_2_by_sum_daily_volume_delta",
        "contracts": [
            {
                "contract": "fu2602",
                "start_trading_day": "20260105",
                "end_trading_day": "20260105",
                "trading_day_count": 1,
                "last_trading_day": "20260105",
                "total_trading_day_count": 1,
                "selected_months": ["2026-01"],
                "trading_days": [
                    {
                        "trading_day": "20260105",
                        "date": "2026-01-05",
                        "source_file": str(missing),
                        "daily_volume": 100.0,
                    }
                ],
            },
            {
                "contract": "fu2603",
                "start_trading_day": "20260105",
                "end_trading_day": "20260105",
                "trading_day_count": 1,
                "last_trading_day": "20260105",
                "total_trading_day_count": 1,
                "selected_months": ["2026-01"],
                "trading_days": [
                    {
                        "trading_day": "20260105",
                        "date": "2026-01-05",
                        "source_file": str(selected),
                        "daily_volume": 100.0,
                    }
                ],
            },
        ],
    }
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    output_root = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.downscale_continuous_by_trading_day",
            "--summary",
            str(summary),
            "--contract",
            "fu2603",
            "--output_root",
            str(output_root),
            "--target_freq",
            "5min",
            "--symbol",
            "fu",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    assert (
        output_root
        / "BASE_FEATURE"
        / "fu"
        / "fu2603"
        / "5min"
        / "2026-01-05.feather"
    ).exists()


def test_downscale_continuous_cli_rejects_old_input_dir_argument(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.downscale_continuous_by_trading_day",
            "--input_dir",
            str(tmp_path / "continuous"),
            "--start_date",
            "2026-01-05",
            "--end_date",
            "2026-01-06",
            "--output_root",
            str(tmp_path / "out"),
            "--target_freq",
            "5min",
            "--symbol",
            "fu",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--summary" in result.stderr


def test_commodity_full_process_reads_contracts_from_summary(tmp_path):
    script_dir = _copy_commodity_script_tree(tmp_path)
    summary = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "CONTINUOUS_RAW"
        / "fu"
        / "main_contract_summary.json"
    )
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "symbol": "fu",
                "commodity_name": "燃料油",
                "start_date": "2026-01-05",
                "end_date": "2026-02-06",
                "selection_rule": "monthly_top_2_by_sum_daily_volume_delta",
                "contracts": [
                    {
                        "contract": "fu2601",
                        "start_trading_day": "20260105",
                        "end_trading_day": "20260105",
                        "trading_day_count": 1,
                        "last_trading_day": "20260105",
                        "total_trading_day_count": 1,
                        "selected_months": ["2026-01"],
                        "trading_days": [
                            {
                                "trading_day": "20260105",
                                "date": "2026-01-05",
                                "source_file": "a.csv",
                                "daily_volume": 100.0,
                            }
                        ],
                    },
                    {
                        "contract": "fu2605",
                        "start_trading_day": "20260205",
                        "end_trading_day": "20260205",
                        "trading_day_count": 1,
                        "last_trading_day": "20260105",
                        "total_trading_day_count": 1,
                        "selected_months": ["2026-02"],
                        "trading_days": [
                            {
                                "trading_day": "20260205",
                                "date": "2026-02-05",
                                "source_file": "b.csv",
                                "daily_volume": 100.0,
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    command = f'''
source "{script_dir / "fu_full_process.sh"}"
run_commodity_summary_contracts "{summary}"
'''

    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.splitlines() == ["fu2601", "fu2605"]


def test_commodity_full_process_shell_exposes_expected_functions():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    assert script.exists()
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert "run_commodity_stitch_main_contract" in text
    assert "run_commodity_full_process" in text
    assert "run_commodity_merge_process" in text
    assert "PREPROCESS_DATASET/commodity-futures/MERGE_CONCAT" in text
    assert "--market_type commodity_futures" in text
    assert "--orderbook_depth 5" in text
    assert "--start_date" in text
    assert "--end_date" in text
    assert "--output_dir" in text
    assert "run_commodity_summary_contracts" in text
    assert "summary_path" in text
    assert "--summary" in text
    assert "--contract" in text
    assert "run_commodity_ic_candidate" not in text
    assert "run_commodity_ic_union_finalize" not in text
    assert '"ic_candidate"' not in text
    assert '"ic_union_finalize"' not in text
    assert "run_commodity_dataset_split" in text
    assert "operator_futures.dataset_split.dataset_split" in text
    assert "run_commodity_feature_union" not in text
    assert "${symbol}_${start_date}_${end_date}.csv" not in text
    assert "continuous_file" not in text
    assert "run_merge_process " not in text
    assert "operator_futures.commodity.downscale_continuous_by_trading_day" in text


def test_commodity_full_process_writes_step_logs_and_preserves_child_log_paths(
    tmp_path,
):
    script_dir = _copy_commodity_script_tree(tmp_path)
    script = script_dir / "fu_full_process.sh"
    original = script.read_text(encoding="utf-8")
    script.write_text(
        original
        + """

run_commodity_stitch_main_contract() {
    mkdir -p "${ROOTPATH}/PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu"
    python - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ["ROOTPATH"])
summary = root / "PREPROCESS_DATASET/commodity-futures/CONTINUOUS_RAW/fu/main_contract_summary.json"
summary.write_text(
    json.dumps(
        {
            "symbol": "fu",
            "commodity_name": "燃料油",
            "start_date": "2026-01-05",
            "end_date": "2026-01-07",
            "selection_rule": "monthly_top_2_by_sum_daily_volume_delta",
            "contracts": [
                {
                    "contract": "fu2601",
                    "start_trading_day": "20260105",
                    "end_trading_day": "20260105",
                    "trading_day_count": 1,
                    "last_trading_day": "20260105",
                    "total_trading_day_count": 1,
                    "selected_months": ["2026-01"],
                    "trading_days": [
                        {
                            "trading_day": "20260105",
                            "date": "2026-01-05",
                            "source_file": "a.csv",
                            "daily_volume": 100.0,
                        }
                    ],
                }
            ],
        }
    ),
    encoding="utf-8",
)
PY
    echo "stitch stdout"
}
    run_commodity_downscale_continuous_by_trading_day() { echo "downscale stderr" >&2; }
    run_commodity_daily_base_feature_process() { echo "daily base feature stdout"; }
    run_commodity_weekly_base_feature_process() { echo "weekly base feature stdout"; }
    run_commodity_cross_month_feature_process() { echo "cross month feature stdout"; }
    run_commodity_daily_mixed_frequency_feature_process() { echo "daily mixed frequency feature stdout"; }
    run_commodity_weekly_mixed_frequency_feature_process() { echo "weekly mixed frequency feature stdout"; }
    run_commodity_mixed_frequency_feature_process() { echo "mixed frequency feature stdout"; }
run_commodity_cross_section_process() {
    local target_freq=$4
    local symbol=$5
    local contract=$7
    local log_dir="log_futures/downscale/cross_section/${target_freq}/${symbol}/${contract}"
    mkdir -p "${log_dir}"
    echo "child cross-section" >"${log_dir}/2026-01-05.log"
}
run_commodity_merge_process() {
    local target_freq=$4
    local symbol=$5
    local contract=$7
    local log_dir="log_futures/merge/${target_freq}/${symbol}/${contract}"
    mkdir -p "${log_dir}"
    echo "child merge" >"${log_dir}/2026-01-05.log"
}
run_commodity_concat_process() { echo "concat stdout"; }
run_commodity_time_feature() { echo "time feature stdout"; }
run_commodity_merge_and_clean() { echo "merge clean stdout"; }
run_commodity_feature_selection() {
    local stage=$1
    local split_root=$2
    local target_freq=$3
    local symbol=$4
    echo "feature_selection:${stage}:${symbol}:${target_freq}:${split_root}"
}
run_commodity_scale_save() {
    local target_freq=$1
    local start_date=$2
    local end_date=$3
    local symbol=$4
    local root_path=$5
    local contract=${6:-}
    echo "scale_save:${symbol}:${contract}:${target_freq}:${start_date}:${end_date}:${root_path}"
}
run_commodity_dataset_split() {
    local summary_path=$1
    local target_freq=$2
    local start_date=$3
    local end_date=$4
    local symbol=$5
    echo "dataset_split:${symbol}:${target_freq}:${start_date}:${end_date}:${summary_path}"
}
run_commodity_maintenance_margin_dict() { echo "maintenance stdout"; }
""",
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "ROOTPATH": str(tmp_path),
        "START_DATE": "2026-01-05",
        "END_DATE": "2026-01-07",
        "TARGET_FREQ": "5min",
        "SYMBOL": "fu",
        "COMMODITY_NAME": "燃料油",
        "MAX_PROCESSES": "1",
        "PYTHONPATH": str(REPO_ROOT / "data_preprocess"),
    }
    subprocess.run(
        ["bash", str(script_dir / "main.sh")],
        cwd=tmp_path,
        env=env,
        check=True,
    )

    total_log = (
        tmp_path
        / "log_futures"
        / "ticker_result"
        / "commodity"
        / "fu_5min_2026-01-05_2026-01-07.log"
    )
    assert total_log.exists()
    text = total_log.read_text(encoding="utf-8")
    symbol_by_step = {
        "stitch_main_contract": "fu",
        "downscale_continuous_by_trading_day": "fu",
        "cross_section": "fu_fu2601",
        "merge": "fu_fu2601",
        "concat": "fu_fu2601",
        "time_feature": "fu_fu2601",
        "merge_clean": "fu_fu2601",
        "dataset_split": "fu",
        "feature_selection_train": "fu",
        "feature_selection_valid": "fu",
        "scale_save": "fu",
        "maintenance_margin_dict": "fu",
    }
    for step_name, step_symbol in symbol_by_step.items():
        step_log = (
            tmp_path
            / "log_futures"
            / "ticker_result"
            / "commodity"
            / "steps"
            / f"{step_symbol}_5min_2026-01-05_2026-01-07_{step_name}.log"
        )
        assert f"[commodity][{step_name}] start -> {step_log}" in text
        assert f"[commodity][{step_name}] success -> {step_log}" in text
        assert step_log.exists()

    downscale_log = (
        tmp_path
        / "log_futures"
        / "ticker_result"
        / "commodity"
        / "steps"
        / "fu_5min_2026-01-05_2026-01-07_downscale_continuous_by_trading_day.log"
    )
    assert "downscale stderr" in downscale_log.read_text(encoding="utf-8")
    dataset_split_log = (
        tmp_path
        / "log_futures"
        / "ticker_result"
        / "commodity"
        / "steps"
        / "fu_5min_2026-01-05_2026-01-07_dataset_split.log"
    )
    assert "dataset_split:fu:5min:2026-01-05:2026-01-07:" in (
        dataset_split_log.read_text(encoding="utf-8")
    )
    feature_train_log = (
        tmp_path
        / "log_futures"
        / "ticker_result"
        / "commodity"
        / "steps"
        / "fu_5min_2026-01-05_2026-01-07_feature_selection_train.log"
    )
    feature_valid_log = (
        tmp_path
        / "log_futures"
        / "ticker_result"
        / "commodity"
        / "steps"
        / "fu_5min_2026-01-05_2026-01-07_feature_selection_valid.log"
    )
    scale_log = (
        tmp_path
        / "log_futures"
        / "ticker_result"
        / "commodity"
        / "steps"
        / "fu_5min_2026-01-05_2026-01-07_scale_save.log"
    )
    assert "feature_selection:train:fu:5min:" in feature_train_log.read_text(
        encoding="utf-8"
    )
    assert "feature_selection:valid:fu:5min:" in feature_valid_log.read_text(
        encoding="utf-8"
    )
    assert "scale_save:fu::5min:2026-01-05:2026-01-07:" in (
        scale_log.read_text(encoding="utf-8")
    )
    assert (
        tmp_path / "log_futures/downscale/cross_section/5min/fu/fu2601/2026-01-05.log"
    ).exists()
    assert (tmp_path / "log_futures/merge/5min/fu/fu2601/2026-01-05.log").exists()


def test_commodity_full_process_step_logging_fails_fast(tmp_path):
    script_dir = _copy_commodity_script_tree(tmp_path)
    script = script_dir / "fu_full_process.sh"
    original = script.read_text(encoding="utf-8")
    script.write_text(
        original
        + """

run_commodity_stitch_main_contract() { echo "stitch ok"; }
run_commodity_downscale_continuous_by_trading_day() { echo "downscale failed" >&2; return 7; }
run_commodity_cross_section_process() { echo "unexpected cross section"; }
run_commodity_merge_process() { echo "unexpected merge"; }
run_commodity_concat_process() { echo "unexpected concat"; }
run_commodity_time_feature() { echo "unexpected time"; }
run_commodity_merge_and_clean() { echo "unexpected clean"; }
run_commodity_ic_correlation() { echo "unexpected ic"; }
run_commodity_scale_save() { echo "unexpected scale"; }
""",
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "ROOTPATH": str(tmp_path),
        "START_DATE": "2026-01-05",
        "END_DATE": "2026-01-07",
        "TARGET_FREQ": "5min",
        "SYMBOL": "fu",
        "COMMODITY_NAME": "燃料油",
        "MAX_PROCESSES": "1",
    }
    result = subprocess.run(
        ["bash", str(script_dir / "main.sh")],
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode == 7
    total_log = (
        tmp_path
        / "log_futures"
        / "ticker_result"
        / "commodity"
        / "fu_5min_2026-01-05_2026-01-07.log"
    )
    text = total_log.read_text(encoding="utf-8")
    failed_step_log = (
        tmp_path
        / "log_futures"
        / "ticker_result"
        / "commodity"
        / "steps"
        / "fu_5min_2026-01-05_2026-01-07_downscale_continuous_by_trading_day.log"
    )
    assert (
        f"[commodity][downscale_continuous_by_trading_day] failed(7) -> {failed_step_log}"
        in text
    )
    assert "unexpected cross section" not in text
    assert "downscale failed" in failed_step_log.read_text(encoding="utf-8")


def test_commodity_cross_section_process_propagates_background_failures(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "${CALL_LOG}"\n'
        'printf "%s\\n" "background failed" >&2\n'
        "exit 6\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    output_root = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"
    base_dir = output_root / "BASE_FEATURE" / "fu" / "5min"
    book_dir = output_root / "DOWNSCALE_ORDERBOOK_25" / "fu" / "5min"
    base_dir.mkdir(parents=True)
    book_dir.mkdir(parents=True)
    (base_dir / "2026-01-05.feather").write_text("", encoding="utf-8")
    (book_dir / "2026-01-05.feather").write_text("", encoding="utf-8")

    call_log = tmp_path / "calls.log"
    command = f"""
set -euo pipefail
export PATH="{fake_bin}:$PATH"
export CALL_LOG="{call_log}"
source "{REPO_ROOT}/data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
cd "{tmp_path}"
run_commodity_cross_section_process "2026-01-05" "2026-01-06" 1 "5min" "fu" "{tmp_path}"
"""

    result = subprocess.run(["bash", "-c", command], cwd=REPO_ROOT)

    assert result.returncode == 6
    assert "--date 2026-01-05" in call_log.read_text(encoding="utf-8")


def test_commodity_full_process_shell_sets_pythonpath_for_operator_scripts():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    text = script.read_text(encoding="utf-8")

    operator_script_calls = [
        "data_preprocess/operator_futures/cross_section/create_feature.py",
        "data_preprocess/operator_futures/scale_describe_save/muti_contract_scale_save.py",
        "data_preprocess/operator_futures/merge_concat/merge.py",
        "data_preprocess/operator_futures/merge_concat/concat.py",
        "data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py",
        "data_preprocess/operator_futures/merge_all/merge_clean.py",
    ]

    for script_path in operator_script_calls:
        assert (
            f'PYTHONPATH="${{root_path}}/data_preprocess" python -u {script_path}'
            in text
            or f'PYTHONPATH="${{root_path}}/data_preprocess" nohup python -u {script_path}'
            in text
        )

    assert 'PYTHONPATH="${root_path}/data_preprocess' in text
    assert "python -u -m operator_futures.dataset_split.dataset_split" in text


def test_commodity_cross_section_shell_skips_missing_downscale_outputs(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "${CALL_LOG}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    output_root = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"
    base_dir = output_root / "BASE_FEATURE" / "fu" / "5min"
    book_dir = output_root / "DOWNSCALE_ORDERBOOK_25" / "fu" / "5min"
    base_dir.mkdir(parents=True)
    book_dir.mkdir(parents=True)
    (base_dir / "2026-01-05.feather").write_text("", encoding="utf-8")
    (book_dir / "2026-01-05.feather").write_text("", encoding="utf-8")

    call_log = tmp_path / "calls.log"
    command = f"""
set -euo pipefail
export PATH="{fake_bin}:$PATH"
export CALL_LOG="{call_log}"
source "{REPO_ROOT}/data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
cd "{tmp_path}"
run_commodity_cross_section_process "2026-01-05" "2026-01-07" 1 "5min" "fu" "{tmp_path}"
"""

    subprocess.run(["bash", "-c", command], cwd=REPO_ROOT, check=True)

    calls = call_log.read_text(encoding="utf-8")
    assert "--date 2026-01-05" in calls
    assert "--date 2026-01-06" not in calls


def test_time_feature_multi_processing_accepts_commodity_orderbook_depth(tmp_path):
    data_dir = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "MERGE_CONCAT"
        / "CONCAT_FEATURE"
        / "fu"
        / "5min"
    )
    data_dir.mkdir(parents=True)
    rows = []
    for idx in range(20):
        row = {
            "timestamp": idx,
            "open": 2600.0 + idx,
            "high": 2601.0 + idx,
            "low": 2599.0 + idx,
            "close": 2600.5 + idx,
            "volume": 100.0 + idx,
            "tradeval": (2600.0 + idx) * (100.0 + idx),
            "open_interest": 1000.0 + idx,
            "buy_spread_oe_max": 4.0,
            "sell_spread_oe_max": 4.0,
            "wap_1": 2600.2 + idx,
            "wap_2": 2600.3 + idx,
            "buy_wap": 2600.1 + idx,
            "sell_wap": 2600.4 + idx,
            "mark_price": 2600.25 + idx,
            "buy_volume_oe": 20.0 + idx,
            "sell_volume_oe": 21.0 + idx,
            "imblance_volume_oe": 1.0,
        }
        for level in range(1, 6):
            row[f"bid{level}_price"] = 2600.0 + idx - level
            row[f"ask{level}_price"] = 2600.0 + idx + level
            row[f"bid{level}_size_n"] = 0.1 * level
            row[f"ask{level}_size_n"] = 0.1 * level
        rows.append(row)
    pd.DataFrame(rows).to_feather(data_dir / "2026-01-05-2026-01-06.feather")

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

    output = (
        tmp_path
        / "PREPROCESS_DATASET"
        / "commodity-futures"
        / "TIME_FEATURE"
        / "fu"
        / "5min"
        / "2026-01-05-2026-01-06.feather"
    )
    out = pd.read_feather(output)
    assert "bid5_price_log_return_2" in out.columns
    assert "bid6_price_log_return_2" not in out.columns


def test_commodity_main_script_wrapper_uses_date_range_full_process_entrypoint(
    tmp_path,
):
    script = _copy_commodity_script_tree(tmp_path) / "main.sh"
    assert script.exists()
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert "fu_full_process.sh" in text
    assert "run_commodity_full_process" in text
    assert "START_DATE" in text
    assert "END_DATE" in text
    assert "TARGET_FREQ" in text
    assert "COMMODITY_NAME" in text
    assert "START_DATE=${START_DATE:-2025-11-03}" in text
    assert "END_DATE=${END_DATE:-2025-11-08}" in text
    assert '"${YEAR}"' not in text


def test_commodity_full_process_shell_sets_time_feature_orderbook_depth():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert "time_operator/create_feature_multi_processing.py" in text
    assert "--orderbook_depth 5" in text


def test_commodity_full_process_shell_scales_ic_selection_output():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert "scale_describe_save/muti_contract_scale_save.py" in text
    assert "scale_describe_save/scale_save.py" not in text


def test_commodity_full_process_shell_runs_scale_after_feature_selection_valid():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert '"dataset_split"' in text
    assert '"feature_selection_train"' in text
    assert '"feature_selection_valid"' in text
    assert "SPLIT-TRAIN-VALID-TEST" in text
    assert "FEATURE_SELECTION/${target_freq}/${symbol}/train/state_features.npy" in text
    assert "--feature_list_path" in text
    assert "--contract" not in text[text.index("run_commodity_scale_save()") : text.index("run_commodity_merge_process()")]
    assert "--split_stage_scale_save" not in text
    assert "--feature_selection_stage valid" not in text
    assert '"feature_union"' not in text
    assert '"ic_candidate"' not in text
    assert '"ic_union_finalize"' not in text
    assert text.index('"merge_clean"') < text.index('"dataset_split"')
    assert text.index('"dataset_split"') < text.index('"feature_selection_train"')
    assert text.index('"feature_selection_train"') < text.index('"feature_selection_valid"')
    assert text.index('"feature_selection_valid"') < text.rindex('"scale_save"')
    assert text.rindex('"scale_save"') < text.index('"maintenance_margin_dict"')


def test_commodity_full_process_shell_passes_feature_blacklist():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert "--feature_blacklist" in text
    assert "wap_1" in text
    assert "midprice" in text
    assert "buy_volume_oe" in text
    assert "volume_buy" in text
    assert "mark_price" not in text
    assert '"ask${level}_price"' not in text
    assert '"bid${level}_size"' not in text


def test_commodity_full_process_shell_preserves_cross_month_features():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert "CROSS_MONTH_FEATURE_COLUMNS=(" in text
    assert "cm_contract_role_main" in text
    assert "cm_current_main_log_price_ratio" in text
    assert "cm_current_sub_log_price_ratio" in text
    assert "cm_main_sub_log_price_ratio" in text
    assert "cm_m1_m2_log_price_spread_velocity_10m" in text
    assert "cm_m1_m2_m3_butterfly_spread_velocity_10m" in text
    assert "PRICE_LIMIT_RATIO_FEATURE_COLUMNS=(" in text
    assert "limit_up_single_sided_ratio" in text
    assert "limit_down_single_sided_ratio" in text
    assert (
        '--mandatory_state_features "${BASE_TIME_FEATURE_COLUMNS[@]}" "${CROSS_MONTH_FEATURE_COLUMNS[@]}" "${PRICE_LIMIT_RATIO_FEATURE_COLUMNS[@]}"'
        in text
    )


def test_commodity_full_process_shell_preserves_session_boundary_features():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert "BASE_TIME_FEATURE_COLUMNS=(" in text
    assert "is_session_first_bar" in text
    assert "is_session_last_bar" in text
    assert '--passthrough_features "${BASE_TIME_FEATURE_COLUMNS[@]}"' in text
    assert '--mandatory_state_features "${BASE_TIME_FEATURE_COLUMNS[@]}"' in text


def test_commodity_full_process_shell_runs_cross_month_before_merge():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert "run_commodity_cross_month_feature_process()" in text
    assert "operator_futures.commodity.cross_month_feature" in text
    assert '"cross_month_feature"' in text
    assert text.index('"cross_section"') < text.index('"cross_month_feature"')
    assert text.index('"cross_month_feature"') < text.index('"merge"')
    full_process = text[text.index("run_commodity_full_process()") :]
    assert full_process.count('done < <(run_commodity_summary_contracts "$summary_path")') >= 2
    first_contract_loop_end = full_process.index(
        'done < <(run_commodity_summary_contracts "$summary_path")'
    )
    first_contract_loop = full_process[:first_contract_loop_end]
    assert '"cross_section"' in first_contract_loop
    assert '"cross_month_feature"' not in first_contract_loop
    assert '"merge"' not in first_contract_loop


def test_commodity_full_process_shell_runs_mixed_frequency_before_merge():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert "MIXED_FREQUENCY_FEATURE_COLUMNS=(" in text
    assert "prev_day_return" in text
    assert "prev_week_return" in text
    assert "run_commodity_daily_base_feature_process()" in text
    assert "run_commodity_weekly_base_feature_process()" in text
    assert "run_commodity_daily_mixed_frequency_feature_process()" in text
    assert "run_commodity_weekly_mixed_frequency_feature_process()" in text
    assert "run_commodity_mixed_frequency_feature_process()" in text
    assert "operator_futures.commodity.daily_base_feature" in text
    assert "operator_futures.commodity.weekly_base_feature" in text
    assert "operator_futures.commodity.daily_mixed_frequency_feature" in text
    assert "operator_futures.commodity.weekly_mixed_frequency_feature" in text
    assert "operator_futures.commodity.mixed_frequency_feature" in text
    assert "MIXED_FREQUENCY_BASE" in text
    assert "MIXED_FREQUENCY_FEATURE" in text
    assert '"daily_base_feature"' in text
    assert '"weekly_base_feature"' in text
    assert "--require_mixed_frequency_feature" in text
    assert '"daily_mixed_frequency_feature"' in text
    assert '"weekly_mixed_frequency_feature"' in text
    assert '"mixed_frequency_feature"' in text
    assert text.index('"daily_base_feature"') < text.index('"weekly_base_feature"')
    assert text.index('"weekly_base_feature"') < text.index('"cross_month_feature"')
    assert text.index('"weekly_base_feature"') < text.index('"daily_mixed_frequency_feature"')
    assert text.index('"daily_mixed_frequency_feature"') < text.index('"weekly_mixed_frequency_feature"')
    assert text.index('"weekly_mixed_frequency_feature"') < text.index('"mixed_frequency_feature"')
    assert text.index('"cross_month_feature"') < text.index('"mixed_frequency_feature"')
    assert text.index('"mixed_frequency_feature"') < text.index('"merge"')


def test_validate_features_checks_feature_union_outputs():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/validate_features.sh"
    )
    text = script.read_text(encoding="utf-8")

    assert "FEATURE_UNION" in text
    assert "feature_union_manifest.json" in text
    assert "state_features.npy" in text
