from pathlib import Path
import os
import subprocess
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_stitch_main_contract_cli_outputs_daily_files(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract(
        raw_root / "燃料油" / "2026" / "01" / "20260105" / "fu2602.csv",
        "fu2602",
        "20260105",
        "20260104",
        [0, 30],
    )
    _write_contract(
        raw_root / "燃料油" / "2026" / "01" / "20260106" / "fu2602.csv",
        "fu2602",
        "20260106",
        "20260105",
        [0, 2],
    )
    _write_contract(
        raw_root / "燃料油" / "2026" / "01" / "20260106" / "fu2603.csv",
        "fu2603",
        "20260106",
        "20260105",
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
            "2026-01-07",
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

    day1 = pd.read_csv(output_dir / "2026-01-05.csv")
    day2 = pd.read_csv(output_dir / "2026-01-06.csv")
    assert day1["main_contract"].tolist() == ["fu2602", "fu2602"]
    assert day2["main_contract"].tolist() == ["fu2602", "fu2602"]
    assert not (output_dir / "fu_2026-01-05_2026-01-07.csv").exists()
    assert "source_file" in day1.columns
    assert "Starting commodity main-contract daily stitch" in result.stderr
    assert "Wrote stitched commodity main-contract daily files" in result.stderr


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


def test_downscale_continuous_cli_reads_daily_input_dir_and_skips_missing_day(
    tmp_path,
):
    input_dir = tmp_path / "continuous" / "fu"
    output_root = tmp_path / "PREPROCESS_DATASET" / "commodity-futures"
    _write_continuous_day(
        input_dir / "2026-01-05.csv", "fu2602", "20260105", "20260105"
    )
    _write_continuous_day(
        input_dir / "2026-01-07.csv", "fu2602", "20260107", "20260107"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.downscale_continuous_by_trading_day",
            "--input_dir",
            str(input_dir),
            "--start_date",
            "2026-01-05",
            "--end_date",
            "2026-01-08",
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
        capture_output=True,
        check=True,
        text=True,
    )

    assert (
        output_root / "DOWNSCALE_DERTIC" / "fu" / "5min" / "2026-01-05.feather"
    ).exists()
    assert not (
        output_root / "DOWNSCALE_DERTIC" / "fu" / "5min" / "2026-01-06.feather"
    ).exists()
    assert (
        output_root / "DOWNSCALE_DERTIC" / "fu" / "5min" / "2026-01-07.feather"
    ).exists()
    assert not (
        output_root / "DOWNSCALE_DERTIC" / "fu" / "5min" / "20260105.feather"
    ).exists()
    assert "Missing commodity continuous daily file" in result.stderr
    assert "2026-01-06" in result.stderr


def test_downscale_continuous_cli_rejects_old_input_file_argument(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.downscale_continuous_by_trading_day",
            "--input",
            str(tmp_path / "fu_2026.csv"),
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
    assert "--input_dir" in result.stderr


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
    assert "--input_dir" in text
    assert "continuous_dir" in text
    assert "${symbol}_${start_date}_${end_date}.csv" not in text
    assert "continuous_file" not in text
    assert "run_merge_process " not in text
    assert "python - <<" not in text
    assert "operator_futures.commodity.downscale_continuous_by_trading_day" in text


def test_commodity_full_process_shell_sets_pythonpath_for_operator_scripts():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/fu_full_process.sh"
    )
    text = script.read_text(encoding="utf-8")

    operator_script_calls = [
        "data_preprocess/operator_futures/cross_section/create_feature.py",
        "data_preprocess/operator_futures/scale_describe_save/scale_save.py",
        "data_preprocess/operator_futures/merge_concat/merge.py",
        "data_preprocess/operator_futures/merge_concat/concat.py",
        "data_preprocess/operator_futures/time_operator/create_feature_multi_processing.py",
        "data_preprocess/operator_futures/merge_all/merge_clean.py",
        "data_preprocess/operator_futures/feature_selection/ic_correlation.py",
    ]

    for script_path in operator_script_calls:
        assert (
            f'PYTHONPATH="${{root_path}}/data_preprocess" python -u {script_path}'
            in text
            or f'PYTHONPATH="${{root_path}}/data_preprocess" nohup python -u {script_path}'
            in text
        )


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


def test_commodity_main_script_uses_date_range_full_process_entrypoint():
    script = (
        REPO_ROOT
        / "data_preprocess/script_preprocess/future_upgraded/commodity/main.sh"
    )
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

    assert "scale_describe_save/scale_save.py" in text
    assert "--ic_choice ic" in text
