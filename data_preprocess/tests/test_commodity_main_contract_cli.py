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


def test_stitch_main_contract_cli_outputs_continuous_file(tmp_path):
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

    output = tmp_path / "continuous" / "fu_2026.csv"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "operator_futures.commodity.stitch_main_contract",
            "--raw_root",
            str(raw_root),
            "--commodity_name",
            "燃料油",
            "--year",
            "2026",
            "--symbol",
            "fu",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        capture_output=True,
        check=True,
        text=True,
    )

    stitched = pd.read_csv(output)
    assert stitched["main_contract"].tolist() == ["fu2602", "fu2602", "fu2602", "fu2602"]
    assert stitched["main_contract_trading_day"].tolist() == [
        20260105,
        20260105,
        20260106,
        20260106,
    ]
    assert "source_file" in stitched.columns
    assert "Starting commodity main-contract stitch" in result.stderr
    assert "Wrote stitched commodity main-contract file" in result.stderr


def test_stitch_main_contract_cli_accepts_date_range(tmp_path):
    raw_root = tmp_path / "data" / "原始下载"
    _write_contract(
        raw_root / "燃料油" / "2023" / "12" / "20231229" / "fu2401.csv",
        "fu2401",
        "20231229",
        "20231228",
        [0, 50],
    )
    _write_contract(
        raw_root / "燃料油" / "2024" / "01" / "20240102" / "fu2401.csv",
        "fu2401",
        "20240102",
        "20240101",
        [0, 2],
    )
    _write_contract(
        raw_root / "燃料油" / "2024" / "01" / "20240102" / "fu2402.csv",
        "fu2402",
        "20240102",
        "20240101",
        [0, 20],
    )

    output = tmp_path / "continuous" / "fu_2023-12-29_2024-01-03.csv"
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
            "2023-12-29",
            "--end_date",
            "2024-01-03",
            "--symbol",
            "fu",
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "data_preprocess")},
        check=True,
    )

    stitched = pd.read_csv(output)
    assert stitched["main_contract_trading_day"].astype(str).tolist() == [
        "20231229",
        "20231229",
        "20240102",
        "20240102",
    ]
    assert stitched["main_contract"].tolist() == [
        "fu2401",
        "fu2401",
        "fu2401",
        "fu2401",
    ]


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
    assert "${symbol}_${start_date}_${end_date}.csv" in text
    assert "run_merge_process " not in text
    assert "python - <<" not in text
    assert "operator_futures.commodity.downscale_continuous_by_trading_day" in text


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
    assert '"${YEAR}"' not in text
