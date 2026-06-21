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
    subprocess.run(
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
        check=True,
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
    assert "run_merge_process " not in text
