import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl


def _parse_date(value):
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _date_str(value):
    return value.isoformat()


def _collect_union_dates(summary):
    dates = set()
    for contract in summary.get("contracts", []):
        for day in contract.get("trading_days", []):
            dates.add(str(day["date"]))
    return sorted(dates)


def calculate_split_boundaries(summary, train_ratio=5, valid_ratio=3, test_ratio=2):
    dates = _collect_union_dates(summary)
    total = len(dates)
    ratio_total = train_ratio + valid_ratio + test_ratio
    train_count = int(math.floor(total * train_ratio / ratio_total))
    valid_count = int(math.floor(total * valid_ratio / ratio_total))
    test_count = total - train_count - valid_count
    if train_count <= 0 or valid_count <= 0 or test_count <= 0:
        raise ValueError(
            "cannot satisfy start < a < b < c with non-empty train/valid/test sets"
        )

    start = dates[0]
    a = dates[train_count]
    b = dates[train_count + valid_count]
    c = _date_str(_parse_date(dates[-1]) + timedelta(days=1))
    if not (_parse_date(start) < _parse_date(a) < _parse_date(b) < _parse_date(c)):
        raise ValueError("cannot satisfy start < a < b < c with computed boundaries")
    return {"start": start, "a": a, "b": b, "c": c}


def _in_range(date, start, end):
    parsed = _parse_date(date)
    return _parse_date(start) <= parsed < _parse_date(end)


def _input_path(input_root, symbol, contract, target_freq, start_date, end_date):
    return (
        Path(input_root)
        / symbol
        / contract
        / target_freq
        / f"{start_date}-{end_date}"
        / "df.feather"
    )


def _contract_output_path(output_root, symbol, set_name, contract):
    return Path(output_root) / symbol / set_name / f"{contract}.feather"


def _merged_output_path(output_root, symbol, set_name):
    return Path(output_root) / symbol / f"{set_name}.feather"


def _date_expr(df):
    if "trading_day" in df.columns:
        return (
            pl.col("trading_day")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        )
    if "TradingDay" in df.columns:
        return (
            pl.col("TradingDay")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, "%Y%m%d", strict=False)
        )
    if "timestamp" not in df.columns:
        raise ValueError("dataset split input missing timestamp column")
    return pl.col("timestamp").cast(pl.Datetime).dt.date()


def _filter_days(df, trading_days, *, contract, set_name, input_path):
    day_set = set(trading_days)
    filtered = (
        df.with_columns(_date_expr(df).alias("__split_date"))
        .filter(pl.col("__split_date").cast(pl.Utf8).is_in(day_set))
        .drop("__split_date")
    )
    if filtered.is_empty():
        raise ValueError(
            "planned trading days produced empty output: "
            f"contract={contract} set={set_name} input={input_path}"
        )
    if "timestamp" in filtered.columns:
        filtered = filtered.sort("timestamp")
    return filtered


def build_manifest(
    summary,
    boundaries,
    *,
    input_root,
    output_root,
    symbol,
    target_freq,
    start_date,
    end_date,
    train_ratio,
    valid_ratio,
    test_ratio,
):
    ranges = {
        "train": (boundaries["start"], boundaries["a"]),
        "valid": (boundaries["a"], boundaries["b"]),
        "test": (boundaries["b"], boundaries["c"]),
    }
    manifest = {
        "symbol": symbol,
        "target_freq": target_freq,
        "split_ratio": {"train": train_ratio, "valid": valid_ratio, "test": test_ratio},
        "boundaries": boundaries,
        "sets": {},
    }
    for set_name, (range_start, range_end) in ranges.items():
        contracts = []
        skipped = []
        for item in summary.get("contracts", []):
            contract = item["contract"]
            days = [
                str(day["date"])
                for day in item.get("trading_days", [])
                if _in_range(day["date"], range_start, range_end)
            ]
            if not days:
                skipped.append(
                    {
                        "contract": contract,
                        "reason": f"no trading days in {set_name} range",
                    }
                )
                continue
            contracts.append(
                {
                    "contract": contract,
                    "range": [range_start, range_end],
                    "trading_days": days,
                    "input_path": str(
                        _input_path(
                            input_root,
                            symbol,
                            contract,
                            target_freq,
                            start_date,
                            end_date,
                        )
                    ),
                    "output_path": str(
                        _contract_output_path(output_root, symbol, set_name, contract)
                    ),
                }
            )
        manifest["sets"][set_name] = {
            "range": [range_start, range_end],
            "merged_output_path": str(_merged_output_path(output_root, symbol, set_name)),
            "contracts": contracts,
            "skipped_contracts": skipped,
        }
    return manifest


def write_split_outputs(manifest):
    for set_name, set_info in manifest["sets"].items():
        frames = []
        total_count = 0
        for contract_info in set_info["contracts"]:
            input_path = Path(contract_info["input_path"])
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Missing df.feather for contract {contract_info['contract']}: {input_path}"
                )
            df = pl.read_ipc(input_path)
            output_df = _filter_days(
                df,
                contract_info["trading_days"],
                contract=contract_info["contract"],
                set_name=set_name,
                input_path=input_path,
            )
            output_path = Path(contract_info["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.write_ipc(output_path)
            contract_info["output_row_count"] = output_df.height
            total_count += output_df.height
            frames.append(output_df)

        if not frames:
            raise ValueError(f"cannot write {set_name}.feather without contract outputs")
        merged = pl.concat(frames, how="vertical")
        merged_output_path = Path(set_info["merged_output_path"])
        merged_output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.write_ipc(merged_output_path)
        set_info["contracts_total_count"] = total_count
        set_info["merged_output_row_count"] = merged.height


def run_dataset_split(
    *,
    summary_path,
    input_root,
    output_root,
    symbol,
    target_freq,
    start_date,
    end_date,
    train_ratio=5,
    valid_ratio=3,
    test_ratio=2,
):
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary_path: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("contracts"):
        raise ValueError("main contract summary contains no contracts")

    boundaries = calculate_split_boundaries(
        summary,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )
    manifest = build_manifest(
        summary,
        boundaries,
        input_root=input_root,
        output_root=output_root,
        symbol=symbol,
        target_freq=target_freq,
        start_date=start_date,
        end_date=end_date,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )
    write_split_outputs(manifest)
    dataset_root = Path(output_root) / symbol
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "dataset_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", type=Path, required=True)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--target_freq", type=str, required=True)
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    parser.add_argument("--train_ratio", type=int, default=5)
    parser.add_argument("--valid_ratio", type=int, default=3)
    parser.add_argument("--test_ratio", type=int, default=2)
    return parser


def main(args=None):
    parsed = build_parser().parse_args(args)
    run_dataset_split(
        summary_path=parsed.summary_path,
        input_root=parsed.input_root,
        output_root=parsed.output_root,
        symbol=parsed.symbol,
        target_freq=parsed.target_freq,
        start_date=parsed.start_date,
        end_date=parsed.end_date,
        train_ratio=parsed.train_ratio,
        valid_ratio=parsed.valid_ratio,
        test_ratio=parsed.test_ratio,
    )


if __name__ == "__main__":
    main()
