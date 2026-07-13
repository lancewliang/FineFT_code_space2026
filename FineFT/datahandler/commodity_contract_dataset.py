import argparse
import json
import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


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


def _stage_output(output_root, symbol, set_name, contract):
    return Path(output_root) / symbol / set_name / f"df_{contract}.feather"


def _build_slice_plan(
    contract_days, output_root, symbol, contract, start_index, chunk_length, early_stop
):
    outputs = []
    if not contract_days:
        return outputs, start_index

    row_start = 0
    index = start_index
    while row_start + chunk_length <= len(contract_days):
        row_end = min(row_start + chunk_length + early_stop, len(contract_days))
        outputs.append(
            {
                "index": index,
                "contract": contract,
                "path": str(
                    Path(output_root)
                    / symbol
                    / "train"
                    / "slice"
                    / f"df_{index}.feather"
                ),
                "source_output": str(_stage_output(output_root, symbol, "train", contract)),
                "trading_days": contract_days[row_start:row_end],
                "row_start": row_start,
                "row_end": row_end,
            }
        )
        index += 1
        row_start += chunk_length
    return outputs, index


def build_dataset_manifest(
    summary,
    boundaries,
    symbol,
    target_freq,
    start_date,
    end_date,
    input_root,
    feature_union_path,
    output_root,
    chunk_length,
    early_stop,
):
    ranges = {
        "train": (boundaries["start"], boundaries["a"]),
        "valid": (boundaries["a"], boundaries["b"]),
        "test": (boundaries["b"], boundaries["c"]),
    }
    manifest = {
        "symbol": symbol,
        "target_freq": target_freq,
        "split_ratio": {"train": 5, "valid": 3, "test": 2},
        "boundaries": boundaries,
        "state_features_path": str(Path(output_root) / symbol / "state_features.npy"),
        "feature_union_path": str(feature_union_path),
        "sets": {},
    }

    next_slice = 0
    for set_name, (range_start, range_end) in ranges.items():
        set_contracts = []
        skipped = []
        for contract in summary.get("contracts", []):
            contract_name = contract["contract"]
            days = [
                str(day["date"])
                for day in contract.get("trading_days", [])
                if _in_range(day["date"], range_start, range_end)
            ]
            if not days:
                skipped.append(
                    {
                        "contract": contract_name,
                        "reason": f"no trading days in {set_name} range",
                    }
                )
                continue

            record = {
                "contract": contract_name,
                "range": [range_start, range_end],
                "trading_days": days,
                "input_path": str(
                    _input_path(
                        input_root,
                        symbol,
                        contract_name,
                        target_freq,
                        start_date,
                        end_date,
                    )
                ),
                "output_path": str(
                    _stage_output(output_root, symbol, set_name, contract_name)
                ),
            }
            if set_name == "train":
                slices, next_slice = _build_slice_plan(
                    days,
                    output_root,
                    symbol,
                    contract_name,
                    next_slice,
                    chunk_length,
                    early_stop,
                )
                record["slice_outputs"] = slices
            set_contracts.append(record)
        manifest["sets"][set_name] = {
            "range": [range_start, range_end],
            "contracts": set_contracts,
            "skipped_contracts": skipped,
        }
    return manifest


def _date_series(df):
    if "trading_day" in df.columns:
        return pd.to_datetime(df["trading_day"].astype(str)).dt.strftime("%Y-%m-%d")
    if "TradingDay" in df.columns:
        return pd.to_datetime(df["TradingDay"].astype(str)).dt.strftime("%Y-%m-%d")
    if "timestamp" not in df.columns:
        raise ValueError("commodity dataset input missing timestamp column")
    return pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")


def _filter_days(df, trading_days):
    day_set = set(trading_days)
    filtered = df.loc[_date_series(df).isin(day_set)].copy()
    if filtered.empty:
        raise ValueError(
            f"planned trading days produced empty output: {sorted(day_set)}"
        )
    if "timestamp" in filtered.columns:
        filtered = filtered.sort_values("timestamp")
    return filtered.reset_index(drop=True)


def write_stage_datasets(manifest):
    feature_union_path = Path(manifest["feature_union_path"])
    if not feature_union_path.exists():
        raise FileNotFoundError(
            f"Missing feature union state_features.npy: {feature_union_path}"
        )
    state_features_path = Path(manifest["state_features_path"])
    state_features_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(feature_union_path, state_features_path)

    for set_info in manifest["sets"].values():
        contracts_total_count = 0
        for contract in set_info["contracts"]:
            input_path = Path(contract["input_path"])
            if not input_path.exists():
                raise FileNotFoundError(
                    f"Missing df.feather for contract {contract['contract']}: {input_path}"
                )
            df = pd.read_feather(input_path)
            output_df = _filter_days(df, contract["trading_days"])
            output_path = Path(contract["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_feather(output_path)
            contract["output_row_count"] = int(len(output_df))
            contracts_total_count += contract["output_row_count"]
        set_info["contracts_total_count"] = contracts_total_count


def rebuild_train_slice_plan(manifest, chunk_length, early_stop):
    next_index = 0
    for contract in manifest["sets"]["train"]["contracts"]:
        output_path = Path(contract["output_path"])
        slice_dir = output_path.parent / "slice"
        row_count = int(contract.get("output_row_count", 0))
        row_start = 0
        slice_outputs = []
        while row_start < row_count:
            row_end = min(row_start + chunk_length + early_stop, row_count)
            slice_outputs.append(
                {
                    "index": next_index,
                    "contract": contract["contract"],
                    "path": str(slice_dir / f"df_{next_index}.feather"),
                    "source_output": str(output_path),
                    "row_start": row_start,
                    "row_end": row_end,
                }
            )
            next_index += 1
            row_start += chunk_length
        contract["slice_outputs"] = slice_outputs


def write_train_slices(manifest):
    expected_index = 0
    for contract in manifest["sets"]["train"]["contracts"]:
        df = pd.read_feather(contract["output_path"])
        for slice_info in contract.get("slice_outputs", []):
            if "index" in slice_info and int(slice_info["index"]) != expected_index:
                raise ValueError("train slice indices must be continuous")
            row_start = int(slice_info["row_start"])
            row_end = int(slice_info["row_end"])
            sliced = df.iloc[row_start:row_end].reset_index(drop=True)
            if sliced.empty:
                continue
            output_path = Path(slice_info["path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sliced.to_feather(output_path)
            slice_info["output_row_count"] = int(len(sliced))
            expected_index += 1


def run_dataset_generation(
    summary_path,
    input_root,
    feature_union_path,
    output_root,
    symbol,
    target_freq,
    start_date,
    end_date,
    train_ratio=5,
    valid_ratio=3,
    test_ratio=2,
    chunk_length=3200,
    early_stop=320,
):
    summary_path = Path(summary_path)
    with summary_path.open("r", encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    dataset_root = Path(output_root) / symbol
    boundaries = calculate_split_boundaries(
        summary,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )
    manifest = build_dataset_manifest(
        summary=summary,
        boundaries=boundaries,
        symbol=symbol,
        target_freq=target_freq,
        start_date=start_date,
        end_date=end_date,
        input_root=input_root,
        feature_union_path=feature_union_path,
        output_root=output_root,
        chunk_length=chunk_length,
        early_stop=early_stop,
    )
    dataset_root.mkdir(parents=True, exist_ok=True)
    write_stage_datasets(manifest)
    rebuild_train_slice_plan(manifest, chunk_length=chunk_length, early_stop=early_stop)
    write_train_slices(manifest)
    (dataset_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_path", type=Path, required=True)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--feature_union_path", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--target_freq", type=str, required=True)
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    parser.add_argument("--train_ratio", type=int, default=5)
    parser.add_argument("--valid_ratio", type=int, default=3)
    parser.add_argument("--test_ratio", type=int, default=2)
    parser.add_argument("--chunk_length", type=int, default=3200)
    parser.add_argument("--early_stop", type=int, default=320)
    return parser


def main(args=None):
    parsed = build_parser().parse_args(args)
    run_dataset_generation(
        summary_path=parsed.summary_path,
        input_root=parsed.input_root,
        feature_union_path=parsed.feature_union_path,
        output_root=parsed.output_root,
        symbol=parsed.symbol,
        target_freq=parsed.target_freq,
        start_date=parsed.start_date,
        end_date=parsed.end_date,
        train_ratio=parsed.train_ratio,
        valid_ratio=parsed.valid_ratio,
        test_ratio=parsed.test_ratio,
        chunk_length=parsed.chunk_length,
        early_stop=parsed.early_stop,
    )


if __name__ == "__main__":
    main()
