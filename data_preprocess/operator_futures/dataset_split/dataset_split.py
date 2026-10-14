import argparse
import json
import math
from datetime import datetime, time, timedelta
from pathlib import Path

import polars as pl

try:
    from operator_futures.commodity.config import get_commodity_config, TradingSession
except ImportError:
    try:
        from data_preprocess.operator_futures.commodity.config import (
            get_commodity_config,
            TradingSession,
        )
    except ImportError:
        get_commodity_config = None
        TradingSession = None

from collections import namedtuple
_FallbackSession = namedtuple("_FallbackSession", ["start", "end"])


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


def generate_trading_calendar(symbol, trading_days, timestamps):
    """Generate trading calendar manifest with dynamic session activation from actual timestamps."""
    ts_set = set()
    if timestamps:
        for ts in timestamps:
            if isinstance(ts, datetime):
                ts_set.add(ts.replace(tzinfo=None) if ts.tzinfo is not None else ts)

    raw_sessions = None
    if get_commodity_config is not None:
        try:
            cfg = get_commodity_config(symbol)
            raw_sessions = cfg.trading_sessions
        except Exception:
            raw_sessions = None

    if not raw_sessions:
        raw_sessions = (
            _FallbackSession(time(9, 0), time(10, 15)),
            _FallbackSession(time(10, 30), time(11, 30)),
            _FallbackSession(time(13, 30), time(15, 0)),
            _FallbackSession(time(21, 0), time(23, 0)),
        )

    day_sessions = []
    night_session = None
    for s in raw_sessions:
        if s.start >= time(20, 0) or s.start < time(4, 0):
            night_session = s
        else:
            day_sessions.append(s)
    day_sessions.sort(key=lambda s: s.start)

    parsed_dates = []
    for td in trading_days:
        td_str = str(td)
        try:
            d = datetime.strptime(td_str, "%Y-%m-%d").date()
        except ValueError:
            d = datetime.strptime(td_str, "%Y%m%d").date()
        parsed_dates.append(d)
    parsed_dates = sorted(list(set(parsed_dates)))

    calendar_days = []
    prev_date = None
    for cur_date in parsed_dates:
        td_key = cur_date.strftime("%Y%m%d")
        date_str = cur_date.strftime("%Y-%m-%d")

        day_entries = []
        for idx, s in enumerate(day_sessions):
            name = "aft" if s.start >= time(13, 0) else f"m{idx + 1}"
            day_entries.append({
                "name": name,
                "start": f"{date_str} {s.start.strftime('%H:%M:%S')}",
                "end": f"{date_str} {s.end.strftime('%H:%M:%S')}",
            })

        night_entry = None
        if night_session is not None and ts_set:
            open_dt = datetime.combine(cur_date, time(9, 0))
            lower_bound = (
                datetime.combine(prev_date, time(15, 0))
                if prev_date is not None
                else open_dt - timedelta(days=4)
            )
            matching_night_dates = {
                ts.date()
                for ts in ts_set
                if lower_bound < ts < open_dt and (ts.hour >= 20 or ts.hour < 4)
            }
            if matching_night_dates:
                night_date = max(matching_night_dates)
                night_entry = {
                    "name": "night",
                    "start": f"{night_date.strftime('%Y-%m-%d')} {night_session.start.strftime('%H:%M:%S')}",
                    "end": f"{night_date.strftime('%Y-%m-%d')} {night_session.end.strftime('%H:%M:%S')}",
                }

        sessions = [night_entry] + day_entries if night_entry is not None else day_entries
        calendar_days.append({
            "trading_day": td_key,
            "sessions": sessions,
        })
        prev_date = cur_date

    return {
        "symbol": symbol,
        "trading_days": calendar_days,
    }


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


def _input_path_candidates(input_root, symbol, contract, target_freq, start_date, end_date):
    input_dir = Path(input_root) / symbol / contract / target_freq
    date_range = f"{start_date}-{end_date}"
    return [
        input_dir / f"{date_range}.feather",
        input_dir / date_range / "df.feather",
    ]


def _input_path(input_root, symbol, contract, target_freq, start_date, end_date):
    candidates = _input_path_candidates(
        input_root,
        symbol,
        contract,
        target_freq,
        start_date,
        end_date,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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
        set_timestamps = set()
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
            if "timestamp" in output_df.columns:
                ts_series = output_df["timestamp"]
                if ts_series.dtype in (pl.Utf8, pl.String):
                    ts_series = ts_series.str.strptime(
                        pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False
                    )
                set_timestamps.update(ts_series.unique().drop_nulls().to_list())

        if not frames:
            raise ValueError(f"cannot write {set_name}.feather without contract outputs")
        merged = pl.concat(frames, how="vertical")
        merged_output_path = Path(set_info["merged_output_path"])
        merged_output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.write_ipc(merged_output_path)
        set_info["contracts_total_count"] = total_count
        set_info["merged_output_row_count"] = merged.height

        # Generate trading calendar for this set
        set_trading_days = sorted(
            list({day for c in set_info["contracts"] for day in c["trading_days"]})
        )
        set_dir = Path(set_info["merged_output_path"]).parent / set_name
        set_dir.mkdir(parents=True, exist_ok=True)
        calendar_path = set_dir / "calendar_manifest.json"
        calendar_data = generate_trading_calendar(
            symbol=manifest.get("symbol", "fu"),
            trading_days=set_trading_days,
            timestamps=set_timestamps,
        )
        calendar_path.write_text(
            json.dumps(calendar_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        set_info["calendar_manifest_path"] = str(calendar_path)


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
